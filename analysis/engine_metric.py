import json, os, argparse, logging 

from pathlib import Path
from result_Bn_check import summarize_engine, parser, diff, format_metric_key, EngineSummary, MetricKey, MetricDelta
from typing import NamedTuple, FrozenSet, Any
from result_Bn_check import metric_name
from run_paths import add_run_args, resolve_run

CONFIG_DIR = Path(__file__).resolve().parent.parent / 'raw' / 'config'
TARGET_NAMES: FrozenSet[str] = frozenset({
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_queue_time_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
    "vllm:request_inference_time_seconds_sum",
    "vllm:request_inference_time_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:prompt_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:prefix_cache_hits_total", # token
    "vllm:prefix_cache_queries_total", # token
    "vllm:prompt_tokens_cached_total", # token
})


class EngineDetail(NamedTuple):
    batch: int
    ttft_ms: float
    queue_ms: float
    prefill_ms: float
    decode_ms: float
    inference_ms: float
    e2e_ms: float
    generation_tokens: float
    prompt_tokens: float
    preemptions: float
    pc_queries_tokens: float | None
    pc_hits_tokens: float | None
    pc_hit_rate: float | None
    pc_cached_tokens: float | None
    pc_cached_token_ratio: float | None
    pc_hits_before: float | None  # before 快照绝对值（含 warmup），用于推导底噪
    pc_queries_before: float | None
    


'''
- vllm:request_queue_time_seconds_{sum,count}
- vllm:request_prefill_time_seconds_{sum,count}
- vllm:request_decode_time_seconds_{sum,count}
- vllm:request_inference_time_seconds_{sum,count}
- vllm:e2e_request_latency_seconds_{sum,count}
- vllm:prompt_tokens_total、vllm:num_preemptions_total
'''

_MISSING: set[str] = set()


def _series(diffs, name) -> list[float]:
    return [rec.delta for key, rec in diffs.items() if metric_name(key) == name]


def _delta_required(diffs, name: str) -> float:
    """核心指标：缺失或多序列都属于数据异常，直接抛错。"""
    vals = _series(diffs, name)
    if len(vals) == 0:
        raise ValueError(f"{name}: 指标不存在（检查采集名单或引擎版本）")
    if len(vals) > 1:
        raise ValueError(
            f"{name}: 期望 1 条序列，实际 {len(vals)}（禁止静默取值）"
        )
    return vals[0]


def _delta_optional(diffs, name: str) -> float | None:
    """可选指标：缺失返回 None，且只记录一次，避免刷屏。"""
    vals = _series(diffs, name)
    if len(vals) == 0:
        _MISSING.add(name)
        return None
    if len(vals) > 1:
        raise ValueError(
            f"{name}: 期望 1 条序列，实际 {len(vals)}（禁止静默取值）"
        )
    return vals[0]


def _abs_value(before: dict, name: str) -> float | None:
    """取绝对量（非差分），同样要求唯一序列。"""
    vals = [v for k, v in before.items() if metric_name(k) == name]
    if len(vals) == 0:
        return None
    if len(vals) > 1:
        raise ValueError(f"{name}: 期望 1 条序列，实际 {len(vals)}（禁止静默取值）")
    return vals[0]


def _mean_ms(diffs, family, expect_count):
    s, c = _delta_required(diffs, family + "_sum"), _delta_required(diffs, family + "_count")
    if c != expect_count:
        raise ValueError(f"{family}: count {c} != {expect_count}")
    return s / c * 1000.0

def summarize_engine_detail(diffs: dict[MetricKey, MetricDelta],
                            before: dict,
                            batch: int, expect_count: int) -> EngineDetail:

    ttft_ms = _mean_ms(diffs, "vllm:time_to_first_token_seconds", expect_count)
    queue_ms = _mean_ms(diffs, "vllm:request_queue_time_seconds", expect_count)
    prefill_ms = _mean_ms(diffs, "vllm:request_prefill_time_seconds", expect_count)
    decode_ms = _mean_ms(diffs, "vllm:request_decode_time_seconds", expect_count)
    inference_ms = _mean_ms(diffs, "vllm:request_inference_time_seconds", expect_count)
    e2e_ms = _mean_ms(diffs, "vllm:e2e_request_latency_seconds", expect_count)
    generation_tokens = _delta_required(diffs, "vllm:generation_tokens_total")
    prompt_tokens     = _delta_required(diffs, "vllm:prompt_tokens_total")
    preemptions       = _delta_required(diffs, "vllm:num_preemptions_total")

    pc_queries_tokens = _delta_optional(diffs, "vllm:prefix_cache_queries_total")
    pc_hits_tokens    = _delta_optional(diffs, "vllm:prefix_cache_hits_total")
    pc_cached_tokens  = _delta_optional(diffs, "vllm:prompt_tokens_cached_total")
    pc_hits_before    = _abs_value(before, "vllm:prefix_cache_hits_total")
    pc_queries_before = _abs_value(before, "vllm:prefix_cache_queries_total")

    # 0 是合法值：只能用 None / 分母判断，不能用真值判断
    if pc_queries_tokens is None or pc_queries_tokens == 0:
        pc_hit_rate = None
    else:
        pc_hit_rate = pc_hits_tokens / pc_queries_tokens

    if prompt_tokens == 0:
        pc_cached_token_ratio = None
    else:
        pc_cached_token_ratio = pc_cached_tokens / prompt_tokens

    return EngineDetail(
        batch=batch,
        ttft_ms=ttft_ms,
        queue_ms=queue_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        inference_ms=inference_ms,
        e2e_ms=e2e_ms,
        generation_tokens=generation_tokens,
        prompt_tokens=prompt_tokens,
        preemptions=preemptions,
        pc_queries_tokens=pc_queries_tokens,
        pc_hits_tokens=pc_hits_tokens,
        pc_hit_rate=pc_hit_rate,
        pc_cached_tokens=pc_cached_tokens,
        pc_cached_token_ratio=pc_cached_token_ratio,
        pc_hits_before=pc_hits_before,
        pc_queries_before=pc_queries_before,
    )

def get_engine(
    before_path: Path,
    after_path: Path,
    batch: int,
    expect_count: int,
) -> EngineDetail:
    before_data = parser(str(before_path), TARGET_NAMES)
    after_data = parser(str(after_path), TARGET_NAMES)

    diffs = diff(before_data, after_data)

    return summarize_engine_detail(diffs, before_data, batch, expect_count)

def load_meta(config: str) -> dict:
    return json.loads((CONFIG_DIR / f"{config}__sweep_meta.json").read_text())

def dicard_main():
    meta = load_meta("baseline")
    expect_count = int(meta["N"])
    dir = Path(__file__).resolve().parent.parent
    path = str(dir) + "/raw/metrics/baseline"
    out: dict[int, EngineDetail] = {}
    with open(dir / 'results' / 'result_Bn.json', 'r') as f:
        client = json.load(f)
    for b in meta["B_list"]:
        engine_summary = get_engine(path, b, expect_count) # NameTuple类落地没有指标名字只能通过索引访问
        out[b] = {key: getattr(engine_summary, key) for key in engine_summary._fields}
        
        out[b]["tokens_s"] = out[b]["generation_tokens"] / client[str(b)]["request_span_s"]
    os.makedirs(dir / 'results', exist_ok=True)
    with open(dir / 'results' / 'engine_Bn.json', 'w') as f:
        json.dump(out, fp=f, indent=2)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    add_run_args(arg_parser)
    args = arg_parser.parse_args()

    run = resolve_run(
        label=args.label,
        request_dir=args.request_dir,
        metrics_dir=args.metrics_dir,
        out_dir=args.out_dir,
    )

    with (run.out_dir / "result_Bn.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        client = json.load(file)

    out: dict[int, dict[str, Any]] = {}

    for job in run.jobs:
        engine_summary = get_engine(
            job.before_path,
            job.after_path,
            job.batch,
            job.n_requests,
        )

        record = {
            key: getattr(engine_summary, key)
            for key in engine_summary._fields
        }
        record["tokens_s"] = (
            record["generation_tokens"]
            / client[str(job.batch)]["request_span_s"]
        )
        out[job.batch] = record

    run.out_dir.mkdir(parents=True, exist_ok=True)

    with (run.out_dir / "engine_Bn.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(out, file, ensure_ascii=False, indent=2)

    if _MISSING:
        print(f"{run.label}: 缺失可选指标 {sorted(_MISSING)}")
    print(f"{run.label}: engine -> {run.out_dir}")