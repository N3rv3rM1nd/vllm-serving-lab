from __future__ import annotations

import json, re, math
import os, argparse

from typing import Dict, Iterable, FrozenSet, Mapping, NamedTuple, TypeAlias, Any
from run_paths import add_run_args, resolve_run

LabelPairs: TypeAlias = tuple[tuple[str, str], ...]
MetricKey: TypeAlias = tuple[str, LabelPairs]


TARGET_METRICS: FrozenSet[str] = frozenset({  # 冻结字典
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
})
GENERATION = "vllm:generation_tokens_total"
SUCCESS = "vllm:request_success_total"
TTFT_SUM = "vllm:time_to_first_token_seconds_sum"
TTFT_COUNT = "vllm:time_to_first_token_seconds_count"
IDENTITY_ABS_TOL_S = 1e-9
IDENTITY_REL_TOL = 1e-12

class MetricDelta(NamedTuple): # 带字段名和类型标注的不可变元组类
    before: float # 保留before和after可以追溯delta出问题的根源
    after: float
    delta: float

class EngineSummary(NamedTuple):
    generation_tokens_delta: float
    success_by_reason: dict[str, float]
    success_total: float
    ttft_delta_sum: float
    ttft_delta_count: float
    engine_ttft_mean_s: float

class RequestMetrics(NamedTuple):
    batch: int
    req_id: int
    n_out: int 

    t_start_s: float
    t_end_s: float

    raw_ttft_s: float
    ttft_s: float
    ttft_abs_err_s: float

    e2e_s: float
    chunk_span_s: float

    sum_itls_s: float
    span_abs_err_s: float

    mean_chunk_interval_s: float | None
    create_s: float
    drain_s: float

class ClientSummary(NamedTuple):
    batch: int
    n_requests: int
    n_nonempty_chunks: int # chunk和token的区别：
                           # token：引擎端计数 chunk：客户端流解析
    
    ttft_mean_s: float
    ttft_p50_s: float
    ttft_p95_s: float

    e2e_mean_s: float
    e2e_p50_s: float
    e2e_p95_s: float

    request_equal_chunk_interval_mean_s: float # 请求内先平均 再对请求平均
    interval_equal_chunk_interval_mean_s: float # 所有chunk间隔直接平均

    itl_p50_s: float   # 全部间隔合并后取分位：每个间隔等权，不是每个请求等权
    itl_p95_s: float
    itl_p99_s: float
    itl_n_intervals: int  # 池子样本量，用于判断分位数可靠度
    n_nonempty_chunks_s: float # 看chunk吞吐

    request_span_s: float

def percentile(values: Iterable[float], p: float) -> float:
    data = sorted(values)
    if not data:
        raise ValueError(f"percentile requires non-empty values")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"invalid percentile: {p}")

    position = (len(values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return data[lower]
    weight = position - lower
    return data[lower] + weight * (data[upper] - data[lower])

def derive_request_metrics(
    batch: int,
    request: Mapping[str, Any],
) -> RequestMetrics:                # 输出文件函数：传入数据结构和必需字段都必须存在
    req_id = int(request["req_id"]) # 读取的输出文件不应该做防御式读取，避免掩盖了输出文件缺失的bug
    n_out = int(request["n_out"])
    raw_ttft_s = float(request["ttft"])

    window = request["window"]
    if len(window) != 2:
        raise ValueError(
            f"B={batch} req_id={req_id}: window must have 2 timestamps"
        )

    window_start_s = float(window[0])
    window_end_s = float(window[1])

    t_start_s = float(request["t_start"])
    t_create_s = float(request["t_create"])
    t_end_s = float(request["t_end"])

    itls = request["itls"]
    if n_out <= 0:
        raise ValueError(
            f"B={batch} req_id={req_id}: n_out must be positive"
        )

    if len(itls) != n_out - 1:
        raise ValueError(
            f"B={batch} req_id={req_id}:"
            f"len(itls)={len(itls)}, expected={n_out-1}"
        )

    if any(interval <= 0 for interval in itls):
        raise ValueError(
            f"B={batch} req_id={req_id}: non-positive interval exists"
        )

    timestamps = (
        t_start_s,
        t_create_s,
        window_start_s,
        window_end_s,
        t_end_s,
    )
    if not all(math.isfinite(value) for value in timestamps):
        raise ValueError(
            f"B={batch} req_id={req_id}: non-finite stamps exists"
        )
    if not (t_start_s <= t_create_s <= window_start_s <= window_end_s <= t_end_s):
        raise ValueError(
            f"B={batch} req_id={req_id}: timestamps timeline is wrong"
        )
    ttft_s = window_start_s - t_start_s
    e2e_s = t_end_s - t_start_s
    chunk_span_s = window_end_s - window_start_s
    sum_itls_s = float(sum(itls))

    if not math.isclose(
        ttft_s, 
        raw_ttft_s,
        rel_tol=IDENTITY_REL_TOL,
        abs_tol=IDENTITY_ABS_TOL_S,
    ):
        raise ValueError(
            f"B={batch} req_id={req_id}: TTFT identity failed"
            f"recaculated={ttft_s}, raw={raw_ttft_s}"
        )
    if not math.isclose(
        sum_itls_s,
        chunk_span_s,
        rel_tol=IDENTITY_REL_TOL,
        abs_tol=IDENTITY_ABS_TOL_S,
    ):
        raise ValueError(
            f"B={batch} req_id={req_id}: chunk_span_s identity failed"
            f"recaculated={sum_itls_s}, raw={chunk_span_s}"
        )
    mean_chunk_interval_s = None
    if n_out > 1:
        mean_chunk_interval_s = chunk_span_s / (n_out - 1)

    return RequestMetrics(
        batch=batch,
        req_id=req_id,
        n_out=n_out,
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        raw_ttft_s=raw_ttft_s,
        ttft_s=ttft_s,
        ttft_abs_err_s=abs(ttft_s - raw_ttft_s),
        e2e_s=e2e_s,
        chunk_span_s=chunk_span_s,
        sum_itls_s=sum_itls_s,
        span_abs_err_s=abs(chunk_span_s - sum_itls_s),
        mean_chunk_interval_s=mean_chunk_interval_s,
        create_s=t_create_s - t_start_s,
        drain_s=t_end_s - window_end_s,
    )
    
def validate_engine_request_count(
        summary: EngineSummary,
        expect_count: int,
) -> None:
    if not math.isclose(
        summary.success_total,
        expect_count,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"success_total isn't equal to expect_count"
        )
    if not math.isclose(
        summary.ttft_delta_count,
        expect_count,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"ttft_delta_count isn't equal to expect_count"
        )

def _parse_labels(src: str, *, path: str, lineno: int) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    i, n = 0, len(src)
    while i < n:
        while i < n and src[i] in " \t":
            i += 1
        if i >= n:
            break

        start = i
        while i < n and src[i] not in "= \t":
            i += 1
        key = src[start:i]
        if not key:
            raise ValueError(f"{path}:{lineno}: the key of label is empty: {src!r}")

        while i < n and src[i] in " \t":
            i += 1
        if i >= n or src[i] != "=":
            raise ValueError(
                f"{path}:{lineno}: label {key!r} is lack of = : {src[i]}"
            )
        i += 1
        while i < n and src[i] in " \t":
            i += 1

        if i >= n or src[i] != '"':
            raise ValueError(
                f"{path}:{lineno}: label {key!r} is lack of {'"'}: {src[i]}"
            )
        i += 1

        chars = []
        closed = False
        while i < n:
            ch = src[i]
            if ch == "\\":
                if i + 1 >= n:
                    raise ValueError(
                        f"{path}:{lineno}: label {key!r} escaping is incomplete"
                    )
                nxt = src[i + 1]
                if nxt == "n":
                    chars.append("\n")
                else:
                    chars.append(nxt)
                i += 2
            elif ch == '"':
                i += 1
                closed = True
                break
            else:
                chars.append(ch)
                i += 1
        if not closed:
            raise ValueError(
                f"{path}:{lineno}: the label {key!r} quotation isn't closed"
            )
        if not labels.get(key):
            labels[key] = "".join(chars) 
        else:
            raise ValueError(
                f"duplication operation: {key} is already here."
            )

        while i < n and src[i] in " \t":
            i += 1
        if i < n and src[i] == ",":
            i += 1
    return labels

def _quote(v: str) -> str: # Prometheus 文本
    return ('"' + v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"') + '"')

def make_metric_key(name: str, labels: Mapping[str, str]) -> MetricKey:
    return name, tuple(sorted(labels.items()))

def format_metric_key(key: MetricKey) -> str: # 身份和完整的信息分开保存防止反复转换
    name, labels = key
    if not labels:
        return name

    inner = ",".join(
        f"{label_name}={_quote(label_value)}"
        for label_name, label_value in labels
    )
    return f"{name}{{{inner}}}"

def metric_name(key: MetricKey) -> str:
    return key[0]

def get_label(key: MetricKey, wanted: str) -> str | None:
    for label_name, label_value in key[1]:
        if label_name == wanted:
            return label_value
    return None

def parser(path: str, target_names: Iterable[str]) -> Dict[MetricKey, float]:
    targets = frozenset(target_names)
    result: Dict[MetricKey, float] = {}

    with open(path, mode='r', encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"): # 注释行跳过
                continue

            try:
                left, value_str = line.rsplit(None, 1)
            except ValueError as e:
                raise ValueError(
                    f"{path}:{lineno}: can't seperate the value from the raw{line!r}"
                ) from e

            brace = left.find("{")
            name = (
                left[:brace].strip()
                if brace >= 0
                else left.strip()
            )

            if not name:
                raise ValueError(
                    f"{path}:{lineno}: empty metric name"
                )

            if name not in targets:
                continue
            
            if brace >= 0:
                tail = left[brace + 1:].strip()
                if not tail.endswith("}"):
                    raise ValueError(
                        f"{path}:{lineno}: label is lack of end: {left!r}"
                    )
                labels = _parse_labels(tail[:-1], path=path, lineno=lineno)
            else:
                labels = {}

            try:
                value = float(value_str)
            except ValueError as e:
                raise ValueError(
                    f"{path}:{lineno}: can't analysis {value_str!r}"
                ) from e

            key = make_metric_key(name, labels)

            if key in result:
                raise ValueError(
                    f"{path}:{lineno}: duplicate series"
                    f"{format_metric_key(key)}"
                )
            
            result[key] = value
    return result

def diff(before: Mapping[MetricKey, float], 
         after: Mapping[MetricKey, float]
         ) -> dict[MetricKey, MetricDelta]:
    # 在进入循环之前先检查两者键的差距否则 进入循环之后报错信息不完整 只有一方能被检测到
    before_only = before.keys() - after.keys()
    after_only = after.keys() - before.keys()

    if before_only or after_only:
        details = []
        if before_only:
            details.append(
                "only in before:"
                + ",".join(
                    format_metric_key(key)
                    for key in before_only
                )
            )
        if after_only:
            details.append(
                "only in after:"
                + ",".join(
                    format_metric_key(key)
                    for key in after_only
                )
            )
        raise ValueError("snapshot key mismatch;" + ",".join(details))

    result: dict[MetricKey, MetricDelta] = {}

    for key in before:
        before_value = before[key]
        after_value = after[key]
        delta = after_value - before_value

        if delta < -1e-12: # 负容差
            raise ValueError(
                "counter decreased for"
                f"{format_metric_key(key)}"
                f"{before_value}-{after_value}"
            )
        result[key] = MetricDelta(
            before=before_value,
            after=after_value,
            delta=delta,
        )

    return result

def summarize_engine(
    diffs: Mapping[MetricKey, MetricDelta], # Mapping 只读
) -> EngineSummary:
    generation_values = [
        record.delta
        for key, record in diffs.items()
        if metric_name(key) == GENERATION
    ]
    if len(generation_values) != 1:
        raise ValueError(
            f"expected exactly one {GENERATION} series,"
            f"got {len(generation_values)}"
        )

    success_by_reason: dict[str, float] = {} # 只有delta部分
    for key, value in diffs.items():
        if metric_name(key) != SUCCESS:
            continue

        reason = get_label(key, "finished_reason")
        if reason is None:
            raise ValueError(
                f"{SUCCESS} has no finished_reason"
                f"{format_metric_key(key)}"
            )

        success_by_reason[reason] = value.delta # 函数内部应该和全局变量区分开

    if not success_by_reason:
        raise ValueError(f"no {SUCCESS} series found")

    ttft_sum_by_labels = {
        key[1]: record.delta
        for key, record in diffs.items()
        if metric_name(key) == TTFT_SUM
    }
    ttft_count_by_labels = {
        key[1]: record.delta
        for key, record in diffs.items()
        if metric_name(key) == TTFT_COUNT
    }

    if ttft_sum_by_labels.keys() != ttft_count_by_labels.keys():
        raise ValueError("TTFT sum/count labels don't match")
    if len(ttft_sum_by_labels) != 1:
        raise ValueError(
            "expected exactly one TTFT sum/count pair, "
            f"got {len(ttft_sum_by_labels)}"
        )

    ttft_labels = next(iter(ttft_sum_by_labels)) # 取出第一个可迭代的元素，
    ttft_delta_sum = ttft_sum_by_labels[ttft_labels]
    ttft_delta_count = ttft_count_by_labels[ttft_labels]

    if ttft_delta_count <= 0:
        raise ValueError(
            f"TTFT count must be positive, got {ttft_delta_count}"
        )
    return EngineSummary(
        generation_tokens_delta=generation_values[0],
        success_by_reason=success_by_reason,
        success_total=sum(success_by_reason.values()),
        ttft_delta_sum=ttft_delta_sum,
        ttft_delta_count=ttft_delta_count,
        engine_ttft_mean_s=ttft_delta_sum / ttft_delta_count,
    )

def summarize_client(
        batch: int,
        records: list[RequestMetrics],
        itl_pool: list[float],
) -> ClientSummary:
    if not records:
        raise ValueError(f"B={batch}: no request records")

    ttft_values = [record.ttft_s for record in records]
    e2e_values = [record.e2e_s for record in records]

    interval_records = [
        record
        for record in records
        if record.n_out > 1
        and record.mean_chunk_interval_s is not None
    ]
    if not interval_records:
        raise ValueError(f"B={batch}: no valid chunk intervals")

    request_mean_intervals = [
        record.mean_chunk_interval_s
        for record in interval_records
    ]

    request_equal_mean = (
        sum(request_mean_intervals) / len(request_mean_intervals)
    )

    total_itls = sum(record.sum_itls_s for record in interval_records)
    total_interval_count = sum(record.n_out - 1 for record in interval_records)
    interval_equal_mean = total_itls / total_interval_count

    # 池子必须与 records 同源：条数与总时长双向对账
    if not itl_pool:
        raise ValueError(f"B={batch}: empty itl pool")
    expected_count = sum(record.n_out - 1 for record in records)
    if len(itl_pool) != expected_count:
        raise ValueError(
            f"B={batch}: itl pool size {len(itl_pool)}"
            f" != sum(n_out-1) {expected_count}"
        )
    if not math.isclose(sum(itl_pool), total_itls, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"B={batch}: itl pool sum {sum(itl_pool):.9f}"
            f" != sum_itls {total_itls:.9f}"
        )

    t_start_min = min(record.t_start_s for record in records)
    t_end_max = max(record.t_end_s for record in records)

    return ClientSummary(
        batch=batch,
        n_requests=len(records),
        n_nonempty_chunks=sum(record.n_out for record in records),
        ttft_mean_s=sum(ttft_values) / len(ttft_values),
        ttft_p50_s=percentile(ttft_values, 0.5),
        ttft_p95_s=percentile(ttft_values, 0.95),
        e2e_mean_s=sum(e2e_values) / len(e2e_values),
        e2e_p50_s=percentile(e2e_values, 0.5),
        e2e_p95_s=percentile(e2e_values, 0.95),
        itl_p50_s=percentile(itl_pool, 0.5),
        itl_p95_s=percentile(itl_pool, 0.95),
        itl_p99_s=percentile(itl_pool, 0.99),
        itl_n_intervals=len(itl_pool),
        request_equal_chunk_interval_mean_s=request_equal_mean,
        interval_equal_chunk_interval_mean_s=interval_equal_mean,
        n_nonempty_chunks_s=sum(record.n_out for record in records) / (t_end_max - t_start_min),
        request_span_s=t_end_max - t_start_min,
    )

if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    add_run_args(arg_parse)
    args = arg_parse.parse_args()

    run = resolve_run(
        label=args.label,
        request_dir=args.request_dir,
        metrics_dir=args.metrics_dir,
        out_dir=args.out_dir,
    )

    client_records_by_batch: dict[int, list[RequestMetrics]] = {}
    itl_pool_by_batch: dict[int, list[float]] = {}
    summary_client: dict[int, ClientSummary] = {}

    for job in run.jobs:
        with job.request_path.open("r", encoding="utf-8") as file:
            request_data = json.load(file)
        if int(request_data["B"]) != job.batch:
            raise ValueError(
                f"{job.request_path}: B={request_data['B']} != {job.batch}"
            )
        requests = request_data.get("requests", [])
        if len(requests) != job.n_requests:
            raise ValueError(
                f"{job.request_path}:"
                f"{len(requests)} requests, expected {job.n_requests}"
            )
        
        client_records = [
            derive_request_metrics(batch=job.batch, request=request)
            for request in requests
        ]
        req_ids = [record.req_id for record in client_records]
        if len(set(req_ids)) != len(req_ids):
            raise ValueError(f"B={job.batch}: duplicate req_id")

        client_records_by_batch[job.batch] = client_records
        itl_pool_by_batch[job.batch] = [
            itl for request in requests for itl in (request.get("itls") or [])
        ]

    for job in run.jobs:
        summary_client[job.batch] = summarize_client(
            job.batch,
            client_records_by_batch[job.batch],
            itl_pool_by_batch[job.batch],
        )

    run.out_dir.mkdir(parents=True, exist_ok=True)

    with (run.out_dir / "result_Bn.json").open(  # Path 内部方法调用
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                batch: summary._asdict()
                for batch, summary in summary_client.items()
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    request_rows = [
        record._asdict()
        for job in run.jobs
        for record in client_records_by_batch[job.batch]
    ]
    with (run.out_dir / "result_Bn_requests.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(request_rows, file, ensure_ascii=False, indent=2)

    print(f"{run.label}: {len(run.jobs)} batches -> {run.out_dir}")
    
            
    


