#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 / Week 1 baseline sweep:  B in {1,4,8,16,32}  -> TTFT / TPOT / 吞吐 / 显存.

为什么另起一个文件（而不是改 run.py）:
  - run.py 是 Phase 0 的交付物，保持原样。
  - 本文件 *复用* run.py 的测量内核（VLLMService / warmup / make_prompts /
    stream_one / run_benchmark / aggregate），客户端 TTFT/ITL/TPOT/吞吐的算式
    与你 Phase 0 验证过的逐字一致。
  - 只补 3 个"会污染基线"的阻塞 + Phase 0 欠账：
      (1) --max-num-seqs 固定为常量（不再 == B）      -> 保证"只调 B"
      (2) 压测前后各抓一次 /metrics 做差分            -> 引擎侧 TTFT/TPOT/吞吐（口径对账）
          + 原始文本落盘（后续可算 p95/直方图）
      (3) 显存只读被 pin 的那张卡（idle / after-load / peak），并记录 CUDA_VISIBLE_DEVICES
      (+) vllm.log 每档改名留痕；env 补 torch / CUDA / CUDA_VISIBLE_DEVICES

跑法（在 VLLMService/ 目录下）:
    export PATH="$HOME/.local/bin:$PATH"
    export HF_HUB_OFFLINE=1
    nohup .venv/bin/python -u baseline_sweep.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --S 1024 --N 64 --max_tokens 512 \
        --B_list 1,4,8,16,32 \
        --gpu_memory_utilization 0.55 \
        --fixed_max_num_seqs 64 \
        > sweep_run.log 2>&1 &
    tail -f sweep_run.log
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pynvml
from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run as R  # noqa: E402  Phase 0 的测量内核


# ------------------------------------------------------------------ GPU 工具
def _h(idx: int):
    return pynvml.nvmlDeviceGetHandleByIndex(idx)


def gpu_used_mib(idx: int) -> int:
    return pynvml.nvmlDeviceGetMemoryInfo(_h(idx)).used // 1024 ** 2


def gpu_total_mib(idx: int) -> int:
    return pynvml.nvmlDeviceGetMemoryInfo(_h(idx)).total // 1024 ** 2


def gpu_name(idx: int) -> str:
    return pynvml.nvmlDeviceGetName(_h(idx))


def all_gpus_used() -> list:
    return [gpu_used_mib(i) for i in range(pynvml.nvmlDeviceGetCount())]


def pick_gpu() -> tuple:
    best, best_used = 0, None
    for i in range(pynvml.nvmlDeviceGetCount()):
        u = gpu_used_mib(i)
        print(f"[gpu] physical {i} ({gpu_name(i)}): {u} MiB used", flush=True)
        if best_used is None or u < best_used:
            best, best_used = i, u
    return best, int(best_used)


# ---------------------------------------------------------------- /metrics
async def scrape_metrics(host: str, port: int) -> str:
    async with httpx.AsyncClient(timeout=5) as c:
        r = await c.get(f"http://{host}:{port}/metrics")
        r.raise_for_status()
        return r.text


def parse_metrics(text: str) -> dict:
    """name -> 各 label 求和（跳过 # HELP / # TYPE）。直方图拆成 _sum/_count/_bucket。"""
    out: dict = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        sp = line.rfind(" ")
        if sp < 0:
            continue
        key, val = line[:sp], line[sp + 1:]
        try:
            v = float(val)
        except ValueError:
            continue
        name = key.split("{", 1)[0]
        out[name] = out.get(name, 0.0) + v
    return out


def _d(before: dict, after: dict, name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def engine_side(before: dict, after: dict, wall: float) -> dict:
    """引擎 /metrics 差分；累计量取本次运行的增量。"""
    def mean_of_hist(base: str):
        cnt = _d(before, after, base + "_count")
        s = _d(before, after, base + "_sum")
        return ((s / cnt) if cnt else None), cnt

    ttft_mean, ttft_n = mean_of_hist("vllm:time_to_first_token_seconds")
    tpot_mean, tpot_n = mean_of_hist("vllm:request_time_per_output_token_seconds")
    e2e_mean, e2e_n = mean_of_hist("vllm:e2e_request_latency_seconds")
    itl_mean, _ = mean_of_hist("vllm:inter_token_latency_seconds")
    queue_mean, _ = mean_of_hist("vllm:request_queue_time_seconds")
    prefill_mean, _ = mean_of_hist("vllm:request_prefill_time_seconds")
    decode_mean, _ = mean_of_hist("vllm:request_decode_time_seconds")

    # 注意：prometheus_client 给 Counter 统一加 _total 后缀（见 vllm/v1/metrics/loggers.py）
    gen_tokens = _d(before, after, "vllm:generation_tokens_total")
    prompt_tokens = _d(before, after, "vllm:prompt_tokens_total")
    return {
        "engine_ttft_mean_s": ttft_mean,
        "engine_ttft_n": ttft_n,
        "engine_tpot_mean_s": tpot_mean,
        "engine_tpot_n": tpot_n,
        "engine_itl_mean_s": itl_mean,
        "engine_e2e_mean_s": e2e_mean,
        "engine_e2e_n": e2e_n,
        "engine_queue_mean_s": queue_mean,
        "engine_prefill_mean_s": prefill_mean,
        "engine_decode_mean_s": decode_mean,
        "engine_generation_tokens": gen_tokens,
        "engine_prompt_tokens": prompt_tokens,
        "engine_token_per_s": (gen_tokens / wall) if wall else None,
        "engine_num_preemptions": _d(before, after, "vllm:num_preemptions_total"),
    }


def engine_gauges(after: dict) -> dict:
    keys = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
    ]
    return {k: after.get(k) for k in keys}


# -------------------------------------------------------------- 显存采样器
async def _sample_mem(idx: int, sink: list, interval: float = 0.5):
    while True:
        sink.append(gpu_used_mib(idx))
        await asyncio.sleep(interval)


# 运行期需要跟踪的指标（Prometheus 名 -> CSV 列名）
TS_FIELDS = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:kv_cache_usage_perc": "kv_usage_perc",
    "vllm:num_preemptions_total": "preemptions_total",
}


async def _sample_metrics_ts(host: str, port: int, out_path: Path,
                             interval: float = 1.0) -> None:
    """按固定间隔抓 /metrics 落成 CSV，用于画运行期时间线。

    每行立即 flush：进程被 kill 时已采到的点仍保留。
    """
    cols = list(TS_FIELDS.values())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("unix_ts," + ",".join(cols) + "\n")
        f.flush()
        while True:
            try:
                text = await scrape_metrics(host, port)
                parsed = parse_metrics(text)
                vals = [f"{parsed.get(name, float('nan')):.6f}"
                        for name in TS_FIELDS]
                f.write(f"{time.time():.3f}," + ",".join(vals) + "\n")
                f.flush()
            except (httpx.HTTPError, OSError, ValueError, asyncio.TimeoutError):
                pass          # 采样失败不打断压测，下一拍再试
            await asyncio.sleep(interval)


# ------------------------------------------------------------------ 单档跑
async def run_one_level(B: int, args, sweep_dir: Path, tag: str) -> dict:
    port = args.port
    gpu = args.gpu
    extra = [
        "--max-num-seqs", str(args.fixed_max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    # 透传任意引擎参数（扫参用）：如 --engine_args "--max-num-batched-tokens 2048"
    if getattr(args, "engine_args", ""):
        import shlex
        extra += shlex.split(args.engine_args)
    ns = SimpleNamespace(
        model=args.model, S=args.S, N=args.N, B=B,
        max_tokens=args.max_tokens, vocab_size=args.vocab_size,
        temperature=args.temperature, top_p=args.top_p,
        prefix_len=args.prefix_len, prefix_ratio=args.prefix_ratio,
        prefix_seed=args.prefix_seed, block_size=args.block_size,
        path=str(sweep_dir),
    )
    svc = R.VLLMService(args.model, port, host=args.host,
                        ready_timeout=args.ready_timeout, extra_args=extra)

    res: dict = {
        "tag": tag, "B": B, "port": port, "ok": False,
        "gpus_used_before": all_gpus_used(),
    }
    mem_idle = gpu_used_mib(gpu)
    mem_sink: list = []
    sampler = None
    ts_sampler = None
    t0 = time.perf_counter()
    try:
        async with svc:
            client = AsyncOpenAI(api_key="EMPTY",
                                 base_url=f"http://127.0.0.1:{port}/v1")
            await R.warmup(client, ns, n_warm=args.n_warm)
            mem_after_load = gpu_used_mib(gpu)

            before_txt = await scrape_metrics(args.host, port)
            (sweep_dir / f"metrics_{tag}_before.txt").write_text(
                before_txt, encoding="utf-8")
            before = parse_metrics(before_txt)

            sampler = asyncio.create_task(_sample_mem(gpu, mem_sink))
            ts_sampler = asyncio.create_task(
                _sample_metrics_ts(args.host, port,
                                   sweep_dir / f"timeseries_{tag}.csv")
            )
            record = await R.run_benchmark(client, ns)
            for task in (sampler, ts_sampler):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            sampler = None
            ts_sampler = None

            after_txt = await scrape_metrics(args.host, port)
            (sweep_dir / f"metrics_{tag}_after.txt").write_text(
                after_txt, encoding="utf-8")
            after = parse_metrics(after_txt)

            wall = record["system"]["wall_time"]
            res.update({
                "ok": True,
                "t_end_iso": datetime.now().isoformat(),
                "record": record,
                "engine": engine_side(before, after, wall),
                "engine_gauges_after": engine_gauges(after),
                "mem_idle_mib": mem_idle,
                "mem_after_load_mib": mem_after_load,
                "mem_peak_mib": max(mem_sink) if mem_sink else None,
                "mem_samples": len(mem_sink),
                "gpus_used_after": all_gpus_used(),
            })
    except BaseException as e:  # noqa: BLE001
        res["error"] = repr(e)
    finally:
        if ts_sampler is not None:
            ts_sampler.cancel()
            try:
                await ts_sampler
            except BaseException:
                pass
        if sampler is not None:
            sampler.cancel()
            try:
                await sampler
            except BaseException:
                pass
        # 留痕：run.py 用的是固定名 vllm.log，每档改名，避免互相覆盖
        try:
            logp = HERE / "vllm.log"
            if logp.exists():
                dst = sweep_dir / f"vllm_{tag}.log"
                if dst.exists():
                    dst.unlink()
                logp.replace(dst)
        except OSError:
            pass
    res["wall_total_s"] = time.perf_counter() - t0
    return res


# --------------------------------------------------------------- 汇总输出
CSV_FIELDS = [
    "tag", "B", "ok",
    "ttft_client_mean_s", "ttft_client_p50_s", "ttft_client_p95_s",
    "tpot_client_mean_s", "itl_client_mean_s", "itl_client_p95_s",
    "throughput_client_tok_s", "request_per_s", "speedup", "wall_time_s",
    "engine_ttft_mean_s", "engine_tpot_mean_s", "engine_token_per_s",
    "engine_queue_mean_s", "engine_prefill_mean_s",
    "mem_idle_mib", "mem_after_load_mib", "mem_peak_mib",
    "errs",
]


def _row(r: dict) -> dict:
    if not r.get("ok"):
        return {"tag": r.get("tag"), "B": r.get("B"), "ok": False,
                "errs": r.get("error")}
    rec = r["record"]
    lat, sysd, eng = rec["latency"], rec["system"], r["engine"]
    return {
        "tag": r.get("tag"), "B": r["B"], "ok": True,
        "ttft_client_mean_s": lat["ttft_mean"],
        "ttft_client_p50_s": lat["ttft_p50"],
        "ttft_client_p95_s": lat["ttft_p95"],
        "tpot_client_mean_s": lat["tpot_mean"],
        "itl_client_mean_s": lat["itl_mean"],
        "itl_client_p95_s": lat["itl_p95"],
        "throughput_client_tok_s": sysd["token_per_s"],
        "request_per_s": sysd["request_per_s"],
        "speedup": sysd["speedup"],
        "wall_time_s": sysd["wall_time"],
        "engine_ttft_mean_s": eng.get("engine_ttft_mean_s"),
        "engine_tpot_mean_s": eng.get("engine_tpot_mean_s"),
        "engine_token_per_s": eng.get("engine_token_per_s"),
        "engine_queue_mean_s": eng.get("engine_queue_mean_s"),
        "engine_prefill_mean_s": eng.get("engine_prefill_mean_s"),
        "mem_idle_mib": r.get("mem_idle_mib"),
        "mem_after_load_mib": r.get("mem_after_load_mib"),
        "mem_peak_mib": r.get("mem_peak_mib"),
        "errs": len(rec.get("error") or []),
    }


def _fmt(v, n=4):
    if v is None or v == "":
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def write_summary(rows, args, sweep_dir: Path):
    out = [_row(r) for r in rows]
    with open(sweep_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for ro in out:
            w.writerow({k: ro.get(k, "") for k in CSV_FIELDS})

    lines = []
    lines.append(f"# Baseline sweep {args.sweep_id}\n")
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- 固定: S_in={args.S}, N_req={args.N}, max_tokens(S_out)={args.max_tokens}, "
                 f"temperature={args.temperature}, top_p={args.top_p}")
    lines.append(f"- 引擎参数(全程不变): `--max-num-seqs {args.fixed_max_num_seqs}`, "
                 f"`--gpu-memory-utilization {args.gpu_memory_utilization}` "
                 f"(prefix-caching 未开, max-model-len 默认)")
    lines.append(f"- GPU: physical index {args.gpu} ({gpu_name(args.gpu)}), "
                 f"CUDA_VISIBLE_DEVICES={args.gpu}")
    lines.append(f"- 只调 B: {args.B_list}\n")

    lines.append("## 表：B × 四量\n")
    hdr = ("| B | TTFT(client) mean/p50/p95 s | TTFT(engine) s | TPOT(client) s | "
           "TPOT(engine) s | 吞吐(client) tok/s | 吞吐(engine) tok/s | 显存 peak MiB |")
    lines.append(hdr)
    lines.append("|" + "---|" * 8)
    for ro in out:
        if not ro.get("ok"):
            lines.append(f"| {ro.get('B')} | FAILED: {ro.get('errs')} | | | | | | |")
            continue
        lines.append(
            f"| {ro['B']} "
            f"| {_fmt(ro['ttft_client_mean_s'])} / {_fmt(ro['ttft_client_p50_s'])} / {_fmt(ro['ttft_client_p95_s'])} "
            f"| {_fmt(ro['engine_ttft_mean_s'])} "
            f"| {_fmt(ro['tpot_client_mean_s'])} "
            f"| {_fmt(ro['engine_tpot_mean_s'])} "
            f"| {_fmt(ro['throughput_client_tok_s'], 1)} "
            f"| {_fmt(ro['engine_token_per_s'], 1)} "
            f"| {_fmt(ro['mem_peak_mib'], 0)} |")
    lines.append("")

    lines.append("## 辅助列\n")
    lines.append("| B | request/s | speedup | wall s | queue s | prefill s | mem idle/after-load MiB | errs |")
    lines.append("|" + "---|" * 8)
    for ro in out:
        if not ro.get("ok"):
            continue
        lines.append(
            f"| {ro['B']} | {_fmt(ro['request_per_s'], 2)} | {_fmt(ro['speedup'], 2)} "
            f"| {_fmt(ro['wall_time_s'], 2)} | {_fmt(ro['engine_queue_mean_s'])} "
            f"| {_fmt(ro['engine_prefill_mean_s'])} "
            f"| {_fmt(ro['mem_idle_mib'], 0)} / {_fmt(ro['mem_after_load_mib'], 0)} "
            f"| {ro['errs']} |")
    lines.append("")

    valid = [ro for ro in out if ro.get("ok")]
    if valid:
        lines.append("## ASCII 趋势\n")
        lines.append("```")
        lines.append("B    tok/s(client)  TTFT(client)  TPOT(client)  mem_peak")
        for ro in valid:
            bar = "#" * max(1, int((ro["throughput_client_tok_s"] or 0) / 100))
            lines.append(f"{str(ro['B']):<4} {_fmt(ro['throughput_client_tok_s'], 1):>10}  "
                         f"{_fmt(ro['ttft_client_mean_s']):>10}  "
                         f"{_fmt(ro['tpot_client_mean_s']):>10}  "
                         f"{_fmt(ro['mem_peak_mib'], 0):>7}  {bar}")
        lines.append("```\n")

    lines.append("## 口径（必须写清）\n")
    lines.append("- **客户端 TTFT vs 引擎 TTFT**：客户端 = `perf_counter` 从 `create()` 前到首个非空 chunk；"
                 "引擎 = `/metrics` 的 `vllm:time_to_first_token_seconds` 差分（压测前后各抓一次）。"
                 "两者差异即 HTTP/SDK 开销，本脚本用 async client 已把它压到最小。")
    lines.append("- **吞吐**：客户端 `token_per_s = 总输出 token / wall_time`（含爬坡+收尾，非纯稳态窗口）；"
                 "引擎 `= Δvllm:generation_tokens / wall_time`。N=64 且 max_tokens=512 时 wall 足够长，"
                 "整段近似稳态（比 Phase 0 的 ~0.2s 窗口可信）。")
    lines.append("- **显存**：只读被 pin 的物理卡；idle = 起服务前，after-load = 服务就绪后"
                 "（权重 + KV 预留），peak = 压测期间 0.5s 采样最大值。")
    lines.append("- **数据质量**：共享卡，全部 `reference-only`；`gpus_used_*` 用于事后判断是否被他人抢卡。\n")

    (sweep_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return out


# -------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8016)
    p.add_argument("--ready_timeout", type=int, default=300)
    p.add_argument("--S", type=int, default=1024)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--B_list", default="1,4,8,16,32")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.55)
    p.add_argument("--fixed_max_num_seqs", type=int, default=64)
    p.add_argument("--engine_args", type=str, default="",
                   help="额外的 vllm serve 参数（单字符串，按 shell 规则切分）")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--prefix_len", type=int, default=0,
                   help="共享前缀长度，必须为 block_size 的整数倍")
    p.add_argument("--prefix_ratio", type=float, default=1.0,
                   help="共享前缀的请求占比，(0, 1]")
    p.add_argument("--n_warm", type=int, default=3,
                   help="warmup 次数，用于推导 prefix cache 底噪")
    p.add_argument("--prefix_seed", type=int, default=999,
                   help="共享前缀的独立随机源种子")
    p.add_argument("--block_size", type=int, default=16,
                   help="必须与引擎解析后的 block_size 一致，见 metrics 的 cache_config_info")
    p.add_argument("--out_dir", default=str(HERE / "results"))
    a = p.parse_args()
    a.B_list = [int(x) for x in str(a.B_list).split(",") if x.strip()]
    return a


async def amain():
    args = parse_args()
    pynvml.nvmlInit()

    if args.gpu is None:
        args.gpu, used = pick_gpu()
        print(f"[gpu] auto-selected physical GPU {args.gpu} (used={used} MiB)",
              flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    args.sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.out_dir) / f"sweep_{args.sweep_id}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    env = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch  # 只读版本号，不触碰 CUDA context
        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
    except Exception as e:  # noqa: BLE001
        env["torch_error"] = repr(e)
    try:
        import importlib.metadata as md
        env["vllm"] = md.version("vllm")
    except Exception as e:  # noqa: BLE001
        env["vllm_error"] = repr(e)

    meta = {
        "sweep_id": args.sweep_id,
        "started": datetime.now().isoformat(),
        "model": args.model,
        "B_list": args.B_list,
        "S": args.S, "N": args.N, "max_tokens": args.max_tokens,
        "temperature": args.temperature, "top_p": args.top_p,
        "vocab_size": args.vocab_size,
        "fixed_max_num_seqs": args.fixed_max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "gpu_physical_index": args.gpu,
        "gpu_name": gpu_name(args.gpu),
        "gpu_total_mib": gpu_total_mib(args.gpu),
        "repeat": args.repeat,
        "env": env,
        "quality": "reference-only",
        "quality_reason": "shared GPU",
        "engine_args": getattr(args, "engine_args", ""),
        "n_warm": args.n_warm,
        "prefix_len": args.prefix_len,
        "prefix_ratio": args.prefix_ratio,
        "prefix_seed": args.prefix_seed,
        "block_size": args.block_size,
    }
    (sweep_dir / "sweep_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[sweep] dir  = {sweep_dir}", flush=True)
    print(f"[sweep] meta = {json.dumps(meta, ensure_ascii=False)}", flush=True)

    rows = []
    for rep in range(args.repeat):
        for B in args.B_list:
            tag = f"B{B}" if args.repeat == 1 else f"B{B}_r{rep}"
            print(f"\n===== [{tag}] start {datetime.now().isoformat()} =====",
                  flush=True)
            r = await run_one_level(B, args, sweep_dir, tag)
            r["repeat"] = rep
            rows.append(r)
            (sweep_dir / f"result_{tag}.json").write_text(
                json.dumps(r, indent=2, ensure_ascii=False, default=float),
                encoding="utf-8")
            if r.get("ok"):
                lat, sysd, eng = (r["record"]["latency"],
                                  r["record"]["system"], r["engine"])
                print(f"[{tag}] OK wall={sysd['wall_time']:.2f}s | "
                      f"client TTFT={lat['ttft_mean']:.4f}s | "
                      f"engine TTFT={_fmt(eng.get('engine_ttft_mean_s'))}s | "
                      f"client TPOT={lat['tpot_mean']:.4f}s | "
                      f"engine TPOT={_fmt(eng.get('engine_tpot_mean_s'))}s | "
                      f"tok/s={sysd['token_per_s']:.1f} | "
                      f"peak={r.get('mem_peak_mib')}MiB | "
                      f"errs={len(r['record'].get('error') or [])}", flush=True)
            else:
                print(f"[{tag}] FAILED: {r.get('error')}", flush=True)

    out = write_summary(rows, args, sweep_dir)
    print("\n================ SUMMARY (B x four quantities) ================",
          flush=True)
    for ro in out:
        print(json.dumps(ro, ensure_ascii=False, default=float), flush=True)
    print(f"\n[done] {datetime.now().isoformat()}", flush=True)
    print(f"[done] results in {sweep_dir}", flush=True)


if __name__ == "__main__":
    asyncio.run(amain())
