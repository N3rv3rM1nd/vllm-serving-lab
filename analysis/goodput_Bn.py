'''
SLO-strict: P95 client TTFT ≤ 500 ms  | 依据：交互式对话首字的常见体感阈值
SLO-loose : P95 client TTFT ≤ 1000 ms | 依据：批式/长任务可容忍等待
SLO-stream: P95 chunk 间隔 ≤ 50 ms    | 依据：低于人阅读速度(180ms/token)的 1/3
'''
import json, argparse
import numpy as np

from pathlib import Path
from typing import Any
from run_paths import add_run_args, resolve_run

# BATCHES = (1, 4, 8, 16, 32)

def discard_main():
    '''
    root = Path(__file__).resolve().parent.parent
    goodput: dict[int, dict[str, Any]] = {b: {} for b in BATCHES}
    with open(root / 'results' / 'result_Bn_requests.json', 'r') as f:
        Bn_requests = json.load(f)
    with open(root / 'results' / 'merge_result_Bn.json') as f:
        merge_Bn = json.load(f)
    chunks: dict[int, list[float]] = {b: [] for b in BATCHES}
    
    for b in BATCHES:
        for r in Bn_requests:
            if r["batch"] == b:
                chunks[b].append(r["sum_itls_s"] / (r["n_out"] - 1))
        goodput[b]["chunk_p95_ms"] = np.percentile(chunks[b], 95) * 1000

        goodput[b]["req_per_s"] = merge_Bn[str(b)]["n_requests"] / merge_Bn[str(b)]["request_span_s"]
        goodput[b]["tokens_s"] = merge_Bn[str(b)]["tokens_s"]
        goodput[b]["ttft_p95_ms"] = merge_Bn[str(b)]["ttft_p95_s"] * 1000
        
        goodput[b]["meets_stream"] = True if goodput[b]["chunk_p95_ms"] < 50 else False
        goodput[b]["meets_stric"] = True if goodput[b]["ttft_p95_ms"] < 500 else False
        goodput[b]["meets_loose"] = True if goodput[b]["ttft_p95_ms"] < 1000 else False

    with open(root / 'results' / 'goodput_Bn.json', 'w') as f:
        json.dump(goodput, f, indent=2)
    '''
    pass
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
    batches = [job.batch for job in run.jobs]
    goodput: dict[int, dict[str, Any]] = {batch: {} for batch in batches}

    with (run.out_dir / "result_Bn_requests.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        bn_requests = json.load(file)

    with (run.out_dir / "merge_result_Bn.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        merge_bn = json.load(file)

    chunks: dict[int, list[float]] = {batch: [] for batch in batches}

    for batch in batches:
        for record in bn_requests:
            if record["batch"] == batch:
                chunks[batch].append(
                    record["sum_itls_s"] / (record["n_out"] - 1)
                )

        goodput[batch]["chunk_p95_ms"] = float(
            np.percentile(chunks[batch], 95) * 1000
        )
        goodput[batch]["req_per_s"] = (
            merge_bn[str(batch)]["n_requests"]
            / merge_bn[str(batch)]["request_span_s"]
        )
        goodput[batch]["tokens_s"] = merge_bn[str(batch)]["tokens_s"]
        goodput[batch]["ttft_p95_ms"] = (
            merge_bn[str(batch)]["ttft_p95_s"] * 1000
        )
        goodput[batch]["meets_stream"] = bool(
            goodput[batch]["chunk_p95_ms"] <= 50
        )
        goodput[batch]["meets_strict"] = bool(
            goodput[batch]["ttft_p95_ms"] <= 500
        )
        goodput[batch]["meets_loose"] = bool(
            goodput[batch]["ttft_p95_ms"] <= 1000
        )

    run.out_dir.mkdir(parents=True, exist_ok=True)

    with (run.out_dir / "goodput_Bn.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(goodput, file, ensure_ascii=False, indent=2)

    print(f"{run.label}: goodput -> {run.out_dir}")        