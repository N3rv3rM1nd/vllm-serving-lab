#!/usr/bin/env python3
"""B=32 参数效应表：9 个配置 vs baseline，用实测 τ 判定。

判定阈值来自 repeat5_r0..r4 的预测区间（含 sqrt(1+1/n)），
即「单次观测 vs n=5 均值」的 95% 区间，单位为 %。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from t_distribution import τ_distribution

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
RESULTS = ROOT / "results"
OUT = RESULTS / "param_effect"
LABELS = ["gmem035", "gmem075", "mnbt2048", "mnbt16384",
          "nochunk", "maxseq16", "maxseq128", "maxseq256"]
BENCH_B = 32


def load_configs() -> dict:
    """?????? results_scans__configs.json?????????

    ???raw/logs/ ? 10-15 ?? vllm_*.log ?????? vllm.log ????
    ????????????????????
    """
    j = json.loads((RAW / "config" / "results_scans__configs.json").read_text())
    defaults = {"fixed_max_num_seqs": 64, "gpu_memory_utilization": 0.55,
                "engine_args": ""}
    out = {}
    for c in j["configs"]:
        d = dict(defaults)
        d.update({k: v for k, v in c.items() if k != "label"})
        out[c["label"]] = d
    return out


CONFIGS = load_configs()


def knobs(label: str) -> dict:
    c = CONFIGS.get(label, {"fixed_max_num_seqs": 64,
                            "gpu_memory_utilization": 0.55,
                            "engine_args": ""})
    args = c.get("engine_args") or ""
    m = re.search(r"--max-num-batched-tokens\s+(\d+)", args)
    return {
        "max_num_seqs": c["fixed_max_num_seqs"],
        "gpu_mem": c["gpu_memory_utilization"],
        "mnbt": int(m.group(1)) if m else None,
        "chunked": "no-enable-chunked-prefill" not in args,
    }


def tau() -> dict:
    tok, ttft = {8: [], 32: []}, {8: [], 32: []}
    for r in range(5):
        p = RESULTS / f"repeat5_r{r}" / "merge_result_Bn.json"
        m = json.loads(p.read_text())
        tok[8].append(m["8"]["tokens_s"])
        tok[32].append(m["32"]["tokens_s"])
        ttft[8].append(m["8"]["ttft_ms"])
        ttft[32].append(m["32"]["ttft_ms"])
    return {
        "tokens_s": τ_distribution(tok[BENCH_B], 0.95),
        "engine_ttft": τ_distribution(ttft[BENCH_B], 0.95),
        "raw": {"tokens_s": tok[BENCH_B], "engine_ttft": ttft[BENCH_B]},
    }


def merge_of(label: str) -> dict:
    d = RESULTS / label
    if not (d / "merge_result_Bn.json").exists():
        d = RESULTS          # baseline ????? results/ ???
    return json.loads((d / "merge_result_Bn.json").read_text())


def row(label: str, base: dict) -> dict:
    m = merge_of(label)[str(BENCH_B)]
    k = knobs(label)
    d_tok = (m["tokens_s"] - base["tokens_s"]) / base["tokens_s"] * 100
    d_ttft = (m["ttft_ms"] - base["ttft_ms"]) / base["ttft_ms"] * 100
    return {
        "config": label,
        **k,
        "tokens_s": m["tokens_s"], "d_tokens_pct": d_tok,
        "engine_ttft_ms": m["ttft_ms"], "d_ttft_pct": d_ttft,
        "queue_ms": m["queue_ms"], "prefill_ms": m["prefill_ms"],
        "client_ttft_p95_ms": m["ttft_p95_s"] * 1000,
    }


def verdict(d: float, t: float) -> str:
    return "超出噪声下限" if abs(d) > t else "在噪声内"


def main() -> None:
    t = tau()
    base = merge_of("baseline")[str(BENCH_B)]
    rows = [row(lbl, base) for lbl in LABELS]

    for r in rows:
        r["tokens_verdict"] = verdict(r["d_tokens_pct"], t["tokens_s"])
        r["ttft_verdict"] = verdict(r["d_ttft_pct"], t["engine_ttft"])

    OUT.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# B={BENCH_B} 参数效应（判定阈值来自 repeat5，n=5 预测区间）\n")
    lines.append(f"τ_tokens_s = **{t['tokens_s']:.2f}%**，"
                 f"τ_engine_ttft = **{t['engine_ttft']:.2f}%**\n")
    lines.append("| 配置 | max_num_seqs | max_num_batched_tokens | chunked | gpu_mem | "
                 "tokens/s | Δ% | 判定 | engine TTFT (ms) | Δ% | 判定 | queue (ms) | prefill (ms) |")
    lines.append("| --- | ---: | ---: | :---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |")
    for r in rows:
        lines.append(
            f"| `{r['config']}` | {r['max_num_seqs']} | {r['mnbt'] or '默认'} | "
            f"{'on' if r['chunked'] else ('off' if r['chunked'] is False else '?')} | "
            f"{r['gpu_mem']} | {r['tokens_s']:.1f} | {r['d_tokens_pct']:+.2f}% | "
            f"{r['tokens_verdict']} | {r['engine_ttft_ms']:.1f} | {r['d_ttft_pct']:+.2f}% | "
            f"{r['ttft_verdict']} | {r['queue_ms']:.1f} | {r['prefill_ms']:.1f} |"
        )
    (OUT / f"param_effect_B{BENCH_B}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv
    cols = list(rows[0])
    with (OUT / f"param_effect_B{BENCH_B}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print("\n".join(lines))
    print(f"\nbaseline @B{BENCH_B}: tokens_s={base['tokens_s']:.1f} "
          f"engine_ttft={base['ttft_ms']:.1f}ms")
    print(f"写入 {OUT}/param_effect_B{BENCH_B}.{{md,csv}}")


if __name__ == "__main__":
    main()
