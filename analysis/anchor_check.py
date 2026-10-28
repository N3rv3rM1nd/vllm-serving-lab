#!/usr/bin/env python3
"""锚点闸门：跨天可比性检查（A / B / C）+ τ_gap + 阶梯污染检查。"""

from __future__ import annotations

import json
from pathlib import Path

from t_distribution import τ_distribution

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "anchor"
CI = 0.95

TAU_ENG = {32: 6.35, 8: 19.50}
TAU_TOK = {32: 6.33, 8: 2.12}
TAU_CLIENT = {32: 25.40, 8: 23.20}


def merge(label: str) -> dict:
    d = RESULTS / label
    if not (d / "merge_result_Bn.json").exists():
        d = RESULTS
    return json.loads((d / "merge_result_Bn.json").read_text())


def rec(label: str, b: int) -> dict:
    m = merge(label)[str(b)]
    client = m["ttft_mean_s"] * 1000
    return {"engine_ttft": m["ttft_ms"], "queue": m["queue_ms"],
            "prefill": m["prefill_ms"], "tokens_s": m["tokens_s"],
            "client_ttft": client, "gap": client - m["ttft_ms"],
            "e2e": m["e2e_mean_s"], "n": m["n_requests"],
            "rate": m["n_requests"] / m["request_span_s"]}


def delta(new: float, old: float) -> float:
    return (new - old) / old * 100


def tau_gap() -> tuple[dict, dict]:
    gaps = {8: [], 32: []}
    for r in range(5):
        m = merge(f"repeat5_r{r}")
        for b in (8, 32):
            gaps[b].append(m[str(b)]["ttft_mean_s"] * 1000 - m[str(b)]["ttft_ms"])
    return {b: τ_distribution(v, CI) for b, v in gaps.items()}, gaps


def judge(d: float, t: float) -> str:
    return "超出" if abs(d) > t else "在噪声内"


def row(name, old, new, t32, t8=None):
    d = delta(new, old)
    j8 = judge(d, t8) if t8 is not None else "—"
    return f"| {name} | {old:.1f} | {new:.1f} | {d:+.2f}% | {t32:.2f} / {t8:.2f} | {judge(d, t32)} / {j8} |"


def main() -> None:
    tg, raw_gaps = tau_gap()
    L = []
    L.append("# 锚点闸门：跨天可比性\n")
    L.append(f"τ_gap: B=8 → **{tg[8]:.2f}%**（{', '.join(f'{g:.0f}' for g in raw_gaps[8])} ms）；"
             f"B=32 → **{tg[32]:.2f}%**（{', '.join(f'{g:.0f}' for g in raw_gaps[32])} ms）\n")
    L.append("判据 `|Δ%| > τ%`；判定列格式 `借B32 / 借B8`；gap 用实测 τ_gap。\n")

    def block(title, o, n):
        L.append(f"## {title}\n")
        L.append("| 指标 | 旧 | 新 | Δ% | τ%（B32 / B8） | 判定 |")
        L.append("| --- | ---: | ---: | ---: | --- | --- |")
        L.append(row("engine TTFT (ms)", o["engine_ttft"], n["engine_ttft"], TAU_ENG[32], TAU_ENG[8]))
        L.append(row("queue (ms)", o["queue"], n["queue"], TAU_ENG[32], TAU_ENG[8]))
        L.append(row("prefill (ms)", o["prefill"], n["prefill"], TAU_ENG[32], TAU_ENG[8]))
        L.append(row("tokens/s", o["tokens_s"], n["tokens_s"], TAU_TOK[32], TAU_TOK[8]))
        L.append(row("client TTFT (ms)", o["client_ttft"], n["client_ttft"], TAU_CLIENT[32], TAU_CLIENT[8]))
        L.append(row("gap (ms)", o["gap"], n["gap"], tg[32], tg[8]))
        L.append("")

    block("检查 A：锚点 B=16（baseline 10-15 vs bpsweep 10-25）",
          rec("baseline", 16), rec("bpsweep", 16))
    block("检查 B：同参数跨天重复（S=8192, B=8）",
          rec("s8192", 8), rec("s8192fill", 8))

    hi, ok = rec("smoke_b16", 16), rec("bpsweep", 16)
    L.append("## 检查 C：宿主负载灵敏度（同配置 B=16）\n")
    L.append("| 指标 | bpsweep（正常） | smoke（高负载） | Δ% |")
    L.append("| --- | ---: | ---: | ---: |")
    for key, nm, unit in [("engine_ttft", "engine TTFT", "ms"), ("tokens_s", "tokens/s", ""),
                          ("client_ttft", "client TTFT", "ms"), ("gap", "gap", "ms")]:
        suffix = f" ({unit})" if unit else ""
        L.append(f"| {nm}{suffix} | {ok[key]:.1f} | {hi[key]:.1f} | {delta(hi[key], ok[key]):+.2f}% |")
    L.append("")

    # 阶梯污染检查
    L.append("## 阶梯是否可信：req/s × E2E 应等于有效并发\n")
    L.append("| B | N=64 req/s | N=64 E2E(s) | N=64 eff | N=256 req/s | N=256 E2E(s) | N=256 eff | 期望 eff |")
    L.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    b64_extra = merge("baseline")
    for b in (16, 20, 24, 28, 32):
        r64 = merge("bpsweep").get(str(b)) or b64_extra.get(str(b))
        r256 = merge("baseline_n256").get(str(b))
        exp = b if b <= 64 else 64
        def cells(r):
            if not r:
                return "—", "—", "—"
            rate = r["n_requests"] / r["request_span_s"]
            return f"{rate:.3f}", f"{r['e2e_mean_s']:.2f}", f"{rate * r['e2e_mean_s']:.1f}"
        c64, c256 = cells(r64), cells(r256)
        L.append(f"| {b} | {c64[0]} | {c64[1]} | {c64[2]} | {c256[0]} | {c256[1]} | {c256[2]} | {exp} |")
    L.append("")

    L.append("## 决策 B 参考：bup_aligned B=64（N=64, seqs=64）\n")
    r, base = rec("bup_aligned", 64), rec("baseline", 32)
    L.append(f"- B=64：tokens/s **{r['tokens_s']:.1f}**，engine TTFT **{r['engine_ttft']:.0f} ms**，"
             f"client TTFT {r['client_ttft']:.0f} ms，eff={r['rate']*r['e2e']:.1f}")
    L.append(f"- B=32（baseline）：tokens/s {base['tokens_s']:.1f}，engine TTFT {base['engine_ttft']:.0f} ms → "
             f"Δtokens **{delta(r['tokens_s'], base['tokens_s']):+.1f}%**，"
             f"ΔTTFT **{delta(r['engine_ttft'], base['engine_ttft']):+.1f}%**")
    L.append("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "anchor_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
