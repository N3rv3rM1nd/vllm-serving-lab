#!/usr/bin/env python3
"""Prefix Cache 命中率自检。

设计前提：**不依赖 label 命名**。
语义全部从 raw/config/<label>__sweep_meta.json 读取（prefix_len / prefix_ratio / n_warm / S / N），
label 只作为索引。

用法：
    python pc_selfcheck.py                 # 扫描所有带 prefix_len 的配置
    python pc_selfcheck.py --labels a b c  # 只看指定 label
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "raw" / "config"
RESULTS = ROOT / "results"
META_SUFFIX = "__sweep_meta.json"
BLOCK = 16          # 命中粒度，用于容差；与引擎 block_size 一致
BLOCK_TOL = BLOCK   # 判定容差：一个 block 的 token 数


def out_dir_of(label: str) -> Path:
    """与 merge_result 保持一致的目录回退规则。"""
    d = RESULTS / label
    if (d / "engine_Bn.json").exists():
        return d
    return RESULTS


def load_meta(label: str) -> dict:
    p = CONFIG_DIR / f"{label}{META_SUFFIX}"
    return json.loads(p.read_text(encoding="utf-8"))


def discover() -> list[str]:
    labels = []
    for p in sorted(CONFIG_DIR.glob(f"*{META_SUFFIX}")):
        label = p.name[: -len(META_SUFFIX)]
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "prefix_len" in meta:
            labels.append(label)
    return labels


def check_label(label: str) -> dict:
    meta = load_meta(label)
    eng_path = out_dir_of(label) / "engine_Bn.json"
    if not eng_path.exists():
        return {"label": label, "status": "SKIP", "reason": "无 engine_Bn.json"}

    rows = json.loads(eng_path.read_text(encoding="utf-8"))

    missing = [k for k in ("N", "S") if k not in meta]
    if missing:
        return {"label": label, "status": "FAIL", "reason": f"meta 缺字段 {missing}"}

    # 旧 meta 没有 prefix_* / n_warm：它们等价于"未启用共享前缀 + 3 次 warmup"
    defaults_used = [k for k in ("prefix_len", "prefix_ratio", "n_warm") if k not in meta]
    N, S = int(meta["N"]), int(meta["S"])
    L = int(meta.get("prefix_len", 0))
    ratio = float(meta.get("prefix_ratio", 1.0))
    n_warm = int(meta.get("n_warm", 3))
    note = f"沿用默认 {defaults_used}" if defaults_used else ""
    n_shared = round(N * ratio)
    expected_gain = (n_shared - 1) * L

    batch = str(max(int(b) for b in rows))
    row = rows[batch]

    hits = row.get("pc_hits_tokens")
    queries = row.get("pc_queries_tokens")
    hits_before = row.get("pc_hits_before")
    queries_before = row.get("pc_queries_before")
    hit_rate = row.get("pc_hit_rate")
    cached_ratio = row.get("pc_cached_token_ratio")

    errs: list[str] = []

    if any(v is None for v in (hits, queries, hits_before, queries_before)):
        return {"label": label, "status": "FAIL", "reason": "pc 字段缺失（引擎未暴露该指标？）"}

    # 1) before 快照必须等于 n_warm 次 warmup 的查询量
    if queries_before != n_warm * S:
        errs.append(f"queries_before {queries_before:.0f} != n_warm*S {n_warm*S}"
                    f"（单位或 warmup 次数异常）")

    # 2) 底噪：warmup 内第 2..n 次各自命中同一可复用前缀
    if hits_before == 0:
        floor = None                      # prefix cache 关闭
    else:
        if n_warm < 2:
            errs.append(f"n_warm={n_warm}，无法推导底噪")
            floor = None
        else:
            floor = hits_before / (n_warm - 1)

    # 3) 判定
    if floor is None:
        verdict = "PASS" if hits == 0 else "FAIL"
        if hits != 0:
            errs.append(f"prefix cache 未启用但 Δhits={hits:.0f}")
        upper = 0.0
    else:
        upper = floor + expected_gain
        if L == 0:
            good = abs(hits - floor) <= BLOCK_TOL
            if not good:
                errs.append(f"control 档 Δhits {hits:.0f} 偏离底噪 {floor:.0f} 超过 {BLOCK_TOL}")
        else:
            if hits == 0:
                errs.append("prefix_len>0 但 Δhits=0：共享前缀未生效")
            elif hits > upper + BLOCK_TOL:
                errs.append(f"Δhits {hits:.0f} 超过上限 {upper:.0f}+{BLOCK_TOL}：口径或单位错误")
            elif hits < floor - BLOCK_TOL:
                errs.append(f"Δhits {hits:.0f} 低于底噪 {floor:.0f}：样本或口径异常")
        verdict = "PASS" if not errs else "FAIL"

    # 4) 两个比率应一致（同一对计数器的两种表达）
    if hit_rate is not None and cached_ratio is not None:
        if abs(hit_rate - cached_ratio) > 1e-9:
            errs.append(f"hit_rate {hit_rate:.6f} != cached_ratio {cached_ratio:.6f}")

    return {
        "label": label,
        "status": verdict,
        "B": batch,
        "prefix_len": L,
        "ratio": ratio,
        "n_shared": n_shared,
        "queries": queries,
        "hits": hits,
        "hit_rate": hit_rate,
        "floor": floor,
        "upper": upper,
        "errors": errs,
        "note": note,
        "reason": "; ".join([r for r in errs] + ([note] if note and not errs else [])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="*", default=None)
    args = ap.parse_args()

    labels = args.labels or discover()
    if not labels:
        print("没有找到带 prefix_len 的配置（先跑一次 prefix 实验）")
        return 1

    print(f"{'label':<28}{'B':>4}{'L':>6}{'ratio':>7}{'n_sh':>6}"
          f"{'queries':>10}{'hits':>9}{'rate':>8}{'floor':>8}{'upper':>9}  判定")
    fails = 0
    for label in labels:
        r = check_label(label)
        if r["status"] == "SKIP":
            print(f"{label:<28}{'-':>4}{'-':>6}{'-':>7}{'-':>6}{'-':>10}{'-':>9}"
                  f"{'-':>8}{'-':>8}{'-':>9}  SKIP {r['reason']}")
            continue
        if r["status"] == "FAIL":
            fails += 1

        def f(v, nd=0):
            return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

        print(f"{r['label']:<28}{r.get('B','-'):>4}{f(r.get('prefix_len')):>6}"
              f"{f(r.get('ratio'),2):>7}{f(r.get('n_shared')):>6}"
              f"{f(r.get('queries')):>10}{f(r.get('hits')):>9}"
              f"{f(r.get('hit_rate'),5):>8}{f(r.get('floor')):>8}{f(r.get('upper')):>9}"
              f"  {r['status']}"
              + (f"  <- {r['reason']}" if r["reason"] else ""))

    print(f"\n合计 {len(labels)} 个 label，失败 {fails} 个")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
