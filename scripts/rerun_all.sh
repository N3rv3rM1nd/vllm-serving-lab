#!/usr/bin/env bash
# 重跑 E1–E4。需要 GPU，约 90 min。
# 注意：label 必须与 experiments.json 一致；跑完请回填清单。
set -euo pipefail
cd "$(dirname "$0")/bench"

PY="${PY:-python}"
GPU="${GPU:-0}"
S=1024
N_DEFAULT=256

run() {
  local out="$1"; shift
  echo "=== $out"
  "$PY" -u baseline_sweep.py --gpu "$GPU" --out_dir "../raw_run/$out" "$@"
}

# ---------- E1 并发与 SLO ----------
run r4090_e1 --S $S --N $N_DEFAULT --max_tokens 512 --B_list 1,2,4,8,16,32,64
for r in 0 1 2 3 4; do
  run "r4090_e1_b8_r$r"  --S $S --N $N_DEFAULT --max_tokens 512 --B_list 8
  run "r4090_e1_b16_r$r" --S $S --N $N_DEFAULT --max_tokens 512 --B_list 16
  run "r4090_e1_b32_r$r" --S $S --N $N_DEFAULT --max_tokens 512 --B_list 32
done

# ---------- E2 Prefix Cache ----------
for L in 0 512 1024 2048; do
  for R in 0.5 1.0; do
    tag="r4090_e2_L${L}_r$(printf '%03d' "$(awk -v x="$R" 'BEGIN{print x*100}')")"
    run "$tag" --S 3072 --N 64 --max_tokens 512 --B_list 16 \
      --prefix_len "$L" --prefix_ratio "$R"
  done
done
run r4090_e2_pcoff --S 3072 --N 64 --max_tokens 512 --B_list 16 \
  --prefix_len 2048 --prefix_ratio 1.0 \
  --engine_args "--no-enable-prefix-caching"

# ---------- E3 Chunked Prefill 预算 ----------
for MB in 512 1024 2048 4096 8192; do
  for SX in 1024 8192; do
    run "r4090_e3_mnbt${MB}_S${SX}" --S "$SX" --N 64 --max_tokens 512 --B_list 32 \
      --engine_args "--max-num-batched-tokens $MB"
  done
done
run r4090_e3_cpoff_S8192 --S 8192 --N 64 --max_tokens 512 --B_list 32 \
  --engine_args "--no-enable-chunked-prefill"

# ---------- E4 KV 稀缺 ----------
for G in 0.35 0.55; do
  for B in 32 64; do
    run "r4090_e4_gmem$(printf '%03d' "$(awk -v x="$G" 'BEGIN{print x*100}')")_b${B}" \
      --S 8192 --N 64 --max_tokens 512 --B_list "$B" --gpu_memory_utilization "$G"
  done
done

echo "完成。下一步：把 raw_run/* 迁移到 raw/<label>/ 并回填 experiments.json"
