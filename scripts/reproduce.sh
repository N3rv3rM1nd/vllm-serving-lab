#!/usr/bin/env bash
# 只跑分析：假设 raw/ 里已有数据，不需要 GPU。
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"

if [ ! -d raw ] || [ -z "$(ls -A raw 2>/dev/null | grep -v .gitkeep || true)" ]; then
  echo "raw/ 为空：请先跑 scripts/rerun_all.sh，或从 release 下载数据解压到 raw/"
  exit 1
fi

labels=$("$PY" - <<'EOF'
import json, pathlib
cfg = pathlib.Path("experiments.json")
data = json.loads(cfg.read_text())
out = []
for exp in data.values():
    out.extend(exp.get("labels", []))
print(" ".join(out))
EOF
)

if [ -z "${labels// /}" ]; then
  echo "experiments.json 里还没有 label；先跑实验并回填清单"
  exit 1
fi

for label in $labels; do
  echo "=== $label"
  "$PY" analysis/run_analysis.py --label "$label"
done

echo "完成：结果在 results/<label>/"
