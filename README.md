# vllm-serving-lab

Qwen2.5-1.5B 单卡 serving 的**受控基准**：并发、Prefix Cache、Chunked Prefill、KV 稀缺四组实验，
每组一个可证伪的假设，结论落到"给定 SLO 该选什么"。

> 状态：**实验进行中**。E1–E4 数据尚未产生，下表为待填模板。
> 上一轮（A40 共享卡）的数据已归档在 `N3rv3rM1nd/VLLMService`，标签 `a40-pilot-20261015`，
> 教训与可复用先验见 [docs/PILOT_A40.md](docs/PILOT_A40.md)。

## 结论速览（待填）

| 实验 | 问题 | 结论 |
| --- | --- | --- |
| E1 并发 | 1 s 首字 SLO 下选哪个 B？ | 待填 |
| E2 Prefix Cache | 命中率到多少才值得开？ | 待填 |
| E3 Chunked Prefill | 预算的膝盖在哪？ | 待填 |
| E4 KV 稀缺 | 24 GB 卡上最大并发是多少？ | 待填 |

完整论证见 [docs/report.md](docs/report.md)，实验设计见 [docs/design.md](docs/design.md)。

## 目录

```
bench/      采集内核（vLLM 启动 + 并发压测 + /metrics 抓取）
analysis/   分析脚本（L0/L1 → 指标 → 判定），不含数据
raw/        L0（日志、metrics 原文）+ L1（逐请求时间戳），按 label 分目录
results/    派生：逐档指标、图、判定表
docs/       实验设计、报告、上一轮教训
scripts/    reproduce.sh（只分析）/ rerun_all.sh（重跑实验）
experiments.json  label → 实验族，分析脚本不靠命名猜语义
env/env.md  硬件、驱动、vLLM/torch 版本
```

## 快速开始

```bash
pip install -r requirements.txt

# 只跑分析（不需要 GPU，前提是 raw/ 里已有数据）
bash scripts/reproduce.sh

# 重跑全部实验（需要 GPU，约 90 min）
bash scripts/rerun_all.sh
```

## 数据口径

- **L1** `raw/<label>/requests/result_B*.json`：逐请求 `ttft / itls / window / t_start / t_create / t_end / n_out`
- **L0** `raw/<label>/metrics/metrics_B*_{before,after}.txt`：Prometheus 原文，引擎侧指标一律从 `after − before` 重算
- **L0** `raw/<label>/logs/vllm_B*.log`：服务日志
- 所有派生量都从这两层重算，**不继承任何历史汇总**

三条容易踩的口径线：

1. `n_out` 是**非空 chunk 数**，不是 token 数
2. client 与 engine 的 TTFT 不可互换（同配置可差 200 ms 以上）
3. 闭环固定并发，测得的是**该窗口的完成速率**，不是生产 QPS

## 已知限制

见 [docs/limitations.md](docs/limitations.md)。
