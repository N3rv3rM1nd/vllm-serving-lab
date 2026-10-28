# A40 pilot：上一轮做了什么、错在哪、留下什么

仓库：`N3rv3rM1nd/VLLMService`，标签 `a40-pilot-20261015`。
环境：单张 A40（48 GB，**共享卡**），vLLM 0.31.0+cu129，torch 2.13.0，Qwen2.5-1.5B（GQA）。

## 可直接复用的先验

| 结论 | 数值 | 用途 |
| --- | --- | --- |
| 运行级噪声下限 τ（B=32） | tokens/s 6.3%、engine TTFT 6.4% | 新实验的判定参照 |
| warmup 污染 | 每条 label 有约 **1008 token** 的"底噪命中"（`S − block`） | Prefix Cache 实验必须减去 |
| 调度指纹 | 同批两条首 token 差 0.6 ms、跨对 70 ms → 每步约 2 条 prefill、预算 ≈2048 token | 解释 ITL 尾部 |
| KV 余量 | B=32、S=1536 需求 1.31 GiB，池 12 GiB（占 11%）、preemption 恒为 0 | 说明"没有稀缺就没有优化" |

## 三个必须避免的坑

1. **日志串号**：旧采集脚本把 `vllm.log` 写到 `cwd`，却从脚本目录 rename，导致日志从未被归档、并发时互相覆盖。新仓库已改为写入脚本目录、并靠 `start_new_session` 隔离进程组。
2. **共享卡宿主争用**：同一配置在不同宿主负载下，client TTFT 差 18%，engine 只差 0.9%。跨会话比较只能用 engine 侧。
3. **闭环尾波**：固定 N 时吞吐不是 B 的平滑函数（N=64、B=20 的尾波只填 4 条）。新实验用 N=256 并把横轴改为有效并发。

## 与新仓库内核的三处差异

| 项 | pilot | 本仓库 |
| --- | --- | --- |
| 日志路径 | `cwd/vllm.log`（与归档路径不一致） | 固定写 `bench/vllm.log`，由采集脚本按 tag 归档 |
| 启动器 | 硬编码 `uv run --env-file .env` + 固定 cwd | 有 `uv` 且存在 `.env` 时用 uv，否则回退到 `python -m vllm.entrypoints.openai.api_server` |
| 路径 | 多个脚本硬编码 `/home/hejiarui/...` | 全部由 `Path(__file__)` 推导仓库根 |
