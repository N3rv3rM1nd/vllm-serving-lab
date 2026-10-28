# 实验设计（跑之前写，跑完对照）

原则：**一问一实验**、同会话对照、头条结论 ≥3 次重复、必含一个"效果应当消失"的边界臂。

---

## E1 · 并发与 SLO

- **假设**：存在一个 B 使 P95 client TTFT 跨过 1 s，且该处的延迟由 queue（排队）主导，而非 prefill。
- **臂**：B = 1, 2, 4, 8, 16, 32, 64；N=256（压缩尾波量化）。
- **重复**：B = 8 / 16 / 32 各 5 次。
- **指标**：TTFT P50/P95、tokens/s、有效并发 `eff = req/s × E2E`、queue/prefill/decode。
- **预期形状**：TTFT 先平后陡；`queue/TTFT` 从 <1% 升到 >60%；吞吐与 eff 近似线性。
- **决策**：1 s SLO 下的推荐 B 与吞吐代价。

## E2 · Prefix Cache 命中率

- **假设**：TTFT 随"命中 token 数"近似线性下降；端到端吞吐增益次线性（prefill 只占 E2E 一小部分）。
- **臂**：`L_shared` ∈ {0, 512, 1024, 2048} × `ratio` ∈ {0.5, 1.0}，S=3072，B=16。
- **边界臂**：`L=0`（效果应消失）、`--no-enable-prefix-caching`（命中应恒为 0）。
- **指标**：命中率、TTFT、prefill_ms、tokens/s、缓存额外显存。
- **预期形状（通用式，勿写死 N）**：

```
理论命中 token ≈ floor + (n_shared − 1) × L_shared
    floor     ≈ S − block_size          # warmup 预热首条请求的贡献
    n_shared  = round(N × prefix_ratio)
理论命中率 ≈ 上述值 ÷ (N × S)
```

  - N=256、S=3072、L=2048、ratio=1：floor=3056，命中率 ≈ (3056 + 255×2048)/786432
  - L=0 时命中率 ≈ 3056/(N×S)（N=256 约 0.389%），**这是必须减掉的底噪**
  - `prefix_ratio < 1` 时公式自动退化，不能再用 N−1

  TTFT 降幅 ≈ 命中比例 × prefill 时长。
- **决策**：命中率到多少时开启 PC 才划算（含显存成本）。

## E3 · Chunked Prefill 预算

- **假设**：存在预算膝盖——再增大则 ITL P99 陡增，而 TTFT 收益递减。
- **臂**：`max_num_batched_tokens` ∈ {512, 1024, 2048, 4096, 8192} × S ∈ {1024, 8192}，B=32。
- **边界臂**：关闭 chunked prefill（应复现 queue≈0、prefill 暴涨）。
- **指标**：TTFT、prefill_ms、**ITL P99**、tokens/s。
- **预期形状**：TTFT 随预算变差、ITL P99 随预算变好；P99 呈 step 时长的整数倍。
- **决策**：同时约束 TTFT 与 ITL 时的最优预算。

## E4 · KV 稀缺（24 GB 卡的关键一组）

- **假设**：一旦出现 preemption，`max_num_seqs` / `max_num_batched_tokens` 才产生超过噪声下限的效应。
- **臂**：S=8192，B ∈ {32, 64}，`gpu_memory_utilization` ∈ {0.35, 0.55}。

> 10-28 修订后的正式设计：以 `--kv-cache-memory-bytes` 固定池、`--max-num-active-seqs` 控制准入。

| 臂 | KV 池 | 准入 M | 作用 |
| --- | ---: | ---: | --- |
| A | 10 GiB | 32 | 容量充足 + 满准入 |
| B | 10 GiB | 16 | 容量充足 + 限准入（限流的纯代价） |
| C | 低容量（E0 选定） | 32 | 容量受限 + 满准入（应出现抢占） |
| D | 低容量 | 16 | 容量受限 + 限准入（限流能否救回） |
| **E** | **低容量** | **64** | **低容量锚点：证明抢占来自容量而非准入** |

**E 臂必需**：没有它，C/D 的差异无法区分"容量造成"还是"准入 32 造成"；D vs E 才是"限流救回多少"的干净对照。每臂 5 次、随机区组，共 25 次。
- **算账**：B=32、S_total=8704 的 KV 需求 ≈ **8.0 GiB**；gmem 0.35 的池约 **5 GiB** → 必然抢占。
- **指标**：preemption 数、queue、TTFT、tokens/s。
- **预期形状**：gmem 0.55 无抢占；0.35 出现抢占且吞吐塌陷。
- **决策**：给定 S 与显存的容量规划曲线。

---

## 判定口径（预先登记）

- 阈值用**重复实验的预测区间**（单次观测 vs n 次均值，因子 √(1+1/n)），不用均值置信区间。
- 跨会话比较只用 engine 侧指标；client 侧受宿主负载影响。
- 结论只写"超出噪声下限 / 在噪声内"，不写"显著"。

---

## 运行组织（10-28 追加）

1. **随机区组可复现**：`experiments.json` 必须记录
   - `block_seed`：生成区组内顺序的随机种子
   - `blocks`：每轮区组实际的臂顺序，例如 `["C","A","D","B","E"]`
   事后可据此证明顺序不是挑出来的。
2. **运行期时间序列**：每个 level 额外产出 `timeseries_<tag>.csv`
   （列：`unix_ts, running, waiting, kv_usage_perc, preemptions_total`，默认 1 s 间隔）。
3. **label 命名**：`r4090_<exp>_<setting>`；语义以 `raw/config/<label>__sweep_meta.json` 为准，
   分析脚本不得依赖 label 字符串匹配。
