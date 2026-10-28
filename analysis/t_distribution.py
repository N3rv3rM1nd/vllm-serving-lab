import json
import numpy as np

from pathlib import Path
from scipy.stats import t
from typing import Any, FrozenSet

CI = 0.95

def τ_distribution(data: list[float], confidence_level: float) -> float:
    '''评估单次测量阈值 '''
    n = len(data)
    if n < 2:
        raise ValueError(f"only one data is computed: {data}")
    x_avg = sum(data) / n
    x_std = (sum((data[i] - x_avg) ** 2 for i in range(n)) 
             / (n - 1)) ** 0.5
    cp = 1 - (1 - confidence_level) / 2
    q = t.ppf(cp, n - 1) # 自由度
    return 100 * q * x_std * (n + 1) ** 0.5 / (x_avg * n ** 0.5)

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent  / 'results'
    repeat_merge: dict[int, dict[str, Any]] = {}
    tok_s: dict[int, list[float]] = {8: [], 32: []}
    engine_ttft: dict[int, list[float]] = {8: [], 32: []}
    client_ttft: dict[int, list[float]] = {8: [], 32: []}
    τ_summary: dict[int, dict[str, float]] = {8: {}, 32: {}}
    for r in range(5):
        with (root / f'repeat5_r{r}' / f'merge_result_Bn.json').open(
            'r',
            encoding="utf-8",
        ) as f:
            repeat_merge[r] = json.load(f)
        tok_s[8].append(repeat_merge[r]["8"]["tokens_s"])
        tok_s[32].append(repeat_merge[r]["32"]["tokens_s"])

        engine_ttft[8].append(repeat_merge[r]["8"]["ttft_ms"]) # 单位统一ms
        engine_ttft[32].append(repeat_merge[r]["32"]["ttft_ms"]) # 单位统一ms

        client_ttft[8].append(repeat_merge[r]["8"]["ttft_mean_s"] * 1000)
        client_ttft[32].append(repeat_merge[r]["32"]["ttft_mean_s"] * 1000)

    τ_summary[8]["t_tokens_s(%)"] = τ_distribution(tok_s[8], CI)
    τ_summary[32]["t_tokens_s(%)"] = τ_distribution(tok_s[32], CI)
    τ_summary[8]["t_engine_ttft(%)"] = τ_distribution(engine_ttft[8], CI)
    τ_summary[32]["t_engine_ttft(%)"] = τ_distribution(engine_ttft[32], CI)
    τ_summary[8]["t_client_ttft(%)"] = τ_distribution(client_ttft[8], CI)
    τ_summary[32]["t_client_ttft(%)"] = τ_distribution(client_ttft[32], CI)
    with (root / f'repeat5_r0' / 'τ_distribution.json').open(
        'w',
        encoding="utf-8",
    ) as f:
        json.dump(τ_summary, f, ensure_ascii=False, indent=2)
    print(f"t_distribution was saved to "
          f"{str(root / f'repeat5_r{r}' / 'τ_distribution.json')}")