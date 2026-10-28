import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
INPUT = RESULTS / "merge_result_Bn.json"


def load_rows():
    with INPUT.open(encoding="utf-8") as file:
        data = json.load(file)

    rows = sorted(data.values(), key=lambda row: row["batch"])

    for row in rows:
        row["req_per_s"] = row["n_requests"] / row["request_span_s"]
        row["ttft_p95_ms"] = row["ttft_p95_s"] * 1000.0

    return rows


def feasible_rate(rows, threshold_ms):
    feasible = [
        row["req_per_s"]
        for row in rows
        if row["ttft_p95_ms"] <= threshold_ms
    ]
    return max(feasible, default=0.0)


def first_break(rows, threshold_ms):
    for row in rows:
        if row["ttft_p95_ms"] > threshold_ms:
            return row["batch"]
    return None


def plot_goodput_vs_b(rows, timestamp):
    batches = [row["batch"] for row in rows]
    rates = [row["req_per_s"] for row in rows]
    p95_values = [row["ttft_p95_ms"] for row in rows]

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.plot(batches, rates, marker="o", linewidth=1.8)

    for batch, rate, p95 in zip(batches, rates, p95_values):
        axis.annotate(
            f"{p95:.1f} ms",
            (batch, rate),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    for threshold in (500, 1000):
        rate = feasible_rate(rows, threshold)
        axis.axhline(
            rate,
            linestyle="--",
            linewidth=1.1,
            label=f"{threshold} ms SLO: {rate:.2f} req/s",
        )

        crossing = first_break(rows, threshold)
        if crossing is not None:
            axis.axvline(
                crossing,
                linestyle=":",
                linewidth=1.1,
                label=f"first break: B={crossing}",
            )

    axis.set_xlabel("batch size B")
    axis.set_ylabel("requests / second")
    axis.set_title("Goodput vs B, with client P95 TTFT")
    axis.set_xticks(batches)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()

    output = RESULTS / f"goodput_vs_b_{timestamp}.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_sensitivity(rows, timestamp):
    thresholds = np.arange(200, 1501, 50)
    rates = [feasible_rate(rows, threshold) for threshold in thresholds]

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.step(
        thresholds,
        rates,
        where="post",
        linewidth=2.0,
        label="max feasible req/s",
    )

    for threshold in (500, 1000):
        rate = feasible_rate(rows, threshold)
        axis.axvline(threshold, linestyle="--", linewidth=1.0)
        axis.scatter([threshold], [rate], zorder=4, s=35)
        axis.annotate(
            f"{rate:.2f}",
            (threshold, rate),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    axis.set_xlim(200, 1500)
    axis.set_xlabel("SLO threshold: client P95 TTFT (ms)")
    axis.set_ylabel("maximum feasible requests / second")
    axis.set_title("Goodput sensitivity to the P95 TTFT target")
    axis.grid(alpha=0.25)
    figure.tight_layout()

    output = RESULTS / f"goodput_sensitivity_{timestamp}.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def main():
    rows = load_rows()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    print(f"{'B':>4} {'req/s':>9} {'tokens/s':>10} {'p95 TTFT ms':>12}")
    for row in rows:
        print(
            f"{row['batch']:>4} "
            f"{row['req_per_s']:>9.3f} "
            f"{row['tokens_s']:>10.2f} "
            f"{row['ttft_p95_ms']:>12.2f}"
        )

    print(plot_goodput_vs_b(rows, timestamp))
    print(plot_sensitivity(rows, timestamp))


if __name__ == "__main__":
    main()