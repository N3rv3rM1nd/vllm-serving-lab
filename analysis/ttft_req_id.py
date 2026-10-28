import json
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
INPUT = RESULTS / 'result_Bn_requests.json'
BATCHES = (1, 4, 8, 16, 32)

def main():
    with INPUT.open(encoding="utf-8") as file:
        rows = json.load(file)

    if not isinstance(rows, list):
        raise ValueError(f"result_Bn_request.json must contain a list")

    frame = pd.DataFrame(rows)

    required = {"batch", "req_id", "ttft_s"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    frame["ttft_ms"] = frame["ttft_s"] * 1000.0

    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(BATCHES),
        figsize=(4.8 * len(BATCHES), 4.8),
        sharex=True,
        squeeze=False,
    )

    for axis, batch in zip(axes[0], BATCHES):
        part = (
            frame[frame["batch"] == batch]
            .sort_values("req_id")
            .copy()
        )

        part["wave"] = part["req_id"] // batch

        if batch > 1:
            for wave_index, (_, wave) in enumerate(part.groupby("wave")):
                if wave_index % 2 == 0:
                    axis.axvspan(
                        wave["req_id"].min() - 0.5,
                        wave["req_id"].max() + 0.5,
                        color="gray",
                        alpha=0.10,
                        linewidth=0,
                    )

        axis.scatter(
            part["req_id"],
            part["ttft_ms"],
            s=24,
            color="#1f77b4",
            alpha=0.85,
            zorder=3,
        )

        mean = part["ttft_ms"].mean()
        p50 = part["ttft_ms"].quantile(0.50)
        p95 = part["ttft_ms"].quantile(0.95)

        axis.axhline(
            mean,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"mean {mean:.1f}",
        )
        axis.axhline(
            p50,
            color="green",
            linestyle=":",
            linewidth=1.2,
            label=f"p50 {p50:.1f}",
        )
        axis.axhline(
            p95,
            color="purple",
            linestyle="-.",
            linewidth=1.2,
            label=f"p95 {p95:.1f}",
        )

        axis.set_title(f"B={batch}")
        axis.set_xlabel("req_id")
        axis.set_ylabel("TTFT (ms)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")

    figure.suptitle("TTFT vs req_id: one point per request")
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = RESULTS / f"ttft_vs_reqid_{timestamp}.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(output)


if __name__ == "__main__":
    main()        