import json
import math
import os, argparse
from datetime import datetime
from typing import Iterable, Mapping, NamedTuple
from run_paths import add_run_args, resolve_run

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from result_Bn_check import ClientSummary
'''
PanelSpec(
    key="ttft",
    title="TTFT",
    dataset="run_summary",
    x="batch",
    y=("ttft_mean_s", "ttft_p50_s", "ttft_p95_s"),
    xlabel="batch size",
    ylabel="seconds",
),
'''
# PATH = "/home/hejiarui/AI_infra_1/VLLMService/results/result_Bn.json"
# REQUEST_PATH = "/home/hejiarui/AI_infra_1/VLLMService/results/result_Bn_requests.json"
# MERGED_PATH = "/home/hejiarui/AI_infra_1/VLLMService/results/merge_result_Bn.json"

class PanelSpec(NamedTuple):
    key: str
    title: str
    dataset: str
    x: str
    y: tuple[str, ...]
    xlabel: str
    ylabel: str
    plot_kind: str = "line"
    sort_x: bool = True
    batch: int | None = None


OLD_PANELS = (
    PanelSpec(
        key="ttft",
        title="TTFT",
        dataset="run_summary",
        x="batch",
        y=("ttft_mean_s", "ttft_p50_s", "ttft_p95_s"),
        xlabel="batch size",
        ylabel="seconds",
    ),
    PanelSpec(
        key="e2e",
        title="End-to-End Latency",
        dataset="run_summary",
        x="batch",
        y=("e2e_mean_s", "e2e_p50_s", "e2e_p95_s"),
        xlabel="batch size",
        ylabel="seconds",
    ),
    PanelSpec(
        key="chunk_interval",
        title="Non-empty Chunk Interval",
        dataset="run_summary",
        x="batch",
        y=(
            "request_equal_chunk_interval_mean_s",
            "interval_equal_chunk_interval_mean_s",
        ),
        xlabel="batch size",
        ylabel="seconds",
    ),
    PanelSpec(
        key="chunk_speed",
        title="Non-empty Chunks speed",
        dataset="run_summary",
        x="batch",
        y=("n_nonempty_chunks_s",),
        xlabel="batch size",
        ylabel="chunks/s",
    ),
    PanelSpec(
        key="chunk-TTFT",
        title="Per-request chunk-TTFT (B=1)",
        dataset="request_metrics",
        x="ttft_s",
        y=("n_out",),
        xlabel="TTFT (s)",
        ylabel="non-empty chunks",
        plot_kind="scatter",
        sort_x=False,
        batch=1,
    ),
)


NEW_PANELS = (
    PanelSpec(
        key="TTFT_queue",
        title="TTFT-queue",
        dataset="merged_summary",
        x="batch",
        y=("ttft_ms",),
        xlabel="batch size",
        ylabel="milliseconds",
    ),
    PanelSpec(
        key="TTFT_prefill",
        title="TTFT-prefill",
        dataset="merged_summary",
        x="batch",
        y=("prefill_ms",),
        xlabel="batch size",
        ylabel="milliseconds",
    ),
    PanelSpec(
        key="TTFT_decode",
        title="TTFT-decode",
        dataset="merged_summary",
        x="batch",
        y=("decode_ms",),
        xlabel="batch size",
        ylabel="milliseconds",
    ),
)

TTFT_PANELS = (
    PanelSpec(
        key="TTFT_gap",
        title="TTFT_gap",
        dataset="ttft_gap",
        x="batch",
        y=("delta_ms",),
        xlabel="batch size",
        ylabel="milliseconds",
    ),
)

PANELS = NEW_PANELS

def client_engine_ttft_gap_ms(path: str) -> pd.DataFrame:
    with open(path, mode='r', encoding="utf-8") as file:
        data = json.load(file)

    ttft_delta_ms = {}
    for b in data:
        ttft_delta_ms[b] = {
            "batch": data[b]["batch"],
            "delta_ms": data[b]["ttft_mean_s"] * 1000 - data[b]["ttft_ms"],
        }

    return (
        pd.DataFrame(ttft_delta_ms.values())
        .sort_values("batch")
        .reset_index(drop=True)
    )

def load_merged_summary(path: str) -> pd.DataFrame:
    with open(path, mode="r", encoding="utf-8") as file:
        data = json.load(file)

    return (
        pd.DataFrame(data.values())
        .sort_values("batch")
        .reset_index(drop=True)
    )

def load_client_summaries(path: str) -> list[ClientSummary]:
    with open(path, mode="r", encoding="utf-8") as file:
        data = json.load(file)

    rows = data.values() if isinstance(data, dict) else data
    summaries = []

    for row in rows:
        if isinstance(row, dict):
            summaries.append(ClientSummary(**row))
        else:
            summaries.append(ClientSummary(*row))

    if not summaries:
        raise ValueError("no ClientSummary records found")

    return sorted(summaries, key=lambda summary: summary.batch)


def to_run_summary(
    summaries: Iterable[ClientSummary],
) -> pd.DataFrame:
    return pd.DataFrame(
        [summary._asdict() for summary in summaries]
    ).sort_values("batch").reset_index(drop=True)


def load_request_metrics(path: str) -> pd.DataFrame:
    with open(path, mode="r", encoding="utf-8") as file:
        rows = json.load(file)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no request metrics found")

    required = {
        "batch",
        "req_id",
        "n_out",
        "ttft_s",
        "e2e_s",
        "chunk_span_s",
        "mean_chunk_interval_s",
        "create_s",
        "drain_s",
        "t_start_s",
        "t_end_s",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"request metrics missing columns: {sorted(missing)}"
        )

    return frame


def validate_panels(
    panels: Iterable[PanelSpec],
    context: Mapping[str, pd.DataFrame],
) -> None:
    seen_keys = set()

    for panel in panels:
        if panel.key in seen_keys:
            raise ValueError(f"duplicate panel key: {panel.key}")
        seen_keys.add(panel.key)

        if not panel.y:
            raise ValueError(f"panel {panel.key} has no y fields")

        if panel.dataset not in context:
            raise ValueError(
                f"panel {panel.key} uses unknown dataset: {panel.dataset}"
            )

        frame = context[panel.dataset]
        required = {panel.x, *panel.y}
        if panel.batch is not None:
            required.add("batch")

        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"panel {panel.key} missing columns: {sorted(missing)}"
            )


def panel_frame(
    context: Mapping[str, pd.DataFrame],
    panel: PanelSpec,
) -> pd.DataFrame:
    frame = context[panel.dataset]
    if panel.batch is not None:
        frame = frame[frame["batch"] == panel.batch]

    result = frame[[panel.x, *panel.y]].melt(
        id_vars=panel.x,
        var_name="series",
        value_name="value",
    )
    result = result.dropna(subset=[panel.x, "value"])

    if result.empty:
        raise ValueError(f"panel {panel.key} has no valid data")

    if panel.sort_x:
        result = result.sort_values(panel.x)

    return result


def draw_panel(
    axis: Axes,
    context: Mapping[str, pd.DataFrame],
    panel: PanelSpec,
) -> None:
    frame = panel_frame(context, panel)

    if panel.plot_kind == "line":
        sns.lineplot(
            data=frame,
            x=panel.x,
            y="value",
            hue="series",
            marker="o",
            sort=panel.sort_x,
            ax=axis,
        )
    elif panel.plot_kind == "scatter":
        sns.scatterplot(
            data=frame,
            x=panel.x,
            y="value",
            hue="series",
            ax=axis,
        )
    elif panel.plot_kind == "bar":
        sns.barplot(
            data=frame,
            x=panel.x,
            y="value",
            hue="series",
            ax=axis,
        )
    else:
        raise ValueError(f"unknown plot kind: {panel.plot_kind}")

    axis.set_title(panel.title)
    axis.set_xlabel(panel.xlabel)
    axis.set_ylabel(panel.ylabel)


def layout_shape(n_panels: int, max_columns: int = 3) -> tuple[int, int]:
    if n_panels <= 0:
        raise ValueError("at least one panel is required")
    if max_columns <= 0:
        raise ValueError("max_columns must be positive")

    ncols = min(max_columns, n_panels)
    nrows = math.ceil(n_panels / ncols)
    return nrows, ncols


def draw(
    context: Mapping[str, pd.DataFrame],
    panels: Iterable[PanelSpec] = PANELS,
    outdir: str = "figs",
    max_columns: int = 3,
    filename: str | None = None,
) -> str:
    panels = tuple(panels)
    validate_panels(panels, context)

    nrows, ncols = layout_shape(len(panels), max_columns)
    sns.set_theme(style="dark")

    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 4.2 * nrows),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for axis, panel in zip(flat_axes, panels):
        draw_panel(axis, context, panel)

    for axis in flat_axes[len(panels):]:
        axis.set_visible(False)

    figure.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    if filename is None:
        filename = (
            f"all_panels_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
        )
    output_path = os.path.join(outdir, filename)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def discard_main():
    '''
    summaries = load_client_summaries(PATH)
    context = {
        "run_summary": to_run_summary(summaries),
        "request_metrics": load_request_metrics(REQUEST_PATH),
        "merged_summary": load_merged_summary(MERGED_PATH),
        "ttft_gap": client_engine_ttft_gap_ms(MERGED_PATH),
    }
    outdir = os.path.dirname(PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # old_path = draw(
    #     context,
    #     panels=OLD_PANELS,
    #     outdir=outdir,
    #     filename=f"all_panels_old_{timestamp}.png",
    # )
    # new_path = draw(
    #     context,
    #     panels=NEW_PANELS,
    #     outdir=outdir,
    #     filename=f"all_panels_new_{timestamp}.png",
    # )
    # print(old_path)
    # print(new_path)
    ttft_delta_path = draw(
        context,
        panels=TTFT_PANELS,
        outdir=outdir,
        filename=f"ttft_gap_{timestamp}.png",
    )
    print(ttft_delta_path)
    '''
    pass
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    add_run_args(arg_parser)
    args = arg_parser.parse_args()

    run = resolve_run(
        label=args.label,
        request_dir=args.request_dir,
        metrics_dir=args.metrics_dir,
        out_dir=args.out_dir,
    )

    summaries = load_client_summaries(
        str(run.out_dir / "result_Bn.json")
    )

    context = {
        "run_summary": to_run_summary(summaries),
        "request_metrics": load_request_metrics(
            str(run.out_dir / "merge_result_Bn.json")
        ),
        "ttft_gap": client_engine_ttft_gap_ms(
            str(run.out_dir / "merge_result_Bn.json")
        ),
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    outdir = str(run.out_dir)

    ttft_delta_path = draw(
        context,
        panels=TTFT_PANELS,
        outdir=outdir,
        filename=f"ttft_gap_{timestamp}.png",
    )
    print(ttft_delta_path)