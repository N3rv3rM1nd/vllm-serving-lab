from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VLLM_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = VLLM_ROOT / "raw" / "config"
REQUEST_ROOT = VLLM_ROOT / "raw" / "requests"
METRIC_ROOT = VLLM_ROOT / "raw" / "metrics"
RESULT_ROOT = VLLM_ROOT / "results"

@dataclass(frozen=True)
class RunJob:
    batch: int
    rep: int | None
    request_path: Path
    before_path: Path
    after_path: Path
    n_requests: int

@dataclass(frozen=True)
class RunPaths:
    label: str
    request_dir: Path
    metrics_dir: Path
    out_dir: Path
    meta: dict[str, Any]
    jobs: tuple[RunJob, ...]

def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--request-dir")
    parser.add_argument("--metrics-dir")
    parser.add_argument("--out-dir")

def _metric_path(
        metrics_dir: Path,
        batch: int,
        rep: int | None,
        phase: str,
) -> Path:
    suffix = f"_r{rep}" if rep is not None else ""
    direct = metrics_dir / f"metrics_B{batch}{suffix}_{phase}.txt"
    if direct.exists():
        return direct

    candidates = sorted(
        metrics_dir.glob(f"metrics_B{batch}*_{phase}.txt")
    )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"cannot resolve metrics for B={batch}, rep={rep}, phase={phase}:"
        f"{[path.name for path in candidates]}"
    )

def resolve_run(
    label: str,
    request_dir: str | None = None,
    metrics_dir: str | None = None,
    out_dir: str | None = None,
) -> RunPaths:
    meta_path = CONFIG_DIR / f"{label}__sweep_meta.json"
    if not meta_path.exists():
        raise ValueError(f"config not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    request_path = Path(request_dir) if request_dir else REQUEST_ROOT / label
    metric_path = Path(metrics_dir) if metrics_dir else METRIC_ROOT / label
    result_path = Path(out_dir) if out_dir else RESULT_ROOT / label

    manifest_path = request_path / "manifest.json"

    if manifest_path.exists():
        entries = json.loads(manifest_path.read_text(encoding="utf-8")) 
        # read_text内部封装的就是with open 运行完之后会自动关闭文件句柄
    else:
        entries = [
            {
                "file": f"result_B{batch}.json",
                "B": batch, 
                "rep": None,
                "n_requests": meta["N"],
            }
            for batch in meta["B_list"]
        ]

    jobs = []

    for entry in entries:
        batch = int(entry["B"])
        rep = entry.get("rep")
        rep = int(rep) if rep is not None else None
        n_requests = int(entry.get("n_requests", meta["N"]))

        request_file = request_path / entry["file"]
        before_file = _metric_path(metric_path, batch, rep, "before")
        after_file = _metric_path(metric_path, batch, rep, "after")

        for path in (request_file, before_file, after_file):
            if not path.exists():
                raise FileNotFoundError(path)
            
        jobs.append(
            RunJob(
                batch=batch,
                rep=rep,
                request_path=request_file,
                before_path=before_file,
                after_path=after_file,
                n_requests=n_requests,
            )
        )

    jobs.sort(
        key=lambda job: (
            job.batch,
            -1 if job.rep is None else job.rep,
        )
    )

    if len({job.batch for job in jobs}) != len(jobs):
        raise ValueError(
            f"{label}: duplicate B values in one run;"
            "use one label per repeat or add repeat to the output key"
        )

    expected_batches = {
        int(batch)
        for batch in meta.get("B_list", [])
    }
    if expected_batches and expected_batches != {job.batch for job in jobs}:
        raise ValueError(
            f"{label}: Config B_list and discovered files differ"
        )

    return RunPaths(
        label=label,
        request_dir=request_path,
        metrics_dir=metric_path,
        out_dir=result_path,
        meta=meta,
        jobs=tuple(jobs),
    )