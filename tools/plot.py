#!/usr/bin/env python3
"""Plot benchmark throughput vs n, one figure per algorithm family.

Each figure shows every implementation in the family as a distinct color. Cold
(L2-flushed, DRAM-fed) and batch (warm, steady-state) sit side-by-side on a
shared y-axis, so the small-n divergence is obvious: where `batch` climbs above
DRAM peak the data is being served from L2; `cold` stays honest. cuBLAS
reference lines are dashed so you can read how close each kernel gets.

Usage:
    ./build/sasum --csv > data/sasum.csv
    ./build/saxpy --csv > data/saxpy.csv
    ./build/sgemm --csv > data/sgemm.csv

    python tools/plot.py data/sasum.csv data/saxpy.csv data/sgemm.csv
    python tools/plot.py data/*.csv --peak 936       # 3090 DRAM peak (GB/s plots)
    python tools/plot.py data/sasum.csv --metric batch   # single plot, batch only

Reads the CSV emitted by `./<bench> --csv`:
    n,kernel,rel_err,cold_ms,cold_rate,batch_ms,batch_rate
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# The CSV rate columns are unit-agnostic, so infer the unit from the family name.
UNITS_BY_FAMILY = {"sgemm": "TFLOP/s", "saxpy": "GB/s", "sasum": "GB/s"}


def family_of(path: Path) -> str:
    """`data/sgemm.csv` / `sgemm_run2.csv` -> `sgemm`."""
    return path.stem.split("_")[0].lower()


def load(path: Path, metrics):
    """kernel -> metric -> sorted [(n, rate)], preserving first-seen kernel order.

    Rows whose rate cell is blank (a FAILED kernel) are skipped, leaving a gap.
    """
    data = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            series = data.setdefault(row["kernel"], {m: [] for m in metrics})
            for m in metrics:
                rate = row.get(f"{m}_rate", "")
                if rate:
                    series[m].append((int(row["n"]), float(rate)))
    for series in data.values():
        for points in series.values():
            points.sort()
    return data


def plot_family(path: Path, metrics, units, peak, logy, outdir) -> None:
    data = load(path, metrics)
    if not data:
        print(f"skip {path}: no data")
        return

    kernels = list(data)
    cmap = plt.get_cmap("tab10" if len(kernels) <= 10 else "tab20")
    color = {kernel: cmap(i % cmap.N) for i, kernel in enumerate(kernels)}

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6.5 * len(metrics), 5), sharey=True, squeeze=False
    )
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        for kernel in kernels:
            points = data[kernel][metric]
            if not points:
                continue
            is_ref = "cublas" in kernel.lower()  # dash + square-marker the references
            ax.plot(
                [n for n, _ in points],
                [r for _, r in points],
                marker="s" if is_ref else "o",
                markersize=4,
                linewidth=2.0 if is_ref else 1.5,
                linestyle="--" if is_ref else "-",
                color=color[kernel],
                label=kernel,
            )
        if peak is not None and units == "GB/s":
            ax.axhline(peak, ls=":", color="0.4", lw=1.0)
            ax.text(
                0.99, peak, f" {peak:g} GB/s peak ",
                transform=ax.get_yaxis_transform(),
                color="0.4", fontsize=8, ha="right", va="bottom",
            )
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set_title(metric)
        ax.set_xlabel("n")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(f"throughput ({units})")

    # One shared legend below, deduped across subplots so every kernel appears.
    handles, labels, seen = [], [], set()
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles, labels, loc="lower center", ncol=min(len(labels), 5),
        frameon=False, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(path.stem)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out = (outdir or path.parent) / f"{path.stem}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv", nargs="+", type=Path, help="CSV file(s) from `./<bench> --csv`")
    ap.add_argument("--metric", choices=["cold", "batch", "both"], default="both")
    ap.add_argument("--units", help="y-axis units (default: inferred from the filename)")
    ap.add_argument("--peak", type=float, help="draw a horizontal DRAM-peak line (GB/s plots)")
    ap.add_argument("--logy", action="store_true", help="log-scale the y-axis")
    ap.add_argument("--outdir", type=Path, help="output dir (default: next to the CSV)")
    args = ap.parse_args()

    metrics = ["cold", "batch"] if args.metric == "both" else [args.metric]
    for path in args.csv:
        units = args.units or UNITS_BY_FAMILY.get(family_of(path), "rate")
        plot_family(path, metrics, units, args.peak, args.logy, args.outdir)


if __name__ == "__main__":
    main()
