#!/usr/bin/env python3
"""Plot benchmark throughput vs n from a CSV produced by `./<bench> --csv`.

Usage:
    ./build/sasum --csv > data/sasum.csv
    python tools/plot.py data/sasum.csv                 # batch (steady state)
    python tools/plot.py data/sasum.csv --metric cold   # L2-flushed, DRAM-fed

The cold/batch gap at small n is the L2 story: where `batch` rises above DRAM
peak (data served from L2) while `cold` stays honest.
"""
import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV file from `./<bench> --csv`")
    ap.add_argument("--metric", choices=["cold", "batch"], default="batch")
    ap.add_argument("-o", "--out", help="output PNG (default: <csv>_<metric>.png)")
    args = ap.parse_args()

    rate_col = f"{args.metric}_rate"
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            rate = row.get(rate_col, "")
            if not rate:  # blank == FAILED -> leave a gap
                continue
            series[row["kernel"]].append((int(row["n"]), float(rate)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for kernel, points in series.items():
        points.sort()
        ax.plot([n for n, _ in points], [r for _, r in points], marker="o", label=kernel)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("n")
    ax.set_ylabel(f"{args.metric} throughput")
    ax.set_title(args.csv)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    out = args.out or f"{args.csv.rsplit('.', 1)[0]}_{args.metric}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
