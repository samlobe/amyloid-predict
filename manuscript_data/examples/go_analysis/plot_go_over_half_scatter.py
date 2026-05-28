#!/usr/bin/env python3
"""Recreate Figure-3-style GO category scatter plots from compact summary CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def color_points(df: pd.DataFrame, x_thr: float, y_thr: float):
    colors = []
    for _, row in df.iterrows():
        x = row["amyloid_over_half_sum"]
        y = row["LLPS_over_half_sum"]
        if x > x_thr and y > y_thr:
            colors.append("mediumvioletred")
        elif x > x_thr:
            colors.append("firebrick")
        elif y > y_thr:
            colors.append("rebeccapurple")
        else:
            colors.append("gray")
    return colors


def make_plot(df: pd.DataFrame, title: str, out_path: Path, x_thr: float, y_thr: float) -> None:
    plt.figure(figsize=(6, 6))
    colors = color_points(df, x_thr=x_thr, y_thr=y_thr)
    plt.scatter(df["amyloid_over_half_sum"], df["LLPS_over_half_sum"], c=colors)
    plt.xlabel("% with aggregation scores > 0.5", fontsize=14)
    plt.ylabel("% with LLPS scores > 0.5", fontsize=14)
    plt.title(title, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.grid(alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot compact GO over-half summary tables.")
    parser.add_argument("--cellcom_csv", required=True)
    parser.add_argument("--molfunc_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    cellcom = pd.read_csv(Path(args.cellcom_csv).expanduser().resolve())
    molfunc = pd.read_csv(Path(args.molfunc_csv).expanduser().resolve())
    out_dir = Path(args.out_dir).expanduser().resolve()

    make_plot(
        cellcom,
        title="Cellular localization GO categories",
        out_path=out_dir / "amyloid_vs_LLPS_cellcom.png",
        x_thr=10.0,
        y_thr=11.2,
    )
    make_plot(
        molfunc,
        title="Molecular function GO categories",
        out_path=out_dir / "amyloid_vs_LLPS_molfunc.png",
        x_thr=10.0,
        y_thr=15.0,
    )

    print(f"Saved GO summary plots in: {out_dir}")


if __name__ == "__main__":
    main()
