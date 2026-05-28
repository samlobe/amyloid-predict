#!/usr/bin/env python3
"""Reproduce residue-level boxenplots using the same settings as boxenplots2.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Residue_code")

    out = g.agg(
        count=("Residue_code", "size"),
        amyloid_mean=("avg_amyloid_score", "mean"),
        amyloid_q125=("avg_amyloid_score", lambda s: s.quantile(0.125)),
        amyloid_q25=("avg_amyloid_score", lambda s: s.quantile(0.25)),
        amyloid_median=("avg_amyloid_score", "median"),
        amyloid_q75=("avg_amyloid_score", lambda s: s.quantile(0.75)),
        amyloid_q875=("avg_amyloid_score", lambda s: s.quantile(0.875)),
        llps_mean=("avg_LLPS_score", "mean"),
        llps_q125=("avg_LLPS_score", lambda s: s.quantile(0.125)),
        llps_q25=("avg_LLPS_score", lambda s: s.quantile(0.25)),
        llps_median=("avg_LLPS_score", "median"),
        llps_q75=("avg_LLPS_score", lambda s: s.quantile(0.75)),
        llps_q875=("avg_LLPS_score", lambda s: s.quantile(0.875)),
    ).reset_index()

    amyloid_order = (
        out.sort_values("amyloid_median", ascending=False)["Residue_code"].tolist()
    )
    llps_order = out.sort_values("llps_median", ascending=False)["Residue_code"].tolist()

    out["amyloid_median_rank"] = out["Residue_code"].map(
        {aa: i + 1 for i, aa in enumerate(amyloid_order)}
    )
    out["llps_median_rank"] = out["Residue_code"].map(
        {aa: i + 1 for i, aa in enumerate(llps_order)}
    )

    return out, amyloid_order, llps_order


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce residue boxenplots from residue-level score CSV.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    needed = {"Residue_code", "avg_amyloid_score", "avg_LLPS_score"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    summary, amyloid_order, llps_order = compute_summary(df)

    summary.to_csv(out_dir / "residue_boxenplot_summary.csv", index=False)
    (out_dir / "amyloid_residue_order.txt").write_text("\n".join(amyloid_order) + "\n", encoding="utf-8")
    (out_dir / "llps_residue_order.txt").write_text("\n".join(llps_order) + "\n", encoding="utf-8")

    sns.set(style="whitegrid")
    fontsize = 16

    plt.figure(figsize=(13, 3))
    sns.boxenplot(
        data=df,
        x="Residue_code",
        y="avg_amyloid_score",
        order=amyloid_order,
        color="firebrick",
        k_depth=3,
        showfliers=False,
    )
    plt.xlabel("amino acid", fontsize=fontsize)
    plt.ylabel("aggregation score\nin IDRome", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim(0, 0.9)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(out_dir / "amyloid_boxenplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(13, 3))
    sns.boxenplot(
        data=df,
        x="Residue_code",
        y="avg_LLPS_score",
        order=llps_order,
        color="rebeccapurple",
        k_depth=3,
        showfliers=False,
    )
    plt.xlabel("amino acid", fontsize=fontsize)
    plt.ylabel("LLPS score\nin IDRome", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim(0.1, 0.65)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(out_dir / "LLPS_boxenplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved reproducible residue boxenplot artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
