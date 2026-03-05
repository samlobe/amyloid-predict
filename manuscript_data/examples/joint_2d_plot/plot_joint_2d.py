#!/usr/bin/env python3
"""Generate a joint amyloid/LLPS 2D plot with IDR background contours."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None


def load_per_res_average(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0)

    # Normalize potential residue index storage patterns.
    if not pd.api.types.is_numeric_dtype(df.index):
        for col in ("residue", "position", "res", "aa_idx"):
            if col in df.columns:
                df = df.set_index(col)
                break

    index = pd.to_numeric(df.index, errors="coerce")
    num = df.apply(pd.to_numeric, errors="coerce")
    avg = num.mean(axis=1)

    out = pd.DataFrame({"residue": index, "avg": avg}).dropna()
    out["residue"] = out["residue"].astype(int)
    out = out.groupby("residue", as_index=True)["avg"].mean().sort_index()
    return out


def contour_levels(hist: np.ndarray) -> tuple[float, float]:
    flat = np.sort(hist.ravel())
    cdf = np.cumsum(flat)
    l68 = flat[np.argmin(np.abs(cdf - (1 - 0.68)))]
    l90 = flat[np.argmin(np.abs(cdf - (1 - 0.90)))]
    return l68, l90


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot averaged per-residue amyloid vs LLPS scores.")
    parser.add_argument("--amyloid_csv", required=True)
    parser.add_argument("--llps_csv", required=True)
    parser.add_argument(
        "--hist_npy",
        required=False,
        help="Optional path to 2D histogram array (.npy) for IDR background density.",
    )
    parser.add_argument(
        "--xedges_npy",
        required=False,
        help="Optional x-bin edges (.npy). Must be set if --hist_npy is provided.",
    )
    parser.add_argument(
        "--yedges_npy",
        required=False,
        help="Optional y-bin edges (.npy). Must be set if --hist_npy is provided.",
    )
    parser.add_argument("--protein", required=True)
    parser.add_argument("--start", type=int, required=False)
    parser.add_argument("--end", type=int, required=False)
    parser.add_argument(
        "--out_png",
        required=False,
        help="Optional output PNG path. Default: <protein>_joint_2d.png in the current working directory.",
    )
    args = parser.parse_args()

    amyloid = load_per_res_average(Path(args.amyloid_csv).expanduser().resolve())
    llps = load_per_res_average(Path(args.llps_csv).expanduser().resolve())

    df = pd.concat([amyloid.rename("amyloid"), llps.rename("llps")], axis=1, join="inner").dropna()
    if args.start is not None:
        df = df[df.index >= args.start]
    if args.end is not None:
        df = df[df.index <= args.end]
    if df.empty:
        raise ValueError("No overlapping residue rows after applying residue range filters.")

    use_background = bool(args.hist_npy)
    if use_background and not (args.xedges_npy and args.yedges_npy):
        raise ValueError("If --hist_npy is set, both --xedges_npy and --yedges_npy are required.")

    hist = xedges = yedges = None
    if use_background:
        hist = np.load(Path(args.hist_npy).expanduser().resolve())
        xedges = np.load(Path(args.xedges_npy).expanduser().resolve())
        yedges = np.load(Path(args.yedges_npy).expanduser().resolve())
        if gaussian_filter is not None:
            hist = gaussian_filter(hist, sigma=2)
        hist = hist / hist.sum()
        l68, l90 = contour_levels(hist)

    fig, (ax, cax) = plt.subplots(
        ncols=2, figsize=(7, 6), gridspec_kw={"width_ratios": [10, 0.5], "wspace": 0.08}
    )

    if use_background:
        heat = np.sqrt(hist)
        ax.imshow(
            heat.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
            cmap="Greys",
        )

        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        ax.contour(X, Y, hist.T, levels=[l90, l68], colors=["black", "black"], linestyles="dotted")

    colors = np.linspace(0, 1, len(df))
    ax.scatter(df["amyloid"], df["llps"], c=colors, cmap="viridis", s=40, alpha=0.9)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("residue progression")

    min_res, max_res = int(df.index.min()), int(df.index.max())
    ticks = np.linspace(0, 1, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([int(round(min_res + t * (max_res - min_res))) for t in ticks])

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("aggregation score")
    ax.set_ylabel("LLPS score")
    title_range = f"{min_res}-{max_res}"
    ax.set_title(f"{args.protein}: {title_range}")

    out_name = args.out_png if args.out_png else f"{args.protein}_joint_2d.png"
    out_path = Path(out_name).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
