"""
Plot the close-price history of every target series in the long-format target
parquet. One subplot per id, drawn over the full available date range, so the
whole panel of 13 metals can be eyeballed for magnitude jumps / unit mixing at
a glance.

Run:
    python val_data/plot_target_close.py
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_PARQUET = REPO_ROOT / "target_variables.parquet"
OUT_DIR = Path(__file__).resolve().parent


def plot_all_targets(parquet_path: Path, out_dir: Path) -> None:
    """
    Draw a grid of close-price line plots, one per target id.

    Parameters
    ----------
    parquet_path : Path
        Long-format target parquet with columns [date, id, ..., close].
    out_dir : Path
        Directory the two PNGs are written to (linear + log-y variants).

    Returns
    -------
    None
        Writes ``target_close.png`` and ``target_close_log.png`` to ``out_dir``.
    """
    df = pd.read_parquet(parquet_path)
    df = df.sort_values(["id", "date"])
    ids = sorted(df["id"].unique())

    n = len(ids)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    for logy in (False, True):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                                 squeeze=False)
        for ax, tid in zip(axes.flat, ids):
            g = df[df["id"] == tid]
            ax.plot(g["date"], g["close"], lw=0.9, color="#1f77b4")
            ax.set_title(tid, fontsize=10)
            ax.tick_params(labelsize=7)
            if logy:
                ax.set_yscale("log")
        # blank any unused cells in the final row
        for ax in axes.flat[n:]:
            ax.axis("off")

        suffix = "_log" if logy else ""
        scale = "log-y" if logy else "linear-y"
        fig.suptitle(f"Target close price over time ({scale})", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out_path = out_dir / f"target_close{suffix}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    plot_all_targets(TARGET_PARQUET, OUT_DIR)
