"""
Overlay our target-parquet close price against Yahoo Finance for the three
metals we care about — gold (XAU), silver (XAG), copper (XCU) — one panel per
metal, on a common per-ounce unit, so our series can be validated against the
Yahoo ground truth (data gaps, unit mistakes and corruption spikes stand out).

Unit alignment (everything drawn as USD per ounce):
  gold   XAU  vs GC=F  — both USD / troy ounce            (Yahoo factor 1)
  silver XAG  vs SI=F  — both USD / troy ounce            (Yahoo factor 1)
  copper XCU  vs HG=F  — ours is per ounce, HG=F per lb   (Yahoo factor 1/16)

Run:
    python val_data/metals_ours_vs_yahoo.py
    TARGET_VARIABLES_PATH=/path/to/target_variables.parquet python val_data/metals_ours_vs_yahoo.py
"""
import os
from pathlib import Path

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET_PARQUET = REPO_ROOT / "target_variables.parquet"

# our id -> (display label, Yahoo ticker, factor applied to the YAHOO close to
# match our per-ounce unit)
METALS = {
    "XAU": ("Gold",   "GC=F", 1.0),
    "XAG": ("Silver", "SI=F", 1.0),
    "XCU": ("Copper", "HG=F", 1.0 / 16.0),   # Yahoo per lb -> per ounce
}


def load_ours(parquet_path: Path, ids: list[str]) -> dict[str, "pd.Series"]:
    """
    Load our close series for the requested ids from the long target parquet.

    Parameters
    ----------
    parquet_path : Path
        Long-format ``target_variables.parquet`` (``date, id, ..., close``).
    ids : list[str]
        Series ids to extract (e.g. ``["XAU", "XAG", "XCU"]``).

    Returns
    -------
    dict[str, pd.Series]
        Maps each id to its close series indexed by date (NaN rows dropped).
    """
    df = pd.read_parquet(parquet_path, columns=["date", "id", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return {mid: df[df["id"] == mid].set_index("date")["close"].dropna().sort_index()
            for mid in ids}


def plot_overlay(parquet_path: Path, out_dir: Path) -> None:
    """
    Draw one panel per metal overlaying our close vs the Yahoo close.

    Parameters
    ----------
    parquet_path : Path
        Path to our ``target_variables.parquet``.
    out_dir : Path
        Directory the PNG is written to.

    Returns
    -------
    None
        Writes ``metals_ours_vs_yahoo.png`` to ``out_dir``.
    """
    ours = load_ours(parquet_path, list(METALS))

    # match Yahoo's window to our data span (+1 day so the last row is included)
    start = min(s.index.min() for s in ours.values())
    end   = max(s.index.max() for s in ours.values()) + pd.Timedelta(days=1)
    tickers = [t for _, t, _ in METALS.values()]
    yahoo = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]

    fig, axes = plt.subplots(len(METALS), 1, figsize=(12, 10), sharex=True)
    for ax, (mid, (label, ticker, factor)) in zip(axes, METALS.items()):
        our_s = ours[mid]
        yh_s  = (yahoo[ticker].dropna() * factor)
        ax.plot(our_s.index, our_s.values, lw=1.1, color="#c0392b",
                label=f"ours ({mid})")
        ax.plot(yh_s.index, yh_s.values, lw=1.1, color="#2c7fb8",
                label=f"Yahoo ({ticker})", alpha=0.85)
        ax.set_title(f"{label} — USD per ounce")
        ax.set_ylabel("close (USD/oz)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("Date")
    fig.suptitle("Target parquet vs Yahoo Finance — metal close prices", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_path = out_dir / "metals_ours_vs_yahoo.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    target_path = Path(os.environ.get("TARGET_VARIABLES_PATH", DEFAULT_TARGET_PARQUET))
    if not target_path.is_file():
        raise SystemExit(f"target parquet not found: {target_path}\n"
                         f"set TARGET_VARIABLES_PATH to your target_variables.parquet")
    plot_overlay(target_path, OUT_DIR)
