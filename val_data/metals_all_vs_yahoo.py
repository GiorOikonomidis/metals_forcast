"""
Overlay every target series in the parquet against Yahoo Finance where a Yahoo
series exists, one panel per id. Ours is drawn in red, Yahoo in blue (unit-
aligned). Most targets (tungsten, cobalt, magnesium, the rare earths) have no
Yahoo instrument, so those panels show our series alone.

Yahoo coverage (all others: ours only):
  XAU  Gold   -> GC=F  (USD/troy oz,  factor 1)
  XAG  Silver -> SI=F  (USD/troy oz,  factor 1)
  XCU  Copper -> HG=F  (USD/lb -> /16 for per ounce)
  IRON Iron   -> TIO=F (USD/metric ton, factor 1)

Units are taken from the caller-supplied table (UNITS). Two figures are written:
a linear-y and a log-y variant — the log one keeps the normal range readable
next to the known corruption spikes (IRON, ND, ...).

Run:
    python val_data/metals_all_vs_yahoo.py
    TARGET_VARIABLES_PATH=/path/to/target_variables.parquet python val_data/metals_all_vs_yahoo.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET_PARQUET = REPO_ROOT / "target_variables.parquet"

# id -> (display name, unit) from the supplied reference table
META = {
    "XAU":      ("Gold (spot)",            "per ounce"),
    "XAG":      ("Silver (spot)",          "per ounce"),
    "XCU":      ("Copper (spot)",          "per ounce"),
    "IRON62":   ("Iron Ore 62% Fe",        "per dry metric ton"),
    "IRON":     ("Iron Ore (generic)",     "per dry metric ton"),
    "TUNGSTEN": ("Tungsten",               "per ounce"),
    "LCO":      ("Cobalt",                 "per ounce"),
    "MG":       ("Magnesium",              "per ounce"),
    "ND":       ("Neodymium",              "per ounce"),
    "LTH":      ("Lanthanum",              "per ounce"),
    "DYS":      ("Dysprosium",             "per ounce"),
    "PRA":      ("Praseodymium",           "per ounce"),
    "TER":      ("Terbium",                "per ounce"),
}

# id -> (Yahoo ticker, factor applied to the Yahoo close to match our unit)
YAHOO = {
    "XAU":  ("GC=F", 1.0),
    "XAG":  ("SI=F", 1.0),
    "XCU":  ("HG=F", 1.0 / 16.0),   # per lb -> per ounce
    "IRON": ("TIO=F", 1.0),         # SGX iron ore, per metric ton
}


def load_ours(parquet_path: Path) -> dict[str, "pd.Series"]:
    """
    Load every id's close series from the long target parquet.

    Parameters
    ----------
    parquet_path : Path
        Long-format ``target_variables.parquet`` (``date, id, ..., close``).

    Returns
    -------
    dict[str, pd.Series]
        Maps each id to its close series indexed by date (NaN dropped, sorted).
    """
    df = pd.read_parquet(parquet_path, columns=["date", "id", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return {mid: g.set_index("date")["close"].dropna().sort_index()
            for mid, g in df.groupby("id")}


def fetch_yahoo(ids: list[str], start, end) -> "pd.DataFrame":
    """
    Download the Yahoo closes for the mapped subset of ids over a date span.

    Parameters
    ----------
    ids : list[str]
        Our ids present in the file; only those in ``YAHOO`` are fetched.
    start, end : Timestamp
        Date span (matched to our data) passed to yfinance.

    Returns
    -------
    pd.DataFrame
        Close prices, one column per Yahoo ticker (empty frame if none map).
    """
    tickers = [YAHOO[i][0] for i in ids if i in YAHOO]
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if data.ndim == 1:
        data = data.to_frame(tickers[0])
    return data


def plot_all(parquet_path: Path, out_dir: Path) -> None:
    """
    Draw one panel per target id — ours vs Yahoo where available — in both
    linear-y and log-y variants.

    Parameters
    ----------
    parquet_path : Path
        Path to our ``target_variables.parquet``.
    out_dir : Path
        Directory the two PNGs are written to.

    Returns
    -------
    None
        Writes ``metals_all_vs_yahoo.png`` and ``metals_all_vs_yahoo_log.png``.
    """
    ours = load_ours(parquet_path)
    ids = sorted(ours)

    start = min(s.index.min() for s in ours.values())
    end   = max(s.index.max() for s in ours.values()) + pd.Timedelta(days=1)
    yahoo = fetch_yahoo(ids, start, end)

    ncols = 3
    nrows = (len(ids) + ncols - 1) // ncols

    for logy in (False, True):
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.2 * nrows),
                                 squeeze=False)
        for ax, mid in zip(axes.flat, ids):
            name, unit = META.get(mid, (mid, "?"))
            our_s = ours[mid]
            xs = our_s[our_s > 0] if logy else our_s
            ax.plot(xs.index, xs.values, lw=1.0, color="#c0392b", label=f"ours ({mid})")

            note = ""
            if mid in YAHOO and not yahoo.empty:
                ticker, factor = YAHOO[mid]
                if ticker in yahoo.columns:
                    yh = (yahoo[ticker].dropna() * factor)
                    yh = yh[yh > 0] if logy else yh
                    ax.plot(yh.index, yh.values, lw=1.0, color="#2c7fb8",
                            alpha=0.85, label=f"Yahoo ({ticker})")
            else:
                note = "  (no Yahoo series)"

            if logy:
                ax.set_yscale("log")
            ax.set_title(f"{mid} · {name} — {unit}{note}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=7)
        for ax in axes.flat[len(ids):]:
            ax.axis("off")

        scale = "log-y" if logy else "linear-y"
        fig.suptitle(f"Target parquet vs Yahoo Finance — all targets ({scale})", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        suffix = "_log" if logy else ""
        out_path = out_dir / f"metals_all_vs_yahoo{suffix}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    target_path = Path(os.environ.get("TARGET_VARIABLES_PATH", DEFAULT_TARGET_PARQUET))
    if not target_path.is_file():
        raise SystemExit(f"target parquet not found: {target_path}\n"
                         f"set TARGET_VARIABLES_PATH to your target_variables.parquet")
    plot_all(target_path, OUT_DIR)
