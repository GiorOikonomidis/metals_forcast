"""
Download gold, silver and copper close prices from Yahoo Finance and plot the
three on a single axis in a common unit (USD per ounce).

Unit note: gold (GC=F) and silver (SI=F) are quoted in USD per troy ounce, but
copper (HG=F) is quoted in USD per pound. To put all three on the same per-ounce
basis, copper is divided by 16 (1 lb = 16 oz). (Copper uses the avoirdupois
ounce while gold/silver use the troy ounce — a ~10% difference — so treat the
copper line as an order-of-magnitude comparison, not an exact troy-ounce price.)

Run:
    python val_data/metals_yahoo_close.py
"""
from pathlib import Path

import yfinance as yf
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent
PERIOD = "10y"   # yfinance period string (e.g. "5y", "10y", "max")

# ticker -> (display label, per-ounce conversion factor applied to the close)
METALS = {
    "GC=F": ("Gold",   1.0),        # USD / troy ounce
    "SI=F": ("Silver", 1.0),        # USD / troy ounce
    "HG=F": ("Copper", 1.0 / 16.0), # USD / pound -> USD / ounce
}


def fetch_close(tickers: list[str], period: str) -> "pd.DataFrame":
    """
    Download daily close prices for a set of Yahoo Finance tickers.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance ticker symbols (e.g. ``["GC=F", "SI=F", "HG=F"]``).
    period : str
        yfinance period string (e.g. ``"10y"``, ``"max"``).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, one column per ticker holding its close price. Columns
        are named by ticker; missing rows are left as NaN (not filled).
    """
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    close = data["Close"]
    # yf returns a Series when a single ticker is requested — normalise to frame
    if close.ndim == 1:
        close = close.to_frame(tickers[0])
    return close[tickers]


def plot_metals(period: str, out_dir: Path) -> None:
    """
    Fetch gold/silver/copper closes, convert to USD per ounce, and plot them.

    Parameters
    ----------
    period : str
        yfinance period string passed to :func:`fetch_close`.
    out_dir : Path
        Directory the two PNGs are written to (linear + log-y variants).

    Returns
    -------
    None
        Writes ``metals_close_per_ounce.png`` and
        ``metals_close_per_ounce_log.png`` to ``out_dir``.
    """
    close = fetch_close(list(METALS), period)

    colors = {"Gold": "#d4af37", "Silver": "#9ca3af", "Copper": "#b87333"}

    for logy in (False, True):
        fig, ax = plt.subplots(figsize=(11, 6))
        for ticker, (label, factor) in METALS.items():
            series = close[ticker].dropna() * factor
            ax.plot(series.index, series.values, label=label, lw=1.2,
                    color=colors[label])
        ax.set_ylabel("Close price (USD per ounce)")
        ax.set_xlabel("Date")
        if logy:
            ax.set_yscale("log")
        scale = "log-y" if logy else "linear-y"
        ax.set_title(f"Gold / Silver / Copper close — USD per ounce ({scale})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        suffix = "_log" if logy else ""
        out_path = out_dir / f"metals_close_per_ounce{suffix}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    plot_metals(PERIOD, OUT_DIR)
