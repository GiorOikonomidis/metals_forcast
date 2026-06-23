"""
Dataset analytical report.
Reads the three parquet cases and produces figures + a printed summary.
Output: datasets/report/*.png
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PARENT  = Path(__file__).resolve().parent.parent
OUT_DIR = str(PARENT / "datasets" / "report")
CASES     = ["case_1_agg_news", "case_2_mask", "case_3_discard"]
CASE_LABELS = ["Case 1\nAgg news", "Case 2\nMask", "Case 3\nDiscard"]
INDEX_FEATURES = [
    "EMA_12", "EMA_26", "MACD", "RSI", "Stoch_K", "Stoch_D",
    "Williams_R", "ROC", "Daily_Return", "Volatility", "Movement",
]
os.makedirs(OUT_DIR, exist_ok=True)


# ── load ──────────────────────────────────────────────────────────────────────

def load_case(case: str) -> dict:
    base = str(PARENT / "datasets" / case)
    return {
        "target":   pd.read_parquet(os.path.join(base, "target.parquet")),
        "feat":     pd.read_parquet(os.path.join(base, "feature_covariates.parquet")),
        "dyn":      pd.read_parquet(os.path.join(base, "dynamic_covariates.parquet")),
    }

print("Loading datasets...")
data = {c: load_case(c) for c in CASES}


# ── 0. printed summary ────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)
for case, d in data.items():
    tgt, feat, dyn = d["target"], d["feat"], d["dyn"]
    dates = pd.to_datetime(tgt["date"])
    news_nan = dyn["prob_positive"].isna().sum()
    company_cols = [c for c in dyn.columns if "__" in c]
    close_cols   = [c for c in company_cols if c.endswith("__Close")]
    price_nan    = dyn[close_cols].isna().sum().sum()
    print(f"\n{case}")
    print(f"  Rows          : {len(tgt)}")
    print(f"  Date range    : {dates.iloc[0].date()} to {dates.iloc[-1].date()}")
    print(f"  Feat cols     : {list(feat.columns)}")
    print(f"  Dyn cols      : {len(dyn.columns)}  ({len(close_cols)} companies, {len(company_cols)} company feature cols total)")
    print(f"  News NaN rows : {news_nan}  ({100*news_nan/len(dyn):.1f}%)")
    print(f"  Price NaN cells: {price_nan}")
    if "trading_day" in feat.columns:
        td = feat["trading_day"]
        print(f"  Trading days  : {int(td.sum())}  |  Non-trading: {int((td==0).sum())}")
print("\n" + "=" * 70)


# ── helpers ───────────────────────────────────────────────────────────────────

def savefig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── fig 1: NDX close price ────────────────────────────────────────────────────

print("\nFig 1: NDX Close price...")
tgt = data["case_1_agg_news"]["target"]
dates = pd.to_datetime(tgt["date"])
close = tgt["Close"]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(dates, close, linewidth=0.8, color="#1f77b4")
ax.set_title("^NDX Close Price (trading days)", fontsize=13)
ax.set_ylabel("Close")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
fig.tight_layout()
savefig("fig1_ndx_close.png")


# ── fig 2: case row counts ────────────────────────────────────────────────────

print("Fig 2: Case row counts...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

row_counts = [len(data[c]["target"]) for c in CASES]
bars = axes[0].bar(CASE_LABELS, row_counts, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.5)
axes[0].set_title("Total rows per case")
axes[0].set_ylabel("Rows")
for bar, val in zip(bars, row_counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 str(val), ha="center", va="bottom", fontsize=10)

# news nan per case
nan_counts = [data[c]["dyn"]["prob_positive"].isna().sum() for c in CASES]
nan_pcts   = [100*n/len(data[c]["target"]) for c, n in zip(CASES, nan_counts)]
bars2 = axes[1].bar(CASE_LABELS, nan_pcts, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.5)
axes[1].set_title("News NaN % per case")
axes[1].set_ylabel("% rows with no news")
for bar, val in zip(bars2, nan_pcts):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
axes[1].yaxis.grid(True, alpha=0.3)
axes[0].yaxis.grid(True, alpha=0.3)
fig.tight_layout()
savefig("fig2_case_overview.png")


# ── fig 3: news coverage timeline ────────────────────────────────────────────

print("Fig 3: News coverage timeline...")

# derive news publication dates from Case 2 parquet (news kept on original dates)
_dyn2     = data["case_2_mask"]["dyn"]
_dyn2_dates = pd.to_datetime(_dyn2["date"] if "date" in _dyn2.columns else _dyn2.index)
_news_idx = _dyn2_dates[_dyn2["prob_positive"].notna().values].sort_values()

def _count_window(t, next_t):
    lo = _news_idx.searchsorted(t)
    hi = _news_idx.searchsorted(next_t) if next_t is not None else len(_news_idx)
    return hi - lo

def _news_counts_case1(trading_dates):
    td = trading_dates.sort_values()
    counts = []
    for i, t in enumerate(td):
        next_t = td[i + 1] if i + 1 < len(td) else None
        counts.append(_count_window(t, next_t))
    return pd.Series(counts, index=td)

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

for ax, case, label in zip(axes, CASES, CASE_LABELS):
    dyn   = data[case]["dyn"]
    dates = pd.to_datetime(data[case]["target"]["date"])

    if case == "case_1_agg_news":
        counts = _news_counts_case1(dates)
        ymax   = 5
    else:
        counts = pd.Series(dyn["prob_positive"].notna().astype(int).values, index=dates)
        ymax   = 2

    # bar plot of counts
    colors = ["#d62728" if v == 0 else "#2ca02c" for v in counts.values]
    ax.bar(counts.index, counts.values, width=2, color=colors, alpha=0.7)

    # find isolated 0s (not part of the 2026 block) and annotate with date
    zero_dates = counts[counts == 0].index
    last_news  = counts[counts > 0].index.max()   # last date with news
    end_gap_start = counts[(counts == 0) & (counts.index > last_news)].index.min()

    # annotate isolated 0s (before the end gap)
    for d in zero_dates:
        if d <= last_news:
            ax.annotate(str(d.date()), xy=(d, 0), xytext=(d, ymax * 0.4),
                        fontsize=7, color="#d62728", ha="center",
                        arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8))

    # shade and annotate the end gap only
    ax.axvspan(end_gap_start, dates.max(), color="#d62728", alpha=0.12)
    ax.axvline(end_gap_start, color="#d62728", linestyle="--", linewidth=1.2)
    ax.annotate(f"News ends\n{last_news.date()}", xy=(end_gap_start, ymax * 0.7),
                xytext=(end_gap_start - pd.Timedelta(days=500), ymax * 0.75),
                fontsize=7.5, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))

    ax.set_ylim(-0.1, ymax)
    ax.set_yticks(range(ymax))
    ax.set_ylabel("News days\nin window", fontsize=8)
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    total   = len(counts)
    missing = (counts == 0).sum()
    ax.set_title(f"{label.replace(chr(10), ' ')}  —  {total} rows  |  {missing} with no news ({100*missing/total:.1f}%)",
                 fontsize=9, loc="left")

axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].tick_params(axis="x", labelsize=9)
axes[-1].set_xlabel("Year", fontsize=9)

fig.suptitle("News days used per trading day — all cases  (Case 1: count of days in window  |  Cases 2 & 3: 0 or 1)", fontsize=11)
fig.tight_layout()
savefig("fig3_news_coverage.png")

# keep dyn1 / dates1 available for later figures
dyn1   = data["case_1_agg_news"]["dyn"]
dates1 = pd.to_datetime(data["case_1_agg_news"]["target"]["date"])


# ── fig 4: sentiment distribution — one PNG per case ─────────────────────────

def _plot_sentiment(case, dyn, dates, fname):
    is_case1   = case == "case_1_agg_news"
    dyn_news   = dyn[dyn["prob_positive"].notna()]
    news_dates = dates[dyn["prob_positive"].notna().values].values
    x_label    = ("Summed probability across window days\n(>1.0 possible: sum over multiple days)"
                  if is_case1 else "Probability for this day (0.0 to 1.0)")
    pie_title  = ("Dominant label\n(argmax of summed window probs)"
                  if is_case1 else "Dominant label per day")
    colors_pie = {"positive": "#2ca02c", "negative": "#d62728", "neutral": "#7f7f7f"}

    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # pie
    ax_pie = fig.add_subplot(gs[0, 0])
    lc = dyn_news["label"].value_counts()
    ax_pie.pie(lc.values, labels=lc.index,
               colors=[colors_pie.get(l, "#aec7e8") for l in lc.index],
               autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    ax_pie.set_title(pie_title, fontsize=9)

    # histograms
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_neg = fig.add_subplot(gs[0, 2])
    ax_neu = fig.add_subplot(gs[1, 0])
    for ax, col, color, title in [
        (ax_pos, "prob_positive", "#2ca02c", "Positive probability"),
        (ax_neg, "prob_negative", "#d62728", "Negative probability"),
        (ax_neu, "prob_neutral",  "#7f7f7f", "Neutral probability"),
    ]:
        vals = dyn_news[col].dropna()
        ax.hist(vals, bins=40, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"mean = {vals.mean():.2f}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(x_label, fontsize=7)
        ax.set_ylabel("Number of rows with news", fontsize=8)
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    # rolling mean time series
    ax_ts = fig.add_subplot(gs[1, 1:])
    roll = pd.DataFrame({
        "pos": pd.Series(dyn_news["prob_positive"].values, index=news_dates),
        "neg": pd.Series(dyn_news["prob_negative"].values, index=news_dates),
        "neu": pd.Series(dyn_news["prob_neutral"].values,  index=news_dates),
    }).rolling(60).mean()
    ax_ts.plot(roll.index, roll["pos"], color="#2ca02c", linewidth=1.5, label="Positive")
    ax_ts.plot(roll.index, roll["neg"], color="#d62728", linewidth=1.5, label="Negative")
    ax_ts.plot(roll.index, roll["neu"], color="#7f7f7f", linewidth=1.5, label="Neutral")
    ax_ts.set_title("60-day rolling mean of sentiment probabilities", fontsize=10)
    ax_ts.set_xlabel("Year", fontsize=9)
    ax_ts.set_ylabel("Probability (rolling mean)", fontsize=9)
    ax_ts.legend(fontsize=10, loc="upper left")
    ax_ts.yaxis.grid(True, alpha=0.3)
    ax_ts.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts.tick_params(axis="x", labelsize=9, rotation=30)
    ax_ts.set_axisbelow(True)

    label = dict(zip(CASES, CASE_LABELS))[case].replace("\n", " ")
    fig.suptitle(f"News sentiment analysis — {label}", fontsize=13)
    savefig(fname)


print("Fig 4: Sentiment distributions (per case)...")
for case, label in zip(CASES, CASE_LABELS):
    fname = f"fig4_{case}.png"
    _plot_sentiment(case, data[case]["dyn"], pd.to_datetime(data[case]["target"]["date"]), fname)


# ── fig 4b: sentiment per-t vs rolling — all 3 cases ─────────────────────────

print("Fig 4b: Sentiment per-t vs rolling, all 3 cases...")

SENT_COLS   = ["prob_positive", "prob_negative", "prob_neutral"]
SENT_COLORS = ["#2ca02c", "#d62728", "#7f7f7f"]
SENT_LABELS = ["Positive", "Negative", "Neutral"]
CASE_TITLES = ["Case 1 — Agg news\n(probs summed over window)", "Case 2 — Mask\n(probs on original date)", "Case 3 — Discard\n(probs on trading day only)"]

fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex="col")

for col_idx, (case, title) in enumerate(zip(CASES, CASE_TITLES)):
    d     = data[case]
    dyn   = d["dyn"]
    tgt   = d["target"]
    dates = pd.to_datetime(tgt["date"])
    mask  = dyn["prob_positive"].notna().values
    news_dates = dates[mask].values
    dyn_news   = dyn[dyn["prob_positive"].notna()]

    for row_idx, label in enumerate(["Per-t (raw)", "60-day rolling mean"]):
        ax = axes[row_idx, col_idx]

        for scol, scolor, slabel in zip(SENT_COLS, SENT_COLORS, SENT_LABELS):
            vals = pd.Series(dyn_news[scol].values, index=news_dates)
            if row_idx == 0:
                ax.plot(vals.index, vals.values, color=scolor, linewidth=0.4,
                        alpha=0.6, label=slabel)
            else:
                rolled = vals.rolling(60).mean()
                ax.plot(rolled.index, rolled.values, color=scolor, linewidth=1.8,
                        label=slabel)

        if row_idx == 0:
            ax.set_title(title, fontsize=9)
            ax.set_ylabel("Summed prob (raw)", fontsize=8)
        else:
            ax.set_ylabel("Summed prob (rolling mean)", fontsize=8)
            ax.set_xlabel("Year", fontsize=8)
            ax.xaxis.set_major_locator(mdates.YearLocator(4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(axis="x", labelsize=8, rotation=30)

        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        if col_idx == 2:
            ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Sentiment probabilities — raw per trading day vs 60-day rolling mean", fontsize=13)
fig.tight_layout()
savefig("fig4b_sentiment_cases.png")


# ── fig 5: technical indicator distributions ──────────────────────────────────

print("Fig 5: Technical indicator distributions...")
dyn1_idx = dyn1[INDEX_FEATURES].dropna()

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(INDEX_FEATURES):
    vals = dyn1_idx[col].dropna()
    axes[i].hist(vals, bins=50, color="#1f77b4", alpha=0.8, edgecolor="none")
    axes[i].set_title(col, fontsize=10)
    axes[i].axvline(vals.mean(),   color="black",  linestyle="--", linewidth=1, label=f"μ={vals.mean():.2f}")
    axes[i].axvline(vals.median(), color="orange", linestyle=":",  linewidth=1, label=f"med={vals.median():.2f}")
    axes[i].legend(fontsize=6)
    axes[i].yaxis.grid(True, alpha=0.3)
    axes[i].set_axisbelow(True)

axes[-1].set_visible(False)
fig.suptitle("Technical indicator distributions (Case 1, trading days)", fontsize=13)
fig.tight_layout()
savefig("fig5_tech_indicators.png")


# ── fig 6: technical indicator time series ────────────────────────────────────

print("Fig 6: Technical indicators over time...")
fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True)
axes = axes.flatten()
for i, col in enumerate(INDEX_FEATURES):
    axes[i].plot(dates1, dyn1[col], linewidth=0.5, color="#1f77b4")
    axes[i].set_title(col, fontsize=9)
    axes[i].yaxis.grid(True, alpha=0.3)
    axes[i].set_axisbelow(True)

axes[-1].set_visible(False)
for ax in axes[-4:]:
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.suptitle("Technical indicators over time (Case 1)", fontsize=13)
fig.tight_layout()
savefig("fig6_tech_timeseries.png")


# ── fig 8: case 2 trading mask ────────────────────────────────────────────────

print("Fig 8: Case 2 trading mask...")
feat2 = data["case_2_mask"]["feat"]
tgt2  = data["case_2_mask"]["target"]
dates2 = pd.to_datetime(tgt2["date"])
td = pd.Series(feat2["trading_day"].values, index=dates2)

# trading day fraction per year
by_year = pd.DataFrame({
    "trading":     td.groupby(td.index.year).sum(),
    "non_trading": (td == 0).groupby(td.index.year).sum(),
})

fig, ax = plt.subplots(figsize=(10, 4))

by_year.plot(kind="bar", stacked=True, ax=ax,
             color=["#1f77b4", "#ff7f0e"], width=0.7, legend=True)
ax.set_title("Case 2 — rows per year: trading vs non-trading")
ax.set_xlabel("Year")
ax.set_ylabel("Days")
ax.tick_params(axis="x", rotation=45)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

fig.tight_layout()
savefig("fig8_case2_mask.png")


# ── fig 9: target return distribution ────────────────────────────────────────

print("Fig 9: Return distribution...")
close_series = pd.Series(tgt["Close"].values, index=dates)
returns = close_series.pct_change().dropna() * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(returns, bins=80, color="#1f77b4", alpha=0.8, edgecolor="none")
axes[0].axvline(returns.mean(),   color="black",  linestyle="--", linewidth=1.2, label=f"μ={returns.mean():.3f}%")
axes[0].axvline(returns.std(),    color="orange", linestyle=":",  linewidth=1.2, label=f"σ={returns.std():.2f}%")
axes[0].set_title("Daily return distribution (^NDX)")
axes[0].set_xlabel("% return")
axes[0].legend()
axes[0].yaxis.grid(True, alpha=0.3)
axes[0].set_axisbelow(True)

cum_ret = (1 + returns / 100).cumprod()
axes[1].plot(close_series.index[1:], cum_ret.values, linewidth=0.9, color="#2ca02c")
axes[1].set_title("Cumulative return (^NDX, base=1)")
axes[1].yaxis.grid(True, alpha=0.3)
axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[1].set_axisbelow(True)

fig.tight_layout()
savefig("fig9_return_distribution.png")



# ── fig 10: company close prices (normalised) ─────────────────────────────────

print("Fig 10: Company close prices...")
dyn1   = data["case_1_agg_news"]["dyn"]
tgt1   = data["case_1_agg_news"]["target"]
dates1 = pd.to_datetime(tgt1["date"])

close_cols = sorted([c for c in dyn1.columns if c.endswith("__Close")])
closes     = dyn1[close_cols].set_index(dates1)

# Normalise each series to 1 at its first valid value
first_vals = closes.bfill().iloc[0]
norm       = closes.div(first_vals)

HIGHLIGHT = ["AAPL__Close", "MSFT__Close", "NVDA__Close", "AMZN__Close",
             "GOOG__Close", "NFLX__Close", "AMD__Close",  "INTC__Close"]
HL_COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

ndx_close = pd.Series(tgt1["Close"].values, index=dates1).dropna()
ndx_norm  = ndx_close / ndx_close.iloc[0]

fig, ax = plt.subplots(figsize=(14, 6))

for col in close_cols:
    if col not in HIGHLIGHT:
        ax.plot(norm.index, norm[col], linewidth=0.4, alpha=0.18, color="#aaaaaa")

for col, color in zip(HIGHLIGHT, HL_COLORS):
    if col in norm.columns:
        ticker = col.split("__")[0]
        ax.plot(norm.index, norm[col], linewidth=1.2, alpha=0.85,
                color=color, label=ticker)

ax.plot(ndx_norm.index, ndx_norm.values, linewidth=2.0, alpha=0.95,
        color="black", linestyle="--", label="^NDX (index)")

ax.set_title("Company & index close prices — normalised to 1 at 2007 start (grey = all 74 companies, coloured = selected, black = ^NDX)", fontsize=10)
ax.set_ylabel("Price / price at start")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.7)

fig.tight_layout()
savefig("fig10_company_closes.png")


print(f"\nAll figures saved to: {OUT_DIR}/")
