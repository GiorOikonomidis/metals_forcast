"""
Exploratory analysis of the NDX full history CSV.
Saves all plots to the same folder as this script.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent

from config import ORIGINAL_DATASETS_DIR, INDEX_DIR

PARENT = Path(__file__).resolve().parent.parent
CSV = str(PARENT / ORIGINAL_DATASETS_DIR / INDEX_DIR / "^NDX.csv")

# -- Load ----------------------------------------------------------------------─
df = pd.read_csv(CSV, skiprows=1, header=0)
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
df["Date"]   = pd.to_datetime(df["Date"])
df["Close"]  = pd.to_numeric(df["Close"],  errors="coerce")
df["Open"]   = pd.to_numeric(df["Open"],   errors="coerce")
df["High"]   = pd.to_numeric(df["High"],   errors="coerce")
df["Low"]    = pd.to_numeric(df["Low"],    errors="coerce")
df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

print("=" * 60)
print("NDX FULL HISTORY — DATA REPORT")
print("=" * 60)

# -- 1. Basic info --------------------------------------------------------------
print(f"\n-- Basic Info --")
print(f"Date range    : {df['Date'].min().date()}  →  {df['Date'].max().date()}")
print(f"Total rows    : {len(df):,}")
print(f"Missing Close : {df['Close'].isna().sum()}")
print(f"Missing Volume: {df['Volume'].isna().sum()}")
print(f"Close range   : {df['Close'].min():.2f}  →  {df['Close'].max():.2f}")

# -- 2. Weekends check --------------------------------------------------------─
df["DayOfWeek"] = df["Date"].dt.day_name()
weekend_rows = df[df["DayOfWeek"].isin(["Saturday", "Sunday"])]
print(f"\n-- Weekend Rows --")
print(f"Rows on Saturday/Sunday: {len(weekend_rows)}")
if len(weekend_rows) > 0:
    print(weekend_rows[["Date", "DayOfWeek", "Close"]].head(10).to_string(index=False))
else:
    print("None — weekends are correctly excluded.")

# -- 3. Day of week distribution ----------------------------------------------─
print(f"\n-- Rows per Day of Week --")
print(df["DayOfWeek"].value_counts().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
).to_string())

# -- 4. Gaps — missing trading days --------------------------------------------
df["PrevDate"] = df["Date"].shift(1)
df["Gap"]      = (df["Date"] - df["PrevDate"]).dt.days
gaps = df[df["Gap"] > 3].dropna(subset=["PrevDate"])  # >3 days = more than a weekend
print(f"\n-- Gaps > 3 calendar days (likely holidays or data issues) --")
print(f"Count: {len(gaps)}")
print(gaps[["PrevDate", "Date", "Gap"]].head(20).to_string(index=False))

# -- 5. Duplicate dates --------------------------------------------------------
dupes = df[df["Date"].duplicated(keep=False)]
print(f"\n-- Duplicate Dates --")
print(f"Count: {len(dupes)}")
if len(dupes) > 0:
    print(dupes[["Date", "Close"]].to_string(index=False))

# -- 6. Zero or negative prices ------------------------------------------------
bad_prices = df[df["Close"] <= 0]
print(f"\n-- Zero or Negative Close Prices --")
print(f"Count: {len(bad_prices)}")

# -- 7. Daily returns stats ----------------------------------------------------
df["Daily_Return"] = df["Close"].pct_change() * 100
print(f"\n-- Daily Return Stats (%) --")
print(df["Daily_Return"].describe().round(4).to_string())
print(f"\nTop 5 best days:")
print(df.nlargest(5, "Daily_Return")[["Date", "Close", "Daily_Return"]].to_string(index=False))
print(f"\nTop 5 worst days:")
print(df.nsmallest(5, "Daily_Return")[["Date", "Close", "Daily_Return"]].to_string(index=False))

# -- 8. Volume stats ----------------------------------------------------------─
print(f"\n-- Volume Stats --")
print(df["Volume"].describe().round(0).to_string())
zero_vol = df[df["Volume"] == 0]
print(f"Zero-volume days: {len(zero_vol)}")
if len(zero_vol) > 0:
    print(zero_vol[["Date", "Close", "Volume"]].head(10).to_string(index=False))

print("\n" + "=" * 60)
print("Saving plots …")

# -- PLOT 1: Full price history ------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["Date"], df["Close"], linewidth=0.7, color="steelblue")
ax.set_title("NASDAQ-100 (NDX) — Full Close Price History")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_1_full_history.png", dpi=150)
plt.close()

# -- PLOT 2: Daily returns distribution --------------------------------------─
fig, ax = plt.subplots(figsize=(10, 4))
returns = df["Daily_Return"].dropna()
ax.hist(returns, bins=150, color="steelblue", edgecolor="none")
ax.axvline(0, color="red", linewidth=1, linestyle="--")
ax.set_title("Distribution of Daily Returns (%)")
ax.set_xlabel("Daily Return (%)")
ax.set_ylabel("Frequency")
ax.set_xlim(returns.quantile(0.001), returns.quantile(0.999))
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_2_return_distribution.png", dpi=150)
plt.close()

# -- PLOT 3: Rolling 30-day volatility over time ------------------------------─
df["Volatility_30d"] = df["Daily_Return"].rolling(30).std()
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df["Date"], df["Volatility_30d"], linewidth=0.7, color="darkorange")
ax.set_title("30-Day Rolling Volatility of Daily Returns")
ax.set_xlabel("Date")
ax.set_ylabel("Std Dev of Daily Return (%)")
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_3_rolling_volatility.png", dpi=150)
plt.close()

# -- PLOT 4: Gap sizes over time ----------------------------------------------─
fig, ax = plt.subplots(figsize=(14, 3))
ax.scatter(gaps["Date"], gaps["Gap"], s=10, color="crimson", alpha=0.7)
ax.set_title("Gaps Between Consecutive Trading Days (> 3 calendar days)")
ax.set_xlabel("Date")
ax.set_ylabel("Gap (days)")
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_4_gaps.png", dpi=150)
plt.close()

# -- PLOT 5: Average return by day of week ------------------------------------─
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
avg_by_day = df.groupby("DayOfWeek")["Daily_Return"].mean().reindex(order)
fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#d9534f" if v < 0 else "#5cb85c" for v in avg_by_day]
ax.bar(avg_by_day.index, avg_by_day.values, color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Average Daily Return by Day of Week (%)")
ax.set_ylabel("Avg Return (%)")
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_5_return_by_weekday.png", dpi=150)
plt.close()

print("Done. Plots saved to:", OUT_DIR)
