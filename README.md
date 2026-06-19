# Price Feature Generation

## What it does

`price_feat_gen.py` is a data pipeline that:
1. **Downloads** raw OHLCV price data from Yahoo Finance for each Nasdaq-100 company and the index itself
2. **Enriches** each file by computing a set of technical indicators
3. **Saves** the enriched files to a separate directory, preserving the original raw data

Raw files land in `data/`, enriched files in `data_enriched/` — same folder structure, same filenames.

```
data/
  companies/   AAPL.csv, MSFT.csv, ...   ← raw OHLCV
  index/       ^NDX.csv                  ← raw index

data_enriched/
  companies/   AAPL.csv, MSFT.csv, ...   ← + technical indicators
  index/       ^NDX.csv                  ← + technical indicators
```

All paths and directory names are configured in `config.py`.

---

## Configuration (`config.py`)

| Variable | Default | Description |
|---|---|---|
| `ORIGINAL_DATASETS_DIR` | `"data"` | Root directory for raw downloaded CSVs |
| `ENRICHED_DATASETS_DIR` | `"data_enriched"` | Root directory for enriched CSVs |
| `COMPANIES_DIR` | `"companies"` | Subdirectory name for company ticker files |
| `INDEX_DIR` | `"index"` | Subdirectory name for index files |
| `NEWS_DIR` | `"news"` | Subdirectory name for news files |
| `nasdaq_100_yahoo` | 101 tickers | List of Nasdaq-100 Yahoo Finance ticker symbols used by `get_enriched_data` |

To change where data is stored, edit `ORIGINAL_DATASETS_DIR` and `ENRICHED_DATASETS_DIR`.  
To use a different set of tickers, replace or extend `nasdaq_100_yahoo`.

---

## Running the full pipeline

```bash
python price_feat_gen.py
```

This downloads and enriches all Nasdaq-100 companies + the `^NDX` index from `2007-01-03` to today.  
To change the date range or index, edit the `__main__` block at the bottom of the file.

---

## Using individual functions

**Download + enrich everything for a custom list of tickers:**
```python
from price_feat_gen import get_enriched_data

get_enriched_data(
    target="companies",
    date_start="2015-01-01",
    date_end="2023-12-31",
    index_companies=["AAPL", "MSFT", "NVDA"]
)
```

**Download + enrich the index only:**
```python
get_enriched_data(
    target="index",
    date_start="2015-01-01",
    date_end=None,
    index="^NDX"
)
```

**Enrich an existing raw CSV:**
```python
from price_feat_gen import enrich_yfin_file

enrich_yfin_file("data/companies/AAPL.csv", "data_enriched/companies/AAPL.csv")
```

**Generate features from a DataFrame directly:**
```python
from price_feat_gen import generate_features
import pandas as pd

df = pd.read_csv("data/companies/AAPL.csv")
enriched = generate_features(df)  # DataFrame indexed by Date
```

**Download a single ticker:**
```python
from price_feat_gen import get_yfinance_ticker

get_yfinance_ticker("TSLA", "2020-01-01", None, "data/companies")
```

---

## Output columns

Each enriched CSV contains the original OHLCV columns plus:

| Column | Description |
|---|---|
| `Movement` | Target label: -1 (down) / 0 (flat) / 1 (up) |
| `Daily_Return` | % price change from previous close |
| `Volatility` | Rolling 5-day std of daily returns |
| `EMA_12`, `EMA_26` | Exponential moving averages (short/long term) |
| `MACD` | EMA_12 − EMA_26 (trend crossover signal) |
| `RSI` | Relative Strength Index 0–100 |
| `Stoch_K`, `Stoch_D` | Stochastic oscillator %K and smoothed %D |
| `Williams_R` | Williams %R, range -100 to 0 |
| `ROC` | % price change over 10 days |

---

# Math Reference

Each feature at time `t` refers to data available at `t` (no manual lag applied).

---

## Target

### Movement
The label we are trying to predict. Measures whether the market opened higher or lower than it closed the previous day — a proxy for overnight sentiment and gap direction.
```
Movement(t) = sign( [Open(t) - Close(t-1)] / Close(t-1) )
```
Returns `-1` (down), `0` (flat), or `1` (up).  
**Edge case:** the first row has no previous close (`np.roll` wraps around), so its magnitude is set to `NaN` and treated as `0` (flat).

---

## Price Features

### Daily Return
Percentage gain or loss from one closing price to the next. Captures the raw day-to-day price movement of the asset.
```
DailyReturn(t) = [Close(t) - Close(t-1)] / Close(t-1) × 100
```

### Volatility
Measures how much the daily returns fluctuate over a rolling window. High volatility means the price is moving a lot; low volatility means it is stable.
```
Volatility(t) = std( DailyReturn(t-w+1), ..., DailyReturn(t) )
```
where `w` = rolling window (default 5 days).

---

## Trend Indicators

### EMA (Exponential Moving Average)
A smoothed average that gives more weight to recent prices than older ones. Used to detect the direction of the trend — rising EMA means uptrend, falling means downtrend.
```
EMA(t) = α · Close(t) + (1 - α) · EMA(t-1)

α = 2 / (span + 1)
```
Computed for span = 12 (short-term) and span = 26 (long-term).

### MACD
Measures the gap between the short-term and long-term EMA. When the short-term EMA crosses above the long-term, it signals a potential upward trend and vice versa.
```
MACD(t) = EMA_12(t) - EMA_26(t)
```
Positive → short-term trend above long-term (bullish).  
Negative → short-term trend below long-term (bearish).

---

## Momentum Indicators

### RSI (Relative Strength Index)
Measures the speed and size of recent price changes to identify overbought or oversold conditions. Useful for detecting when a price has moved too far too fast and a reversal may be coming.
```
AvgGain(t) = mean( max(ΔClose(i), 0) )   for i = t-w+1 ... t
AvgLoss(t) = mean( max(-ΔClose(i), 0) )  for i = t-w+1 ... t

RS(t) = AvgGain(t) / AvgLoss(t)         (∞ if AvgLoss = 0)
RSI(t) = 100 - 100 / (1 + RS(t))
```
Range: 0–100. >70 overbought, <30 oversold.  
**Edge case:** when `AvgLoss = 0` (all gains in the window), division is avoided by setting `RS = ∞` directly via masking → `RSI = 100` (fully overbought).

### ROC (Rate of Change)
Expresses how much the price has changed over a fixed number of days as a percentage. A momentum indicator — a rising ROC means accelerating price growth, falling ROC means deceleration or reversal.
```
ROC(t) = [Close(t) - Close(t-p)] / Close(t-p) × 100
```
where `p` = lookback period (default 10 days).  
Positive → upward momentum. Negative → downward momentum.

---

## Oscillators

### Stochastic %K and %D
Compares the current closing price to its price range over a lookback window. Tells you where the price sits relative to its recent high and low — useful for spotting reversals at extremes. %D is a smoothed version of %K used to confirm signals.
```
L(t) = min( Low(t-w+1), ..., Low(t) )
H(t) = max( High(t-w+1), ..., High(t) )

%K(t) = 100 × [Close(t) - L(t)] / [H(t) - L(t)]
%D(t) = mean( %K(t-s+1), ..., %K(t) )
```
where `w` = 14 days, `s` = 3 days.  
>80 overbought, <20 oversold.  
**Edge case:** when `H(t) = L(t)` (flat price day, zero range), the denominator is `0` → result is set to `NaN` to avoid division by zero.

### Williams %R
Similar to Stochastic %K but inverted — measures how close the current price is to the highest high of the lookback window. Useful for identifying extreme buying or selling pressure.
```
%R(t) = -100 × [H(t) - Close(t)] / [H(t) - L(t)]
```
Range: -100 to 0. >-20 overbought, <-80 oversold.  
Inverted version of Stochastic %K.  
**Edge case:** same as Stochastic — when `H(t) = L(t)`, result is set to `NaN`.
