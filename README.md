# Price Feature Generation — Math Reference

Generates technical indicators from raw OHLCV price data for Nasdaq-100 companies.
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
