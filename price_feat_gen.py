import pandas as pd


def generate_features(df: pd.DataFrame, volatility_window: int = 5) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values(by="Date", ascending=True).reset_index(drop=True)
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data["High"]  = pd.to_numeric(data["High"],  errors="coerce")
    data["Low"]   = pd.to_numeric(data["Low"],   errors="coerce")
    data["Open"]  = pd.to_numeric(data["Open"],  errors="coerce")

    # --- target ---
    data["Movement"] = ((data["Open"].shift(-1) - data["Close"]) / data["Close"])
    data["Movement"] = data["Movement"].shift(1)
    data["Movement"] = (data["Movement"] > 0).astype(int)

    # --- basic ---
    data["Daily_Return"] = data["Close"].pct_change() * 100
    data["Volatility"]   = data["Daily_Return"].rolling(window=volatility_window).std()

    # --- lag features ---
    data["Close_lag1"]        = data["Close"].shift(1)
    data["High_lag1"]         = data["High"].shift(1)
    data["Volume_lag1"]       = data["Volume"].shift(1)
    data["Daily_Return_lag1"] = data["Daily_Return"].shift(1)
    data["Volatility_lag1"]   = data["Volatility"].shift(1)

    # --- EMA ---
    data["EMA_12_lag1"] = data["Close"].ewm(span=12, adjust=False).mean().shift(1)
    data["EMA_26_lag1"] = data["Close"].ewm(span=26, adjust=False).mean().shift(1)

    # --- MACD ---
    data["MACD_lag1"] = data["EMA_12_lag1"] - data["EMA_26_lag1"]

    # --- RSI (14) ---
    delta = data["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    data["RSI_lag1"] = (100 - (100 / (1 + gain / loss))).shift(1)

    # --- Stochastic %K and %D (14) ---
    low14  = data["Low"].rolling(14).min()
    high14 = data["High"].rolling(14).max()
    data["Stoch_K_lag1"] = (100 * (data["Close"] - low14) / (high14 - low14)).shift(1)
    data["Stoch_D_lag1"] = data["Stoch_K_lag1"].rolling(3).mean()

    # --- Williams %R (14) ---
    data["Williams_R_lag1"] = (-100 * (high14 - data["Close"]) / (high14 - low14)).shift(1)

    # --- ROC (10) ---
    data["ROC_lag1"] = data["Close"].pct_change(periods=10).shift(1) * 100

    data = data.dropna()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date")

    return data


if __name__ == "__main__":
    df = pd.read_csv("data/NDX/ndx_full_history.csv", skiprows=[1])
    data = generate_features(df)
    print(data.columns.tolist())
    print(data.head())
