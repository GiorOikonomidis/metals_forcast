import pandas as pd


def generate_features(df: pd.DataFrame, volatility_window: int = 5) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values(by="Date", ascending=True).reset_index(drop=True)

    data["Movement"] = ((data["Open"].shift(-1) - data["Close"]) / data["Close"])
    data["Movement"] = data["Movement"].shift(1)
    data["Movement"] = (data["Movement"] > 0).astype(int)

    data["Daily_Return"] = data["Close"].pct_change() * 100
    data["Volatility"] = data["Daily_Return"].rolling(window=volatility_window).std()

    data["Close_lag1"] = data["Close"].shift(1)
    data["High_lag1"] = data["High"].shift(1)
    data["Volume_lag1"] = data["Volume"].shift(1)
    data["Daily_Return_lag1"] = data["Daily_Return"].shift(1)
    data["Volatility_lag1"] = data["Volatility"].shift(1)

    data = data.dropna()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date")

    return data


if __name__ == "__main__":
    df = pd.read_csv("data/NDX/ndx_full_history.csv", skiprows=[1])
    data = generate_features(df)
    print(data.columns.tolist())
    print(data.head())
