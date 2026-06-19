import os
import numpy as np
import pandas as pd
import yfinance as yf
from numpy.typing import NDArray


# function indepenedt per features , anotate numpy arrays m ksekathara val decl , docstring inp/out
# xvris lagg kai ara den xrhsimopoioume Open  gia ekeinh thn mera 
# prosthkh date fatures san sinocoidal encoding

# affou eipame oti ta feature gia ena entry kanoun refer gia auto to timestamp xwris lags  kai den xrhsimopoiountai sto pred
# tote movement(t) =  [ Open(t) - Close(t-1) ]  / Close(t-1)
def generate_movement(close_price: NDArray[np.float64], open_price: NDArray[np.float64]) -> NDArray[np.int64]:
    """
    movement(t) =  [ Open(t) - Close(t-1) ]  / Close(t-1)

    Args:
        close_price:      array of daily Close prices, shape (N,)
        open_price:  array of daily Open prices , shape (N,)
    Returns:
        movement_direction:   -1 , 0 , 1 array, shape (N,)
    """
    close_shift: NDArray[np.float64] = np.roll(close_price, 1)
    movement_magnitude: NDArray[np.float64] = (open_price - close_shift) / close_shift
    movement_direction: NDArray[np.int64] = np.sign(movement_magnitude).astype(np.int64)
    return movement_direction


def generate_daily_return(close_price: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    daily_return = [ close_price(t) - close_price(t-1) ] / close_price(t-1)
    Percentage change in Close price day over day.
    Args:
        close_price:  array of daily Close prices, shape (N,)
    Returns:
        daily_return: array of % returns, shape (N,)
    """
    daily_return: NDArray[np.float64] = pd.Series(close_price).pct_change().to_numpy() * 100
    return daily_return


def generate_volatility(daily_return: NDArray[np.float64], window: int = 5) -> NDArray[np.float64]:
    """
    σ_t = std( r_{t-w+1 : t} )
    Rolling standard deviation of daily returns.

    Args:
        daily_return:   array of % returns, shape (N,)
        window:         rolling window size in days
    Returns:
        volatility:     array of rolling std, shape (N,)
    """
    volatility: NDArray[np.float64] = pd.Series(daily_return).rolling(window).std().to_numpy()
    return volatility


def generate_lag1(series: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Shift series by 1 day (yesterday's value).

    Args:
        series: input array, shape (N,)
    Returns:
        lagged: array shifted by 1, shape (N,)
    """
    lagged: NDArray[np.float64] = np.roll(series, 1).astype(np.float64)
    lagged[0] = np.nan
    return lagged


def generate_ema(close_price: NDArray[np.float64], span: int) -> NDArray[np.float64]:
    """
    EMA_t = a · Close_t + (1 - a) · EMA_{t-1}
    where:
        a = 2 / (span + 1)

    Exponential Moving Average of Close prices.

    Args:
        close_price:  array of daily Close prices, shape (N,)
        span:         EMA span in days (e.g. 12 or 26)
    Returns:
        EMA values, shape (N,)
    """
    ema: NDArray[np.float64] = pd.Series(close_price).ewm(span=span, adjust=False).mean().to_numpy()
    return ema





def generate_rsi(close_price: NDArray[np.float64], window: int = 14) -> NDArray[np.float64]:
    """
    RSI_t = 100 - 100 / (1 + RS_t)
    RS_t = AvgGain_t / AvgLoss_t

    with:
        AvgGain_t = mean( max(Δp_i, 0) )   over i = t-w+1 ... t
        AvgLoss_t = mean( max(-Δp_i, 0) )  over i = t-w+1 ... t
        Δp_t      = Close_t - Close_{t-1}

    When AvgLoss = 0 (all gains), RS = inf → RSI = 100 (fully overbought).

    Relative Strength Index.

    Args:
        close_price:  array of daily Close prices, shape (N,)
        window:       RSI window in days
    Returns:
        RSI values (0-100), shape (N,)
    """
    s = pd.Series(close_price)
    delta = s.diff()
    gain: NDArray[np.float64] = delta.clip(lower=0).rolling(window).mean().to_numpy()
    loss: NDArray[np.float64] = (-delta.clip(upper=0)).rolling(window).mean().to_numpy()
    rs: NDArray[np.float64]   = np.where(loss == 0, np.inf, gain / loss)
    rsi: NDArray[np.float64]  = 100 - (100 / (1 + rs))
    return rsi


def generate_stochastic(close_price: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64],
                         window: int = 14, smooth: int = 3) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
        Stochastic Oscillator %K and %D.

        %K_t = 100 * (Close_t - L_t) / (H_t - L_t)

        L_t = min(Low_{t-w+1 : t})
        H_t = max(High_{t-w+1 : t})
        %D_t = mean(%K_{t-s+1 : t})

        where:
            w = window (lookback period)
            s = smoothing window


        Args:
            close_price:  array of daily Close prices, shape (N,)
            high:         array of daily High prices, shape (N,)
            low:          array of daily Low prices, shape (N,)
            window:       lookback window (w)
            smooth:       smoothing window for %D (s)

        Returns:
            stoch_k: %K values (0-100), shape (N,)
            stoch_d: %D values (smoothed %K), shape (N,)
    """
    low_min:  NDArray[np.float64] = pd.Series(low).rolling(window).min().to_numpy()
    high_max: NDArray[np.float64] = pd.Series(high).rolling(window).max().to_numpy()
    stoch_k:  NDArray[np.float64] = 100 * (close_price - low_min) / (high_max - low_min)
    stoch_d:  NDArray[np.float64] = pd.Series(stoch_k).rolling(smooth).mean().to_numpy()
    return stoch_k, stoch_d


def generate_williams_r(close_price: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64],
                         window: int = 14) -> NDArray[np.float64]:
    """
    
    %R_t = -100 * (H_t - Close_t) / (H_t - L_t)

    L_t = min(Low_{t-w+1 : t})
    H_t = max(High_{t-w+1 : t})

    Williams %R, Range: -100 to 0.

    Args:
        close_price:  array of daily Close prices, shape (N,)
        high:   array of daily High prices, shape (N,)
        low:    array of daily Low prices, shape (N,)
        window: lookback window in days
    Returns:
        Williams %R , shape (N,)
    """
    low_min:    NDArray[np.float64] = pd.Series(low).rolling(window).min().to_numpy()
    high_max:   NDArray[np.float64] = pd.Series(high).rolling(window).max().to_numpy()
    williams_r: NDArray[np.float64] = -100 * (high_max - close_price) / (high_max - low_min)
    return williams_r


def generate_roc(close_price: NDArray[np.float64], periods: int = 10) -> NDArray[np.float64]:
    """
    ROC_t = 100 * (Close_t - Close_{t-p}) / Close_{t-p}

    where:
        p = number of periods (lookback window)

    Rate of Change: % price change over N periods.

    Args:
        close_price:   array of daily Close prices, shape (N,)
        periods: number of days to look back
    Returns:
        ROC values , shape (N,)
    """
    roc: NDArray[np.float64] = pd.Series(close_price).pct_change(periods=periods).to_numpy() * 100

    return roc


def generate_features(df: pd.DataFrame, volatility_window: int = 5) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values(by="Date", ascending=True).reset_index(drop=True)

    data["Close"]  = pd.to_numeric(data["Close"],  errors="coerce")
    data["High"]   = pd.to_numeric(data["High"],   errors="coerce")
    data["Low"]    = pd.to_numeric(data["Low"],    errors="coerce")
    data["Open"]   = pd.to_numeric(data["Open"],   errors="coerce")
    data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce")

    close_price = data["Close"].to_numpy()
    open_price  = data["Open"].to_numpy()
    high        = data["High"].to_numpy()
    low         = data["Low"].to_numpy()
    volume      = data["Volume"].to_numpy()

    daily_return     = generate_daily_return(close_price)
    volatility       = generate_volatility(daily_return, window=volatility_window)
    ema_12           = generate_ema(close_price, span=12)
    ema_26           = generate_ema(close_price, span=26)
    stoch_k, stoch_d = generate_stochastic(close_price, high, low)

    data["Movement"]      = generate_movement(close_price, open_price)
    data["Daily_Return"]  = daily_return
    data["Volatility"]    = volatility
    data["Close"]         = close_price
    data["High"]          = high
    data["Volume"]        = volume
    data["EMA_12"]        = ema_12
    data["EMA_26"]        = ema_26
    data["MACD"]          = ema_12 - ema_26
    data["RSI"]           = generate_rsi(close_price)
    data["Stoch_K"]       = stoch_k
    data["Stoch_D"]       = stoch_d
    data["Williams_R"]    = generate_williams_r(close_price, high, low)
    data["ROC"]           = generate_roc(close_price)

    data = data.dropna()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date")

    return data


nasdaq_100_yahoo = [
    "ADBE","ADP","AMD","ABNB","ALNY","GOOGL","GOOG","AMZN","AEP","AMGN",
    "ADI","AAPL","AMAT","APP","ARM","ASML","TEAM","ADSK","AXON","BKR",
    "BKNG","AVGO","CDNS","CHTR","CTAS","CSCO","CCEP","CTSH","CMCSA","CEG",
    "CPRT","CSGP","COST","CRWD","CSX","DDOG","DXCM","FANG","DASH","EA",
    "EXC","FAST","FER","FTNT","GEHC","GILD","HON","IDXX","INSM","INTC",
    "INTU","ISRG","KDP","KLAC","KHC","LRCX","LIN","MAR","MRVL","MELI",
    "META","MCHP","MU","MSFT","MSTR","MDLZ","MPWR","MNST","NFLX","NVDA",
    "NXPI","ODFL","ORLY","PCAR","PLTR","PANW","PAYX","PYPL","PDD","PEP",
    "QCOM","REGN","ROP","ROST","STX","SHOP","SBUX","SNPS","TTWO","TSLA",
    "TXN","TRI","TMUS","VRSK","VRTX","WMT","WBD","WDC","WDAY","XEL","ZS"
]


def nasdaq_100(start: str = "2007-01-03", end: str = None):
    os.makedirs("data/companies", exist_ok=True)

    for ticker in nasdaq_100_yahoo:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        df = df.reset_index()
        path = f"data/companies/{ticker}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {ticker} → {path}")


if __name__ == "__main__":
    #nasdaq_100()
    
    df = pd.read_csv("data/NDX/ndx_full_history.csv", skiprows=[1])
    data = generate_features(df)
    print(data.columns.tolist())
    print(data.head())
    
