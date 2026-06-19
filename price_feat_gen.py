import os
import numpy as np
import pandas as pd
import yfinance as yf
from numpy.typing import NDArray

from config import *

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
    movement_magnitude[0] = np.nan
    movement_direction: NDArray[np.int64] = np.sign(np.nan_to_num(movement_magnitude, nan=0)).astype(np.int64)
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
    rs: NDArray[np.float64]   = np.full_like(gain, np.inf)
    mask = loss != 0
    rs[mask] = gain[mask] / loss[mask]
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
    denom: NDArray[np.float64] = high_max - low_min
    stoch_k: NDArray[np.float64] = np.where(denom == 0, np.nan, 100 * (close_price - low_min) / np.where(denom == 0, 1, denom))
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
    denom: NDArray[np.float64] = high_max - low_min
    williams_r: NDArray[np.float64] = np.where(denom == 0, np.nan, -100 * (high_max - close_price) / np.where(denom == 0, 1, denom))
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

# handle missing values an kai auto nomizw to exei kanei h evgenia

def generate_features(df: pd.DataFrame, volatility_window: int = 5) -> pd.DataFrame:
    """
    Computes all technical indicators from raw OHLCV data and appends them as new columns.
    Drops rows with NaN (introduced by rolling windows) and sets Date as the index.

    Args:
        df:                raw DataFrame with columns: Date, Open, High, Low, Close, Volume
        volatility_window: rolling window size in days for volatility calculation
    Returns:
        enriched DataFrame indexed by Date with all feature columns added
    """
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

def get_yfinance_ticker(ticker: str, date_start: str, date_end: str, file_path: str):
    """
    Downloads OHLCV data for a single ticker from Yahoo Finance and saves it as a CSV.

    Args:
        ticker:     Yahoo Finance ticker symbol (e.g. "AAPL")
        date_start: start date string, format "YYYY-MM-DD"
        date_end:   end date string, format "YYYY-MM-DD" (None = today)
        file_path:  directory where the CSV will be saved
    """
    df = yf.download(ticker, start=date_start, end=date_end, auto_adjust=True, progress=False)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df = df.reset_index()
    path = f"{file_path}/{ticker}.csv"
    df.to_csv(path, index=False)
    print(f"Saved {ticker} → {path}")


def get_stocks_of_index(dir: str, index_companies: NDArray[np.str_], date_start: str , date_end: str = None):
    """
    Downloads OHLCV data for a list of tickers and saves one CSV per ticker.

    Args:
        dir:              directory where CSVs will be saved
        index_companies:  array of ticker symbols to download
        date_start:       start date string, format "YYYY-MM-DD"
        date_end:         end date string, format "YYYY-MM-DD" (None = today)
    """
    for ticker in index_companies:
        get_yfinance_ticker(ticker , date_start , date_end , dir)
        


def enrich_yfin_file(orig_file: str , dest_file: str):
    """
        INPUTS :
            orgi_file : the orginal filepath that you want to enrich
            dest_file : the fileapth that you want to save the enriched file
    """
    df = pd.read_csv(orig_file)
    enriched = generate_features(df)
    enriched.to_csv(dest_file, index=True)


def get_enriched_data(target: str, date_start: str, date_end: str, index_companies: NDArray[np.str_] = None, index: str = None):
    """
    Downloads raw data, enriches it with technical indicators, and saves the results.
    Handles both company CSVs (one per ticker) and a single index CSV.

    Args:
        target:           directory name — either COMPANIES_DIR or INDEX_DIR
        date_start:       start date string, format "YYYY-MM-DD"
        date_end:         end date string, format "YYYY-MM-DD" (None = today)
        index_companies:  array of ticker symbols (required when target == COMPANIES_DIR)
        index:            index ticker symbol (required when target == INDEX_DIR)
    """

    origi_dir = os.path.join(ORIGINAL_DATASETS_DIR,target)
    os.makedirs(origi_dir , exist_ok=True)

    if target == COMPANIES_DIR : get_stocks_of_index(origi_dir, index_companies, date_start, date_end)
    elif target == INDEX_DIR : 
        get_yfinance_ticker(index, date_start, date_end ,origi_dir)

    enrch_path = os.path.join(ENRICHED_DATASETS_DIR,target)
    os.makedirs(enrch_path , exist_ok=True)

    for file in os.listdir(origi_dir): 
        print(file)
        enrich_yfin_file(os.path.join(origi_dir, file),os.path.join(enrch_path, file))

        





if __name__ == "__main__":
    date_start = "2007-01-03"
    date_end = None
    index = "NDX"
    get_enriched_data(target=COMPANIES_DIR, date_start=date_start, date_end=date_end, index_companies=nasdaq_100_yahoo)
    get_enriched_data(target=INDEX_DIR, date_start=date_start, date_end=date_end, index=index)
    
