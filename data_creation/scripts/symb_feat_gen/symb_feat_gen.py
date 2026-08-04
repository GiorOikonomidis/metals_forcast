import os
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from constants import PRICE_COLS
from scripts.paths import KIND_COVARIATES, KIND_TARGET, enriched_dir, raw_dir

# function indepenedt per features , anotate numpy arrays m ksekathara val decl , docstring inp/out
# xvris lagg kai ara den xrhsimopoioume Open  gia ekeinh thn mera 
# prosthkh date fatures san sinocoidal encoding


def generate_date_feat(dates: pd.Series) -> pd.DataFrame:
    """
    Encodes cyclic date features as sin/cos pairs on the unit circle.

    For each cyclic period P and raw value v:
        sin_feat = sin(2π · v / P)
        cos_feat = cos(2π · v / P)

    Features:
        day of week  (P=7):   0=Mon ... 6=Sun
        month        (P=12):  1=Jan ... 12=Dec
        day of year  (P=365): 1 ... 365

    Args:
        dates: Series of dates (datetime-compatible), shape (N,)
    Returns:
        DataFrame with columns sin_dow, cos_dow, sin_month, cos_month,
        sin_doy, cos_doy, shape (N, 6)
    """
    dt = pd.to_datetime(dates)

    result = pd.DataFrame(index=dates.index)
    result["sin_dow"]   = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    result["cos_dow"]   = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    result["sin_month"] = np.sin(2 * np.pi * dt.dt.month / 12)
    result["cos_month"] = np.cos(2 * np.pi * dt.dt.month / 12)
    result["sin_doy"]   = np.sin(2 * np.pi * dt.dt.dayofyear / 365)
    result["cos_doy"]   = np.cos(2 * np.pi * dt.dt.dayofyear / 365)

    return result


# affou eipame oti ta feature gia ena entry kanoun refer gia auto to timestamp xwris lags  kai den xrhsimopoiountai sto pred
# tote movement(t) =  [ Open(t) - Close(t-1) ]  / Close(t-1) 
# na to doume os timh oxi 1 , -1 , 0
def generate_movement(close_price: NDArray[np.float64], open_price: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    movement(t) =  [ Open(t) - Close(t-1) ]  / Close(t-1)

    Args:
        close_price:  array of daily Close prices, shape (N,)
        open_price:   array of daily Open prices, shape (N,)
    Returns:
        movement: percentage change array, shape (N,)  — NaN at index 0
    """
    close_shift: NDArray[np.float64] = np.roll(close_price, 1).astype(np.float64)
    movement: NDArray[np.float64] = (open_price - close_shift) / close_shift
    movement[0] = np.nan
    return movement

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
    Drops rows with NaN (from rolling windows) and sets Date as the index.

    Enrichment always produces raw (undifferenced) levels. Differencing is a
    model-layer transform applied at load time (model_impl/helpers.apply_differencing),
    AFTER the merge step has interpolated onto the full calendar, so the diffs
    telescope back exactly to the interpolated level.

    Args:
        df:                raw DataFrame with columns: Date, Open, High, Low, Close, Volume
        volatility_window: rolling window size in days for volatility calculation
    Returns:
        enriched DataFrame indexed by Date with all feature columns added
    """
    data = df.copy()
    data = data.sort_values(by="Date", ascending=True).reset_index(drop=True)

    
    data[PRICE_COLS] = data[PRICE_COLS].astype(float)
    data[PRICE_COLS] = data[PRICE_COLS].ffill().bfill()

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

    date_feats = generate_date_feat(data["Date"])
    data = pd.concat([data, date_feats], axis=1)

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


def enrich_yfin_file(orig_file: str , dest_file: str):
    """
        INPUTS :
            orgi_file : the orginal filepath that you want to enrich
            dest_file : the fileapth that you want to save the enriched file
    """
    df = pd.read_csv(orig_file)
    enriched = generate_features(df)
    enriched.to_csv(dest_file, index=True)

# checkare an kanw handlig ta missing values
def pipe_line(base_dir: str, dataset: str, mode: int = 0) -> None:
    """
    Enrich a dataset's downloaded CSVs with technical indicators.

    Handles either role of a dataset tree — the single target CSV or the one
    CSV per covariate ticker — with the same per-file logic.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key; selects which tree is read and written.
    mode : int, optional
        ``0`` enriches the target series, ``1`` enriches the covariates.

    Returns
    -------
    None
        Writes one enriched CSV per input, under the dataset's
        ``data_enriched/<role>/`` directory, keeping the input filename.

    Raises
    ------
    ValueError
        If ``mode`` is neither 0 nor 1.
    FileNotFoundError
        If the corresponding download step has not been run.
    """
    if mode == 0:
        kind = KIND_TARGET
    elif mode == 1:
        kind = KIND_COVARIATES
    else:
        raise ValueError(f"unknown mode {mode!r} - expected 0 (target) or 1 (covariates)")

    origi_dir  = raw_dir(base_dir, dataset, kind)
    enrch_path = enriched_dir(base_dir, dataset, kind)

    if not os.path.isdir(origi_dir):
        raise FileNotFoundError(
            f"no downloaded {kind} data for dataset {dataset!r} at {origi_dir} - "
            f"run the matching download step first"
        )

    os.makedirs(enrch_path, exist_ok=True)

    # Only CSVs: the directory can also pick up stray files, and feeding one to
    # pd.read_csv fails deep inside the parser rather than here.
    files = sorted(f for f in os.listdir(origi_dir) if f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(
            f"no CSVs to enrich for dataset {dataset!r} in {origi_dir} - "
            f"run the matching download step first"
        )

    for file in files:
        print(f"enriching -> {file}")
        enrich_yfin_file(os.path.join(origi_dir, file), os.path.join(enrch_path, file))

        





    
