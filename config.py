ORIGINAL_DATASETS_DIR = "data"
ENRICHED_DATASETS_DIR = "data_enriched"

COMPANIES_DIR = "companies"
INDEX_DIR = "index"
NEWS_DIR = "news"

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

DATASETS_DIR   = "datasets"
DATE_FEAT_COLS = ["sin_dow", "cos_dow", "sin_month", "cos_month", "sin_doy", "cos_doy"]
INDEX_FEATURES = [
    "Open","High","Low","Volume",
    "EMA_12", "EMA_26", "MACD", "RSI", "Stoch_K", "Stoch_D",
    "Williams_R", "ROC", "Daily_Return", "Volatility", "Movement",
]