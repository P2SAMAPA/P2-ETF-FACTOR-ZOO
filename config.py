import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-factor-zoo-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Factor construction parameters
LAG_DAYS = [1, 5, 21]                     # lagged returns as factors
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]
TECHNICAL_WINDOWS = [5, 10, 20, 50]       # for RSI, volatility, etc.

# Compression method: 'double_lasso' or 'ppca'
COMPRESSION_METHOD = "double_lasso"

# Double Lasso parameters
FIRST_LASSO_ALPHA = 0.01
SECOND_LASSO_ALPHA = 0.005

# PPCA parameters
PPCA_COMPONENTS = 20

# Rolling window for training (days)
TRAIN_WINDOW = 252

TOP_N = 3
