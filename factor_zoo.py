import pandas as pd
import numpy as np

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def build_factor_zoo(returns_df, macro_df, lag_days=[1,5,21], tech_windows=[5,10,20,50]):
    """
    Build a DataFrame of candidate factors (columns) from ETF returns and macro.
    Returns a factor matrix (days x factors) and list of factor names.
    """
    factors = []
    factor_names = []
    # 1. Lagged returns of each ETF as factors
    for lag in lag_days:
        lagged = returns_df.shift(lag).fillna(0)
        for col in lagged.columns:
            factors.append(lagged[col])
            factor_names.append(f"{col}_lag{lag}")
    # 2. Macro factors (levels)
    for col in macro_df.columns:
        factors.append(macro_df[col])
        factor_names.append(f"macro_{col}")
    # 3. Rolling volatility (annualised) for each ETF
    for window in tech_windows:
        vol = returns_df.rolling(window).std() * np.sqrt(252)
        for col in vol.columns:
            factors.append(vol[col])
            factor_names.append(f"{col}_vol{window}")
    # 4. RSI for each ETF (window=14)
    for col in returns_df.columns:
        rsi = compute_rsi(returns_df[col], window=14)
        factors.append(rsi)
        factor_names.append(f"{col}_rsi14")
    # 5. Cross-sectional factors: median return, dispersion
    # Add cross-sectional mean return
    cs_mean = returns_df.mean(axis=1)
    factors.append(cs_mean)
    factor_names.append("cs_mean_return")
    # Add cross-sectional standard deviation
    cs_std = returns_df.std(axis=1)
    factors.append(cs_std)
    factor_names.append("cs_std_return")
    # Combine into DataFrame
    factor_df = pd.concat(factors, axis=1)
    factor_df.columns = factor_names
    # Fill any remaining NaN with 0
    factor_df = factor_df.fillna(0)
    return factor_df, factor_names
