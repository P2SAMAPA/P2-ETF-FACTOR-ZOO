import pandas as pd
import numpy as np

def compute_rsi(series, window=14):
    """Relative Strength Index."""
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
    Build a large set of candidate factors from ETF returns and macro data.
    Returns:
        factor_df: DataFrame with factors as columns, aligned with returns_df index.
        factor_names: list of column names.
    """
    factors = []
    factor_names = []
    # 1. Lagged returns of each ETF
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
    # 5. Cross-sectional factors: median return, dispersion (standard deviation)
    cs_mean = returns_df.mean(axis=1)
    factors.append(cs_mean)
    factor_names.append("cs_mean_return")
    cs_std = returns_df.std(axis=1)
    factors.append(cs_std)
    factor_names.append("cs_std_return")
    # Combine into a single DataFrame
    factor_df = pd.concat(factors, axis=1)
    factor_df.columns = factor_names
    # Fill any remaining NaN with 0
    factor_df = factor_df.fillna(0)
    return factor_df, factor_names
