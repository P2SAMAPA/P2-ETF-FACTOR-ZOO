import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from factor_zoo import build_factor_zoo
from compression import double_lasso_compress, ppca_compress
from sklearn.preprocessing import StandardScaler

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Factor Zoo Compression) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < config.TRAIN_WINDOW + 50:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Get macro data
        macro = df[config.MACRO_COLS].copy() if all(c in df.columns for c in config.MACRO_COLS) else pd.DataFrame()
        if macro.empty:
            print("  No macro data; using zeros")
            macro = pd.DataFrame(0, index=returns.index, columns=config.MACRO_COLS)

        # Build factor zoo for the entire period
        factor_df, factor_names = build_factor_zoo(returns, macro, config.LAG_DAYS, config.TECHNICAL_WINDOWS)
        # Align with returns index
        common_idx = returns.index.intersection(factor_df.index)
        factor_df = factor_df.loc[common_idx]
        returns = returns.loc[common_idx]

        # Prepare training data (rolling window)
        predictions = {}
        for i in range(config.TRAIN_WINDOW, len(returns)-1):
            X = factor_df.iloc[i-config.TRAIN_WINDOW:i].values
            y = returns.iloc[i-config.TRAIN_WINDOW:i].values
            # Flatten X: rows are days, columns are factors. But we need to predict next day for EACH ETF? We'll model each ETF separately.
            # Simpler: treat each ETF's return as a separate target, but that would require many models.
            # Alternative: use the factors to predict the cross‑sectional return of each ETF individually.
            # For each ETF, we need its own training data. That's a lot of models.
            # To keep it practical, we'll build a single model that predicts the return of a specific ETF? Not correct.
            # The standard approach: factors are asset‑agnostic (macro, market‑wide), so we can train one model per ETF.
            # We'll loop over ETFs and train per‑ETF models.
            # That's what we will do: For each ETF, build a dataset of (factor vectors, next‑day return of that ETF).
            # Then compress and predict.

        # We restructure: For each ETF, we have a time series of its own returns and the same factor matrix.
        # We'll train per‑ETF models.

        etf_predictions = {}
        for etf in tickers:
            if etf not in returns.columns:
                continue
            # Build X (factors) and y (ETF returns) over time
            X_all = factor_df.values
            y_all = returns[etf].values
            # Align indices
            valid = ~np.isnan(y_all)
            X_all = X_all[valid]
            y_all = y_all[valid]
            if len(X_all) < config.TRAIN_WINDOW + 20:
                continue
            # Rolling walk‑forward prediction: train on last TRAIN_WINDOW days, predict next day
            # For simplicity, train on the entire history? But we need a single prediction for tomorrow.
            # We'll take the last TRAIN_WINDOW days as training, then predict the next day.
            X_train = X_all[-config.TRAIN_WINDOW:]
            y_train = y_all[-config.TRAIN_WINDOW:]
            # Standardise features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            if config.COMPRESSION_METHOD == "double_lasso":
                coef, selected = double_lasso_compress(X_train_scaled, y_train,
                                                       alpha1=config.FIRST_LASSO_ALPHA,
                                                       alpha2=config.SECOND_LASSO_ALPHA)
                # Predict on the most recent factor vector (the day after the training window)
                # The factors for the next day (today) are the last row of factor_df
                X_last = factor_df.iloc[-1].values.reshape(1, -1)
                X_last_scaled = scaler.transform(X_last)
                pred = np.dot(X_last_scaled, coef)[0]
            else:  # ppca
                predict_func, _, _, _ = ppca_compress(X_train_scaled, y_train, n_components=config.PPCA_COMPONENTS)
                X_last = factor_df.iloc[-1].values.reshape(1, -1)
                X_last_scaled = scaler.transform(X_last)
                pred = predict_func(X_last_scaled)[0]
            etf_predictions[etf] = pred

        if not etf_predictions:
            print("  No predictions")
            all_results[universe_name] = {"top_etfs": []}
            continue

        sorted_etfs = sorted(etf_predictions.items(), key=lambda x: x[1], reverse=True)
        top_etfs = []
        full_scores = {}
        for ticker, pred in sorted_etfs[:config.TOP_N]:
            top_etfs.append({"ticker": ticker, "pred_return": float(pred)})
            full_scores[ticker] = float(pred)
        print(f"  Top 3 ETFs by predicted return: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/factor_zoo_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Factor Zoo Compression Engine complete ===")

if __name__ == "__main__":
    main()
