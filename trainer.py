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
        if returns.empty or len(returns) < min(config.WINDOWS) + 50:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        macro = df[config.MACRO_COLS].copy() if all(c in df.columns for c in config.MACRO_COLS) else pd.DataFrame()
        if macro.empty:
            print("  No macro data; using zeros")
            macro = pd.DataFrame(0, index=returns.index, columns=config.MACRO_COLS)

        factor_df, factor_names = build_factor_zoo(returns, macro, config.LAG_DAYS, config.TECHNICAL_WINDOWS)
        common_idx = returns.index.intersection(factor_df.index)
        factor_df = factor_df.loc[common_idx]
        returns = returns.loc[common_idx]

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 20:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            etf_pred = {}
            for etf in tickers:
                if etf not in returns.columns:
                    continue
                X_all = factor_df.iloc[-win:].values
                y_all = returns[etf].iloc[-win:].values
                valid = ~np.isnan(y_all)
                X_train = X_all[valid]
                y_train = y_all[valid]
                if len(X_train) < 20:
                    continue
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_train)
                if config.COMPRESSION_METHOD == "double_lasso":
                    coef, selected = double_lasso_compress(X_scaled, y_train,
                                                           alpha1=config.FIRST_LASSO_ALPHA,
                                                           alpha2=config.SECOND_LASSO_ALPHA)
                    X_last = factor_df.iloc[-1].values.reshape(1, -1)
                    X_last_scaled = scaler.transform(X_last)
                    pred = np.dot(X_last_scaled, coef)[0]
                else:
                    predict_func, _, _, _ = ppca_compress(X_scaled, y_train, n_components=config.PPCA_COMPONENTS)
                    X_last = factor_df.iloc[-1].values.reshape(1, -1)
                    X_last_scaled = scaler.transform(X_last)
                    pred = predict_func(X_last_scaled)[0]
                if np.isnan(pred) or np.isinf(pred):
                    pred = 0.0
                etf_pred[etf] = pred
            window_results[win] = etf_pred
            for etf, pred in etf_pred.items():
                if etf not in best_per_etf or pred > best_per_etf[etf][0]:
                    best_per_etf[etf] = (pred, win)

        # If all best predictions are zero (or very small), fallback to historical mean
        all_zeros = all(abs(pred) < 1e-6 for pred, _ in best_per_etf.values())
        if all_zeros:
            print("  All predictions zero, falling back to historical mean return")
            best_per_etf = {}
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (mean_ret, 0)

        if not best_per_etf:
            print("  No valid predictions")
            all_results[universe_name] = {"top_etfs": []}
            continue

        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = []
        full_scores = {}
        for ticker, (pred, win) in sorted_etfs[:config.TOP_N]:
            top_etfs.append({"ticker": ticker, "pred_return": float(pred), "best_window": win})
            full_scores[ticker] = {"score": float(pred), "best_window": win}
        print(f"  Top 3 ETFs: {[e['ticker'] for e in top_etfs]} (predictions: {[e['pred_return'] for e in top_etfs]})")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/factor_zoo_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Factor Zoo Compression Engine (multi‑window) complete ===")

if __name__ == "__main__":
    main()
