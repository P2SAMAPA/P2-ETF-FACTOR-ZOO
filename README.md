# Factor Zoo Compression Engine

Addresses the "factor zoo" problem by compressing hundreds of candidate factors (lagged returns, macro, volatility, RSI, cross‑sectional moments) into a minimal non‑redundant set using Double Lasso (Belloni et al.) or Probabilistic PCA. Then predicts next‑day ETF returns.

- **Factors:** >200 per universe
- **Compression:** Double Lasso (default) or PPCA
- **Training:** Rolling 252‑day window
- **Output:** top 3 ETFs per universe by predicted return

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
