import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Factor Zoo Compression", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
    .etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white; border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .etf-ticker { font-size: 1.3rem; font-weight: bold; }
    .etf-score { font-size: 0.9rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📚 Factor Zoo Compression Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Double Lasso / PPCA | Compresses 200+ factors to minimal non‑redundant set | Next‑day return prediction | Multi‑window (252/504/1008/2016d)</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 📚 Factor Zoo")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Compression method:** {config.COMPRESSION_METHOD}")
if config.COMPRESSION_METHOD == "double_lasso":
    st.sidebar.markdown(f"**Alpha1:** {config.FIRST_LASSO_ALPHA}, **Alpha2:** {config.SECOND_LASSO_ALPHA}")
else:
    st.sidebar.markdown(f"**PPCA components:** {config.PPCA_COMPONENTS}")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'factor_zoo_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("🏆 Top ETFs by Compressed‑Factor Predicted Return (Best Window)")

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            ticker = etf.get('ticker', '?')
            pred = etf.get('pred_return', 0.0)
            best_win = etf.get('best_window', 'N/A')
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{ticker}</div>
                <div class="etf-score">pred return = {pred:.4f}</div>
                <div class="etf-score">best window = {best_win}d</div>
            </div>
            """, unsafe_allow_html=True)
    # Show window summary (optional)
    win_res = uni_data.get("window_results", {})
    if win_res:
        with st.expander("📊 Window summary (number of ETFs with predictions)"):
            win_summary = {win: len(scores) for win, scores in win_res.items()}
            st.write(win_summary)
    with st.expander("📋 Full ranking (all ETFs, best window per ETF)"):
        full = uni_data.get("full_scores", {})
        if full:
            # Convert dict to DataFrame safely
            rows = []
            for ticker, info in full.items():
                if isinstance(info, dict):
                    score = info.get('score', 0.0)
                    win = info.get('best_window', 'N/A')
                else:
                    score = info
                    win = 'N/A'
                rows.append({"ETF": ticker, "Best Predicted Return": score, "Best Window": win})
            df = pd.DataFrame(rows).sort_values("Best Predicted Return", ascending=False)
            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

st.caption("The engine compresses a factor zoo (>200 factors) using Double Lasso or PPCA across multiple rolling windows (252, 504, 1008, 2016 days). For each ETF, we select the window that gives the highest predicted return. Higher predicted return → stronger long signal.")
