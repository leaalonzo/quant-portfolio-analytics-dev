import streamlit as st
import pandas as pd
import plotly.express as px
from utils.optimize import optimize_portfolio
from utils.risk import risk_contribution
import numpy as np

st.set_page_config(page_title="Quant Portfolio Dashboard", layout="wide")

@st.cache_data
def load_data():
    results = pd.read_csv("data/portfolio_results_all.csv", parse_dates=["date"])
    stats = pd.read_csv("data/portfolio_stats_all.csv")
    asset_returns = pd.read_csv("data/asset_daily_returns.csv", parse_dates=["date"]).set_index("date")

    # 🔧 Ensure all values are floats
    asset_returns = asset_returns.apply(pd.to_numeric, errors="coerce")

    return results, stats, asset_returns


results, stats, asset_returns = load_data()

st.sidebar.header("⚙️ Configuration")

asset_type = st.sidebar.selectbox("Asset Type", sorted(results["asset_type"].unique()))
factor = st.sidebar.selectbox("Factor", sorted(results["factor"].unique()))
mode = st.sidebar.radio("Portfolio Mode", ["long_only", "long_short"])
tab = st.sidebar.radio("View", ["Performance", "Optimization & Risk"])

subset = results[
    (results["asset_type"] == asset_type)
    & (results["factor"] == factor)
    & (results["mode"] == mode)
]

subset_stats = stats[
    (stats["asset_type"] == asset_type)
    & (stats["factor"] == factor)
    & (stats["mode"] == mode)
]

if subset.empty:
    st.warning("⚠️ No data for this configuration.")
    st.stop()

# === TAB 1: Performance ===
if tab == "Performance":
    st.title(f"📈 {asset_type} Portfolio Performance — {factor.title()} ({mode.replace('_',' ').title()})")

    col1, col2, col3, col4 = st.columns(4)
    stats_dict = subset_stats.iloc[0].to_dict() if not subset_stats.empty else {}

    col1.metric("Cumulative Return", f"{stats_dict.get('Cumulative Return', 0):.2%}")
    col2.metric("Sharpe Ratio", f"{stats_dict.get('Sharpe Ratio', 0):.2f}")
    col3.metric("Volatility", f"{stats_dict.get('Volatility', 0):.2%}")
    col4.metric("Max Drawdown", f"{stats_dict.get('Max Drawdown', 0):.2%}")

    st.subheader("Cumulative Portfolio Performance")
    fig = px.line(subset, x="date", y="cumulative", title="Cumulative Performance")
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rolling 60-Day Sharpe Ratio")
    subset["rolling_sharpe"] = subset["portfolio_return"].rolling(60).mean() / subset["portfolio_return"].rolling(60).std()
    fig2 = px.line(subset, x="date", y="rolling_sharpe", title="Rolling 60-Day Sharpe")
    fig2.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📄 Show Raw Data"):
        st.dataframe(subset.tail(20))

# === TAB 2: Optimization & Risk ===
else:
    st.title("🧠 Portfolio Optimization & Risk Attribution")
    st.markdown("Optimize weights using **PyPortfolioOpt** and decompose portfolio risk.")
    
    opt_method = st.selectbox("Optimization Method", ["max_sharpe", "min_volatility"])
    
    st.subheader("🔍 Data Preparation")
    
    # === FILTER BY ASSET TYPE (NEW CODE) ===
    # Determine which columns belong to the selected asset_type
    def get_asset_type_from_ticker(ticker):
        """Determine if a ticker is Equity or Crypto based on naming convention"""
        if '-USD' in ticker:
            return 'Crypto'
        else:
            return 'Equity'
    
    # Filter asset_returns to only include the selected asset_type
    available_tickers = [col for col in asset_returns.columns 
                        if get_asset_type_from_ticker(col) == asset_type]
    
    if not available_tickers:
        st.error(f"❌ No {asset_type} assets found in asset_daily_returns.csv")
        st.stop()
    
    st.info(f"📊 Filtering to {len(available_tickers)} {asset_type} assets")
    
    # Use only the filtered assets
    pivot = asset_returns[available_tickers].copy()
    # === END NEW CODE ===
    
    pivot = pivot.select_dtypes(include=["number"])
    
    # Calculate NaN percentages
    nan_pct = (pivot.isna().sum() / len(pivot) * 100).round(2)
    
    st.write("**Top 10 assets by data quality:**")
    st.dataframe(nan_pct.sort_values().head(10).to_frame('NaN %'))
    
    # Use 50% threshold
    threshold = 50
    pivot_clean = pivot.loc[:, nan_pct < threshold]
    
    st.write(f"\n**Assets with <{threshold}% NaNs:** {list(pivot_clean.columns)}")
    st.write(f"**Shape:** {pivot_clean.shape}")
    
    if pivot_clean.shape[1] < 2:
        st.error(f"❌ Only {pivot_clean.shape[1]} {asset_type} assets available")
        st.stop()
    
    # Check for infinities BEFORE filling NaNs
    inf_count_before = np.isinf(pivot_clean).sum().sum()
    st.write(f"**Infinity values found:** {inf_count_before}")
    
    if inf_count_before > 0:
        st.write("**Columns with infinities:**")
        inf_cols = pivot_clean.columns[np.isinf(pivot_clean).any()].tolist()
        for col in inf_cols:
            inf_count = np.isinf(pivot_clean[col]).sum()
            st.write(f"  - {col}: {inf_count} infinities")
    
    # Replace infinities with NaN, then fill
    pivot_clean = pivot_clean.replace([np.inf, -np.inf], np.nan)
    st.write(f"**After replacing infinities with NaN:** {pivot_clean.isna().sum().sum()} NaNs")
    
    # Fill NaNs
    pivot_clean = pivot_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Clip extreme values
    pivot_clean = pivot_clean.clip(lower=-0.50, upper=1.00)
    st.write(f"**After clipping to [-50%, +100%]:** Max={pivot_clean.max().max():.4f}, Min={pivot_clean.min().min():.4f}")
    
    # Drop zero variance
    pivot_clean = pivot_clean.loc[:, pivot_clean.std() > 1e-6]
    st.write(f"**After dropping zero variance:** {pivot_clean.shape}")
    
    # Final check for NaNs and infinities
    final_nans = pivot_clean.isna().sum().sum()
    final_infs = np.isinf(pivot_clean).sum().sum()
    
    st.write(f"\n**Final data quality:**")
    st.write(f"  - NaNs: {final_nans}")
    st.write(f"  - Infinities: {final_infs}")
    st.write(f"  - Min value: {pivot_clean.min().min():.4f}")
    st.write(f"  - Max value: {pivot_clean.max().max():.4f}")
    
    if final_nans > 0 or final_infs > 0:
        st.error("❌ Data still contains NaNs or infinities after cleaning")
        st.stop()
    
    if pivot_clean.shape[1] < 2:
        st.error(f"❌ Insufficient {asset_type} assets after cleaning")
        st.stop()
    
    st.success(f"✅ Ready: {pivot_clean.shape[0]} days × {pivot_clean.shape[1]} {asset_type} assets")
    st.write(f"**Assets:** {list(pivot_clean.columns)}")
    
    # Run optimization
    try:
        weights, perf, mu, S = optimize_portfolio(pivot_clean, method=opt_method)
        
        w_series = pd.Series(weights)
        
        st.subheader("Optimized Weights")
        fig = px.bar(x=w_series.index, y=w_series.values, 
                     labels={'x': 'Asset', 'y': 'Weight'},
                     title=f'{asset_type} Portfolio Allocation')
        st.plotly_chart(fig, use_container_width=True)
        
        ann_return, ann_vol, sharpe = perf
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Return", f"{ann_return:.2%}")
        col2.metric("Volatility", f"{ann_vol:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        st.subheader("Portfolio Composition")
        comp_df = pd.DataFrame({
            'Asset': w_series.index,
            'Weight': w_series.values,
            'Expected Return': mu[w_series.index].values,
        }).sort_values('Weight', ascending=False)
        
        st.dataframe(comp_df.style.format({
            'Weight': '{:.2%}',
            'Expected Return': '{:.2%}'
        }))
        
    except Exception as e:
        st.error(f"⚠️ Optimization failed: {e}")
        import traceback
        st.code(traceback.format_exc())