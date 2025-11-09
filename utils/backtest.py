# utils/backtest.py
import pandas as pd
import numpy as np

def form_portfolios(
    df: pd.DataFrame,
    quantile: float = 0.2,
    long_short: bool = True,
    group_col: str | None = None,
):
    """
    Form long/short portfolios based on factor scores.
    Returns:
        portfolios_df: Portfolio holdings with positions
        asset_daily_returns: Complete return history for ALL assets in universe
    """
    df.columns = df.columns.str.lower()
    portfolios = []
    
    # === BUILD PORTFOLIO HOLDINGS ===
    group_iter = df.groupby([group_col, "date"]) if group_col else df.groupby("date")
    
    for keys, group in group_iter:
        group = group.dropna(subset=["multi_factor_score", "return"])
        n = len(group)
        
        if n < 5:  # skip tiny groups
            continue
        
        cutoff = max(1, int(n * quantile))
        
        if long_short:
            if cutoff * 2 > n:
                continue
            long = group.nlargest(cutoff, "multi_factor_score").copy()
            short = group.nsmallest(cutoff, "multi_factor_score").copy()
            long["position"] = 1 / len(long)
            short["position"] = -1 / len(short)
            combined = pd.concat([long, short])
        else:
            long = group.nlargest(cutoff, "multi_factor_score").copy()
            long["position"] = 1 / len(long)
            combined = long
        
        combined["weighted_return"] = combined["return"] * combined["position"]
        
        if group_col:
            combined[group_col] = group[group_col].iloc[0]
        combined["date"] = group["date"].iloc[0]
        
        portfolios.append(combined)
    
    if not portfolios:
        print("⚠️ Warning: No portfolio data formed.")
        return pd.DataFrame(), pd.DataFrame()
    
    portfolios_df = pd.concat(portfolios, ignore_index=True)
    
    # === BUILD COMPLETE ASSET RETURNS (FIX IS HERE) ===
    # Instead of using portfolios_df, use the ORIGINAL df to get ALL asset returns
    asset_daily_returns = df.pivot_table(
        index="date", 
        columns="ticker", 
        values="return", 
        aggfunc="first"  # Use first instead of mean to avoid duplicates
    ).sort_index()
    
    # Only keep tickers that appeared in at least one portfolio
    # (optional - you can remove this to keep ALL tickers)
    portfolio_tickers = portfolios_df['ticker'].unique()
    asset_daily_returns = asset_daily_returns[
        [col for col in asset_daily_returns.columns if col in portfolio_tickers]
    ]
    
    return portfolios_df, asset_daily_returns


def compute_performance(portfolios: pd.DataFrame):
    """
    Compute portfolio performance metrics.
    """
    if portfolios.empty:
        print("⚠️ Warning: No portfolio data available for this subset.")
        empty_perf = pd.DataFrame(columns=["date", "portfolio_return", "cumulative"])
        empty_stats = {
            "Cumulative Return": 0.0,
            "Sharpe Ratio": 0.0,
            "Volatility": 0.0,
            "Max Drawdown": 0.0,
        }
        return empty_perf, empty_stats

    daily_returns = portfolios.groupby("date")["weighted_return"].sum()
    cumulative = (1 + daily_returns).cumprod()

    stats = {
        "Cumulative Return": cumulative.iloc[-1] - 1,
        "Sharpe Ratio": daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() > 0
        else 0,
        "Volatility": daily_returns.std() * np.sqrt(252),
        "Max Drawdown": (cumulative / cumulative.cummax() - 1).min(),
    }

    perf = pd.DataFrame({
        "date": daily_returns.index,
        "portfolio_return": daily_returns.values,
        "cumulative": cumulative.values
    })

    return perf, stats
