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
    Form long/short portfolios based on factor scores, optionally by asset_type.
    Handles cases where some asset groups (e.g. Crypto) start later.
    """
    df.columns = df.columns.str.lower()
    portfolios = []

    group_iter = df.groupby([group_col, "date"]) if group_col else df.groupby("date")

    for keys, group in group_iter:
        # Skip groups with no factor scores
        if "multi_factor_score" not in group.columns:
            continue

        # Handle missing returns (e.g. for early crypto dates)
        if "return" not in group.columns:
            group["return"] = np.nan

        group = group.dropna(subset=["multi_factor_score"])
        group["return"] = group["return"].fillna(0.0)

        if len(group) < 5:  # not enough assets to form portfolios
            continue

        cutoff = int(len(group) * quantile)
        if cutoff == 0:
            continue

        if long_short:
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
        print("⚠️ No valid portfolios formed — possibly missing data for some asset groups.")
        return pd.DataFrame()

    print(f"✅ Portfolios formed: {len(portfolios)} groups total.")
    return pd.concat(portfolios, ignore_index=True)

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
