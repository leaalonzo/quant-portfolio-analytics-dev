# utils/factors.py

import pandas as pd
import numpy as np

# =========================
# Factor Construction
# =========================

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily percent change returns per asset."""
    df = df.sort_values(["ticker", "date"])
    df["return"] = df.groupby("ticker")["close"].pct_change()
    return df


def momentum_factor(df: pd.DataFrame, long_window=252, short_window=21) -> pd.DataFrame:
    """Momentum = past 12-month return minus last 1-month return."""
    df = df.sort_values(["ticker", "date"])
    df["momentum"] = (
        df.groupby("ticker")["close"].transform(lambda x: x.pct_change(long_window))
        - df.groupby("ticker")["close"].transform(lambda x: x.pct_change(short_window))
    )
    return df


def volatility_factor(df: pd.DataFrame, window=30) -> pd.DataFrame:
    """Volatility = rolling std of daily returns."""
    df = df.sort_values(["ticker", "date"])
    if "return" not in df.columns:
        df = compute_returns(df)
    df["volatility"] = df.groupby("ticker")["return"].transform(lambda x: x.rolling(window).std())
    return df


def value_factor(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for value factor (only for equities)."""
    # If P/E ratio available, use it; otherwise simulate with inverse price
    df["value"] = 1 / df["close"]
    return df


def quality_factor(df: pd.DataFrame) -> pd.DataFrame:
    """Quality = synthetic proxy using smoothed returns."""
    df = df.sort_values(["ticker", "date"])
    df["quality"] = df.groupby("ticker")["return"].transform(lambda x: x.rolling(60).mean())
    return df


def standardize_factors(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Winsorize and z-score factors."""
    for col in cols:
        df[col] = df[col].clip(df[col].quantile(0.05), df[col].quantile(0.95))
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


def combine_factors(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Combine multiple standardized factors into a multi-factor score."""
    df["multi_factor_score"] = df[cols].mean(axis=1)
    return df
