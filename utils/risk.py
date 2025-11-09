# utils/risk.py
import pandas as pd
import numpy as np

def compute_covariance_matrix(returns: pd.DataFrame):
    """
    Compute covariance matrix of asset returns.
    """
    return returns.cov()

def compute_correlation_matrix(returns: pd.DataFrame):
    """
    Compute correlation matrix between assets.
    """
    return returns.corr()

def marginal_contribution_to_risk(weights: np.ndarray, cov_matrix: np.ndarray):
    """
    Compute Marginal Contribution to Total Risk (MCTR)
    MCTR_i = w_i * (Σ * w)_i / (w.T * Σ * w)
    """
    portfolio_var = weights.T @ cov_matrix @ weights
    portfolio_vol = np.sqrt(portfolio_var)
    mctr = (weights * (cov_matrix @ weights)) / portfolio_var
    return mctr, portfolio_vol

def risk_contribution(weights: np.ndarray, cov_matrix: np.ndarray, tickers=None):
    """
    Compute percentage risk contributions.
    """
    mctr, vol = marginal_contribution_to_risk(weights, cov_matrix)
    pct_contrib = mctr / mctr.sum()
    if tickers is not None:
        return pd.DataFrame({
            "Asset": tickers,
            "Weight": weights,
            "Risk Contribution": pct_contrib
        })
    else:
        return pct_contrib
