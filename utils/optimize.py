# utils/optimize.py
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

def optimize_portfolio(returns: pd.DataFrame, constraint_crypto=0.2, method='max_sharpe'):
    """
    Optimize portfolio using mean-variance optimization with robust handling.
    """
    # Ensure no infinities or NaNs
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Clip extreme values
    returns = returns.clip(lower=-0.5, upper=1.0)
    
    # Remove any remaining problematic rows
    returns = returns[np.isfinite(returns).all(axis=1)]
    
    if len(returns) < 252:
        raise ValueError(f"Insufficient data after cleaning: only {len(returns)} days")
    
    # Calculate expected returns
    mu = expected_returns.mean_historical_return(returns)
    
    # Use sample covariance with strong regularization
    S_raw = returns.cov() * 252
    
    # Add diagonal regularization
    regularization = 0.1  # 10% regularization for max_sharpe stability
    S = S_raw + np.eye(len(S_raw)) * regularization
    
    # Ensure symmetry
    S = (S + S.T) / 2
    
    # Convert back to DataFrame
    S = pd.DataFrame(S, index=returns.columns, columns=returns.columns)
    
    # Try optimization with different solvers
    solvers_to_try = ['ECOS', 'SCS', 'OSQP']
    
    for solver in solvers_to_try:
        try:
            # Create fresh optimizer for each attempt
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.5), solver=solver)
            
            if method == 'max_sharpe':
                ef.max_sharpe(risk_free_rate=0.02)
            elif method == 'min_volatility':
                ef.min_volatility()
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # If we got here, optimization succeeded
            weights = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=True)
            return weights, perf, mu, S
            
        except Exception as e:
            print(f"{solver} solver failed: {e}")
            continue
    
    # All solvers failed - use equal weights
    print("All solvers failed, using equal weights")
    weights = {ticker: 1.0/len(returns.columns) for ticker in returns.columns}
    
    # Calculate performance manually
    weights_series = pd.Series(weights)
    portfolio_returns = returns @ weights_series
    portfolio_return = portfolio_returns.mean() * 252
    portfolio_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
    
    return weights, (portfolio_return, portfolio_vol, sharpe), mu, S

def efficient_frontier_curve(returns: pd.DataFrame, points=50):
    """
    Compute points along the efficient frontier.
    """
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    returns = returns.clip(lower=-0.5, upper=1.0)
    returns = returns[np.isfinite(returns).all(axis=1)]
    
    mu = expected_returns.mean_historical_return(returns)
    
    S_raw = returns.cov() * 252
    S = S_raw + np.eye(len(S_raw)) * 0.1
    S = (S + S.T) / 2
    
    frontier_y = []
    frontier_x = []
    
    for r in np.linspace(mu.min(), mu.max(), points):
        for solver in ['ECOS', 'SCS']:
            try:
                ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.5), solver=solver)
                ef.efficient_return(r)
                ret, vol, _ = ef.portfolio_performance()
                frontier_y.append(vol)
                frontier_x.append(ret)
                break
            except:
                continue
    
    return frontier_x, frontier_y