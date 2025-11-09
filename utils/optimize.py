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
    
    # Check if mu has NaNs or infinities
    if mu.isna().any() or np.isinf(mu).any():
        print("⚠️ Expected returns contain NaN or infinity, replacing with 0")
        mu = mu.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Use sample covariance with strong regularization
    S_raw = returns.cov() * 252
    
    # Add diagonal regularization
    regularization = 0.1  # 10% regularization for max_sharpe stability
    S = S_raw + np.eye(len(S_raw)) * regularization
    
    # Ensure symmetry
    S = (S + S.T) / 2
    
    # Check for NaN/inf in covariance matrix
    if np.isnan(S.values).any() or np.isinf(S.values).any():
        print("⚠️ Covariance matrix has NaN/inf, using diagonal matrix")
        variances = returns.var() * 252
        variances = variances.fillna(0.01).clip(lower=0.001)  # Minimum variance
        S = pd.DataFrame(np.diag(variances), index=returns.columns, columns=returns.columns)
    
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
            
            # Check if weights are valid
            weights_series = pd.Series(weights)
            if weights_series.sum() == 0 or weights_series.isna().any():
                print(f"⚠️ {solver} returned invalid weights, trying next solver")
                continue
            
            try:
                perf = ef.portfolio_performance(verbose=False)
                # Verify performance metrics are valid
                if any(np.isnan(p) or np.isinf(p) for p in perf):
                    print(f"⚠️ {solver} returned invalid performance metrics")
                    continue
                
                print(f"✅ Optimization succeeded with {solver} solver")
                return weights, perf, mu, S
            except Exception as perf_error:
                print(f"⚠️ {solver} portfolio_performance failed: {perf_error}")
                continue
            
        except Exception as e:
            print(f"❌ {solver} solver failed: {str(e)[:100]}")
            continue
    
    # All solvers failed - use equal weights with robust calculation
    print("⚠️ All solvers failed, using equal weights fallback")
    weights = {ticker: 1.0/len(returns.columns) for ticker in returns.columns}
    
    # Calculate performance manually with error handling
    try:
        weights_series = pd.Series(weights)
        
        # Ensure weights sum to 1
        weights_series = weights_series / weights_series.sum()
        
        # Calculate portfolio returns
        portfolio_returns = returns @ weights_series
        
        # Remove any NaN/inf from portfolio returns
        portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(portfolio_returns) == 0:
            raise ValueError("All portfolio returns are NaN")
        
        # Calculate metrics
        portfolio_return = portfolio_returns.mean() * 252
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        
        # Handle edge cases
        if np.isnan(portfolio_vol) or portfolio_vol == 0:
            portfolio_vol = 0.01  # Minimum volatility to avoid division by zero
        
        sharpe = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0.0
        
        # Ensure all metrics are finite
        portfolio_return = 0.0 if np.isnan(portfolio_return) or np.isinf(portfolio_return) else portfolio_return
        portfolio_vol = 0.01 if np.isnan(portfolio_vol) or np.isinf(portfolio_vol) else portfolio_vol
        sharpe = 0.0 if np.isnan(sharpe) or np.isinf(sharpe) else sharpe
        
        print(f"Equal weights portfolio: Return={portfolio_return:.2%}, Vol={portfolio_vol:.2%}, Sharpe={sharpe:.2f}")
        
        return weights, (portfolio_return, portfolio_vol, sharpe), mu, S
        
    except Exception as e:
        print(f"❌ Even equal weights calculation failed: {e}")
        # Ultimate fallback with zero metrics
        return weights, (0.0, 0.01, 0.0), mu, S

def efficient_frontier_curve(returns: pd.DataFrame, points=50):
    """
    Compute points along the efficient frontier.
    """
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    returns = returns.clip(lower=-0.5, upper=1.0)
    returns = returns[np.isfinite(returns).all(axis=1)]
    
    mu = expected_returns.mean_historical_return(returns)
    mu = mu.fillna(0).replace([np.inf, -np.inf], 0)
    
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
                if not (np.isnan(ret) or np.isnan(vol) or np.isinf(ret) or np.isinf(vol)):
                    frontier_y.append(vol)
                    frontier_x.append(ret)
                break
            except:
                continue
    
    return frontier_x, frontier_y