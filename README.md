# 📊 Quantitative Portfolio Analytics Platform

Production-grade portfolio optimization and risk analysis system. Processes 24,500+ days of market data across 25+ equities and cryptocurrencies using modern portfolio theory.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/leaalonzo/)

---

## Features

**Portfolio Optimization Engine**
- Mean-variance optimization (maximize Sharpe ratio / minimize volatility)
- Ledoit-Wolf covariance shrinkage with matrix regularization
- Multi-solver support (ECOS/SCS/OSQP) with automatic fallback
- Numerical stability handling: eigenvalue analysis, positive semi-definite enforcement

**Data Infrastructure**
- DuckDB analytics database for high-performance time-series analysis
- Yahoo Finance API integration with automated validation
- Robust data pipeline with error handling and outlier detection
- Efficient column-oriented storage with SQL interface

**Risk Analytics**
- Sharpe ratio, volatility, maximum drawdown analysis
- Rolling performance windows (60-day metrics)
- Portfolio risk attribution
- Strategy comparison and benchmark analysis

**Interactive Dashboard**
- Real-time portfolio visualization with Plotly
- Multiple optimization strategies and asset classes
- Dynamic weight allocation display
- Performance tracking with cumulative returns

---

## Tech Stack

Python • pandas • NumPy • PyPortfolioOpt • DuckDB • Streamlit • Plotly • yfinance

---

## Quick Start
```bash
# Setup
git clone https://github.com/leaalonzo/quant-portfolio-analytics-dev.git
cd quant-portfolio-analytics-dev
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python -m utils.fetch_all_data           # Fetch market data
python scripts/build_factors.py          # Process data
python scripts/run_backtest.py           # Run backtests
streamlit run app/app.py                 # Launch dashboard
```

---

## Project Structure
```bash
├── app/app.py                    # Streamlit dashboard
├── data/                         # CSV & DuckDB files
├── scripts/
│   ├── build_factors.py         # Data processing
│   └── run_backtest.py          # Portfolio backtesting
├── utils/
│   ├── optimize.py              # Portfolio optimization
│   ├── backtest.py              # Portfolio construction
│   └── risk.py                  # Risk analytics
│   └── fetch_all_data.py        # Fetch raw data from API
└── requirements.txt
```

---

## Optimization Implementation

### Mean-Variance Framework
- **Max Sharpe:** Maximize risk-adjusted returns using PyPortfolioOpt
- **Min Volatility:** Minimize portfolio variance
- **Covariance:** Ledoit-Wolf shrinkage for robust estimation
- **Constraints:** Position limits (0-50% per asset)

### Numerical Robustness
```python
# Covariance regularization
S = returns.cov() * 252
S = S + np.eye(len(S)) * 0.1  # 10% diagonal loading

# Matrix symmetry enforcement
S = (S + S.T) / 2

# Multi-solver fallback
solvers = ['ECOS', 'SCS', 'OSQP']
for solver in solvers:
    try:
        ef = EfficientFrontier(mu, S, solver=solver)
        # Optimization succeeds
    except:
        # Try next solver
```

### Data Cleaning Pipeline
- Infinity value replacement
- NaN handling (forward-fill → backward-fill → zero-fill)
- Extreme value clipping (-50% to +100% daily returns)
- Zero variance detection and removal

---

## Key Features

**Robust Optimization**
- Automatic solver selection and fallback mechanism
- Handles numerical instability in crypto data
- Equal-weight fallback when optimization fails
- Performance validation (detects invalid results)

**Data Quality Management**
- Dynamic NaN threshold (80% for equities, 50% for crypto)
- Asset-class specific data handling
- Comprehensive validation checks
- Diagnostic logging and error reporting

**Portfolio Strategies**
- Long-only and long-short portfolios
- Equal-weight and optimized allocations
- Multiple rebalancing approaches
- Strategy-specific asset selection

---

## Sample Results
```
Equity Portfolio - Max Sharpe
├── Expected Return: 12.4%
├── Volatility: 15.8%
├── Sharpe Ratio: 0.66
└── Assets: 15 equities

Crypto Portfolio - Min Volatility
├── Expected Return: 28.7%
├── Volatility: 42.1%
├── Sharpe Ratio: 0.63
└── Assets: 7 cryptocurrencies
```

---

## System Architecture
```
Data Collection → DuckDB Storage → Portfolio Backtesting → Optimization → Dashboard
     ↓                ↓                    ↓                    ↓            ↓
  yfinance         SQL DB          Equal-weight         PyPortfolioOpt   Streamlit
                                   Long/Short           Ledoit-Wolf      Plotly
```

---

## Performance Metrics

**Returns**
- Cumulative returns
- Annualized returns
- Rolling performance

**Risk**
- Volatility (annualized)
- Maximum drawdown
- Downside deviation

**Risk-Adjusted**
- Sharpe ratio
- Sortino ratio
- Rolling Sharpe

---

## Deployment

**Local**
```bash
streamlit run app/app.py
```

**Docker**
```bash
docker build -t quant-portfolio .
docker run -p 8501:8501 quant-portfolio
```

**Cloud:** Streamlit Cloud, AWS, GCP

---

## Database Queries
```sql
-- Portfolio performance
SELECT date, cumulative_return
FROM portfolio_results
WHERE asset_type = 'Equity';

-- Holdings analysis
SELECT ticker, COUNT(*) as appearances
FROM portfolio_holdings
GROUP BY ticker
ORDER BY appearances DESC;
```

---

## Author

**Lea Alonzo**  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/leaalonzo/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/leaalonzo)

---

## Disclaimer

Personal educational project developed independently. Not affiliated with any employer. Uses only publicly available market data. Not investment advice.

---

## License

MIT License

---

**⭐ Star this repo if you find it useful!**