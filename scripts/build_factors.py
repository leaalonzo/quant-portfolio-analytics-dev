# scripts/build_factors.py

import pandas as pd
import duckdb
from utils.factors import (
    compute_returns,
    momentum_factor,
    volatility_factor,
    value_factor,
    quality_factor,
    standardize_factors,
    combine_factors,
)

# === Load data ===
eq = pd.read_csv("data/raw_equities.csv", parse_dates=["date"])
cr = pd.read_csv("data/raw_crypto.csv", parse_dates=["date"])

# Standardize column names
#eq.rename(columns={"Ticker": "ticker", "Close": "close"}, inplace=True)
#cr.rename(columns={"Ticker": "ticker", "Close": "close"}, inplace=True)

# Add returns and compute factors
eq = compute_returns(eq)
eq = momentum_factor(eq)
eq = volatility_factor(eq)
eq = value_factor(eq)
eq = quality_factor(eq)

cr = compute_returns(cr)
cr = momentum_factor(cr)
cr = volatility_factor(cr)
cr = quality_factor(cr)  # crypto may not have value

# Standardize
factor_cols_eq = ["momentum", "volatility", "value", "quality"]
factor_cols_cr = ["momentum", "volatility", "quality"]

eq = standardize_factors(eq, factor_cols_eq)
cr = standardize_factors(cr, factor_cols_cr)

# Combine
eq = combine_factors(eq, factor_cols_eq)
cr = combine_factors(cr, factor_cols_cr)

# Add labels
eq["asset_type"] = "Equity"
cr["asset_type"] = "Crypto"

# Before saving
#eq = eq.drop(columns=["return"], errors="ignore")
#cr = cr.drop(columns=["return"], errors="ignore")

#if "return" not in cr.columns:
#    cr = cr.sort_values(["ticker", "date"])
#    cr["return"] = cr.groupby("ticker")["close"].pct_change()

# Save outputs
eq.to_csv("data/factors_equities.csv", index=False)
cr.to_csv("data/factors_crypto.csv", index=False)

# === Optional: Store in DuckDB and create combined view ===
con = duckdb.connect("data/factors.duckdb")

con.execute("CREATE OR REPLACE TABLE factors_equities AS SELECT * FROM eq")
con.execute("CREATE OR REPLACE TABLE factors_crypto AS SELECT * FROM cr")

# Align columns for union (missing ones replaced with NULL)
common_cols = set(eq.columns) & set(cr.columns)
query = f"""
CREATE OR REPLACE VIEW factors_all AS
SELECT {', '.join(common_cols)} FROM factors_equities
UNION ALL
SELECT {', '.join(common_cols)} FROM factors_crypto;
"""
con.execute(query)

print(con.execute("SELECT asset_type, COUNT(*) FROM factors_all GROUP BY 1;").fetchdf())
con.close()
