import pandas as pd
import duckdb
from pathlib import Path

# Load CSVs
eq_df = pd.read_csv("data/raw_equities.csv", parse_dates=["date"])
crypto_df = pd.read_csv("data/raw_crypto.csv", parse_dates=["date"])

# Rename columns for consistency
#eq_df.rename(columns={"ticker": "symbol", "date": "date"}, inplace=True)
#crypto_df.rename(columns={"ticker": "symbol", "Date": "date"}, inplace=True)

# Add asset_type
eq_df["asset_type"] = "equity"
crypto_df["asset_type"] = "crypto"

# Keep relevant columns
cols = ["date", "ticker", "open", "high", "low", "close", "volume", "return", "asset_type"]
eq_df = eq_df[cols]
crypto_df = crypto_df[cols]

# Merge datasets
combined_df = pd.concat([eq_df, crypto_df])
combined_df.sort_values(by=["date", "ticker"], inplace=True)
combined_df.reset_index(drop=True, inplace=True)

# Save to DuckDB
Path("data").mkdir(exist_ok=True)
con = duckdb.connect("data/market_data.duckdb")
con.execute("CREATE OR REPLACE TABLE market_data AS SELECT * FROM combined_df")

print("💾 Combined data saved with asset_type column!")
