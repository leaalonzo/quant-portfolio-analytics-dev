# scripts/run_backtest.py
import pandas as pd
from utils.backtest import form_portfolios, compute_performance
import os
import duckdb

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# === Load prepared factor data ===
df_eq = pd.read_csv(os.path.join(DATA_DIR, "factors_equities.csv"), parse_dates=["date"])
df_cr = pd.read_csv(os.path.join(DATA_DIR, "factors_crypto.csv"), parse_dates=["date"])
df = pd.concat([df_eq, df_cr], ignore_index=True)
df.columns = df.columns.str.lower()

# === Configuration ===
factors = ["momentum", "value", "quality", "multi_factor_score"]
asset_types = ["Equity", "Crypto"]
modes = ["long_short", "long_only"]

# === Results containers ===
all_perf = []
all_stats = []
all_asset_returns = []
all_holdings = []  # NEW: Track portfolio holdings

# === Run backtests ===
for asset in asset_types:
    for factor in factors:
        for mode in modes:
            print(f"\n▶️ Running {asset} — {factor} — {mode}")
            subset = df[df["asset_type"] == asset].copy()
            if subset.empty:
                continue
            
            subset["multi_factor_score"] = subset[factor]
            print(subset[['date','ticker','multi_factor_score','return']].tail())
            print(subset.dropna(subset=['multi_factor_score','return']).shape)
            
            quant = 0.2 if asset == "Equity" else 0.4  # more inclusive for small universes
            
            # --- Portfolio formation now returns both portfolio & per-asset returns ---
            portfolios, asset_returns = form_portfolios(
                subset,
                quantile=quant,
                long_short=(mode == "long_short"),
                group_col="asset_type",
            )
            
            perf, stats = compute_performance(portfolios)
            
            if perf.empty:
                print(f"⚠️ No results for {asset} — {factor} — {mode}")
                continue
            
            # Tag metadata
            perf["asset_type"] = asset
            perf["factor"] = factor
            perf["mode"] = mode
            
            stats["asset_type"] = asset
            stats["factor"] = factor
            stats["mode"] = mode
            
            asset_returns["asset_type"] = asset
            asset_returns["factor"] = factor
            asset_returns["mode"] = mode
            
            # NEW: Save which tickers were selected by this strategy
            if not portfolios.empty:
                tickers_used = sorted(portfolios['ticker'].unique())
                holdings_record = {
                    'asset_type': asset,
                    'factor': factor,
                    'mode': mode,
                    'tickers': ','.join(tickers_used),  # Comma-separated string
                    'num_tickers': len(tickers_used)
                }
                all_holdings.append(holdings_record)
                print(f"   ✓ Selected {len(tickers_used)} tickers: {', '.join(tickers_used[:5])}{'...' if len(tickers_used) > 5 else ''}")
            
            all_perf.append(perf)
            all_stats.append(stats)
            all_asset_returns.append(asset_returns)

# === Combine & Save Results ===
perf_df = pd.concat(all_perf, ignore_index=True)
stats_df = pd.DataFrame(all_stats)
holdings_df = pd.DataFrame(all_holdings)  # NEW

# For asset returns, concatenate but DON'T keep the metadata columns in the CSV
asset_ret_df = pd.concat(all_asset_returns, ignore_index=False)

# Remove metadata columns before saving to CSV
asset_ret_csv = asset_ret_df.drop(columns=['asset_type', 'factor', 'mode'], errors='ignore')

perf_df.to_csv(f"{DATA_DIR}/portfolio_results_all.csv", index=False)
stats_df.to_csv(f"{DATA_DIR}/portfolio_stats_all.csv", index=False)
asset_ret_csv.to_csv(f"{DATA_DIR}/asset_daily_returns.csv")
holdings_df.to_csv(f"{DATA_DIR}/portfolio_holdings.csv", index=False)  # NEW

print("\n✅ All results saved to:")
print(" - data/portfolio_results_all.csv")
print(" - data/portfolio_stats_all.csv")
print(" - data/asset_daily_returns.csv")
print(" - data/portfolio_holdings.csv")  # NEW

# === Store to DuckDB ===
con = duckdb.connect("data/backtest_results.duckdb")
con.execute("CREATE OR REPLACE TABLE portfolio_results AS SELECT * FROM perf_df")
con.execute("CREATE OR REPLACE TABLE portfolio_stats AS SELECT * FROM stats_df")
con.execute("CREATE OR REPLACE TABLE portfolio_holdings AS SELECT * FROM holdings_df")  # NEW
# Keep metadata in DuckDB version if you want
con.execute("CREATE OR REPLACE TABLE asset_daily_returns AS SELECT * FROM asset_ret_df")
con.close()

print("✅ Saved results to DuckDB: data/backtest_results.duckdb")

# Print summary
print("\n📊 Summary:")
print(f"Total strategies run: {len(holdings_df)}")
print("\nHoldings by strategy:")
print(holdings_df[['asset_type', 'factor', 'mode', 'num_tickers']])