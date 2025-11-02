# scripts/run_backtest.py

from utils.backtest import form_portfolios, compute_performance
import pandas as pd

# === Load factor data ===
eq = pd.read_csv("data/factors_equities.csv", parse_dates=["date"])
cr = pd.read_csv("data/factors_crypto.csv", parse_dates=["date"])

# Ensure consistent column names
#eq.rename(columns={"Ticker": "ticker", "Date": "date"}, inplace=True)
#cr.rename(columns={"Ticker": "ticker", "Date": "date"}, inplace=True)

# Unify factor column name
eq["multi_factor_score"] = eq["multi_factor_score"]
cr["multi_factor_score"] = cr["multi_factor_score"]

# Assign asset types
eq["asset_type"] = "Equity"
cr["asset_type"] = "Crypto"

# === Add return column (simple % change) ===
# Assume you have raw price data in /data/raw_equities.csv and /data/raw_crypto.csv
try:
    prices_eq = pd.read_csv("data/raw_equities.csv", parse_dates=["date"])
    prices_cr = pd.read_csv("data/raw_crypto.csv", parse_dates=["date"])

    prices_eq["return"] = prices_eq.groupby("ticker")["close"].pct_change()
    prices_cr["return"] = prices_cr.groupby("ticker")["close"].pct_change()

    # Merge returns into factor data
    eq = pd.merge(eq, prices_eq[["date", "ticker", "return"]], on=["date", "ticker"], how="left")
    cr = pd.merge(cr, prices_cr[["date", "ticker", "return"]], on=["date", "ticker"], how="left")

except FileNotFoundError:
    print("⚠️ raw_equities.csv or raw_crypto.csv not found — generating synthetic returns for demo.")
    eq["return"] = eq.groupby("ticker")["multi_factor_score"].diff() * 0.01
    cr["return"] = cr.groupby("ticker")["multi_factor_score"].diff() * 0.01

# === Combine data ===
df = pd.concat([eq, cr], ignore_index=True)

# Ensure both datasets share a common start date (for consistency)
crypto_start = df[df["asset_type"] == "Crypto"]["date"].min()
df = df[df["date"] >= crypto_start]
print(f"📅 Backtest start aligned to {crypto_start.date()} to include both asset classes.")

# === Run backtest ===
portfolios = form_portfolios(df, quantile=0.2, long_short=True, group_col="asset_type")

perf_eq, stats_eq = compute_performance(portfolios[portfolios["asset_type"] == "Equity"])
perf_eq["asset_type"] = "Equity"

perf_cr, stats_cr = compute_performance(portfolios[portfolios["asset_type"] == "Crypto"])
perf_cr["asset_type"] = "Crypto"

print("CRYPTO PORTFOLIOS")
print(perf_cr.tail())

perf = pd.concat([perf_eq, perf_cr])
stats = pd.DataFrame([stats_eq, stats_cr])

print(perf.head())

# === Save results ===
perf.to_csv("data/portfolio_results.csv", index=False)
# Combine results and save stats cleanly
stats_all = []

# Equity
stats_eq["asset_type"] = "Equity"
stats_all.append(stats_eq)

# Crypto (if exists)
if not perf_cr.empty:
    stats_cr["asset_type"] = "Crypto"
    stats_all.append(stats_cr)
else:
    print("⚠️ Crypto results empty — skipped adding to stats.")

# Create DataFrame
stats_df = pd.DataFrame(stats_all)

# Save outputs
stats_df.to_csv("data/portfolio_stats.csv", index=False)
perf.to_csv("data/portfolio_results.csv", index=False)

print("✅ Portfolio results saved to data/portfolio_results.csv")
print("✅ Portfolio stats saved to data/portfolio_stats.csv")

print("\n📊 Portfolio Stats Summary:")
print(stats)
