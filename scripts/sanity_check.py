# scripts/sanity_check.py
import pandas as pd
import os

DATA_DIR = "data"

def inspect_asset(df: pd.DataFrame, asset_name: str):
    print(f"\n=== 🧩 Inspecting {asset_name} ===")
    df = df.copy()
    df.columns = df.columns.str.lower()

    print(df.groupby("date")["ticker"].nunique().describe())
    print("\nDate range:")
    print(df["date"].agg(["min", "max", "nunique"]))

    print("\nMissing value summary:")
    print(df[["multi_factor_score", "return"]].isna().mean())

    # Check small-sample dates (under 10 assets)
    small_dates = df.groupby("date")["ticker"].nunique()
    small_dates = small_dates[small_dates < 10]
    if not small_dates.empty:
        print(f"\n⚠️ Dates with too few assets (<10): {len(small_dates)}")
        print(small_dates.head(10))
    else:
        print("\n✅ All dates have sufficient assets for backtest.")

    # Example quantile size check
    avg_n = df.groupby("date")["ticker"].nunique().mean()
    q_size = int(avg_n * 0.2)
    print(f"\nExpected per-side quantile cutoff (20%): {q_size} assets per tail.")


def main():
    eq_path = os.path.join(DATA_DIR, "factors_equities.csv")
    cr_path = os.path.join(DATA_DIR, "factors_crypto.csv")

    if not os.path.exists(eq_path) or not os.path.exists(cr_path):
        print("❌ Missing input files. Run build_factors.py first.")
        return

    eq = pd.read_csv(eq_path, parse_dates=["date"])
    cr = pd.read_csv(cr_path, parse_dates=["date"])

    inspect_asset(eq, "Equity")
    inspect_asset(cr, "Crypto")

    print("\n✅ Sanity check complete.")


if __name__ == "__main__":
    main()
