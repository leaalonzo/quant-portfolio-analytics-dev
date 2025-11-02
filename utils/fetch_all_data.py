import yfinance as yf
import pandas as pd
from pathlib import Path
import yaml

# -------------------------
# Config loader
# -------------------------
def load_config(path="config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -------------------------
# Generic fetch function
# -------------------------
def fetch_yfinance_data(tickers, start=None, end=None):
    print(f"📡 Fetching {len(tickers)} tickers from Yahoo Finance...")
    data = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True)

    frames = []
    for ticker in tickers:
        df = data[ticker].copy()
        df["ticker"] = ticker
        df["return"] = df["Close"].pct_change()
        frames.append(df)

    combined = pd.concat(frames)
    combined.reset_index(inplace=True)
    combined.dropna(inplace=True)
    print("✅ Data fetched and cleaned.")
    return combined

# -------------------------
# Save to CSV
# -------------------------
def save_data(df, filename):
    Path("data").mkdir(exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"💾 Data saved to {filename}")

# -------------------------
# Main script
# -------------------------
if __name__ == "__main__":
    config = load_config()

    # --- Equities ---
    eq_cfg = config["equities"]
    eq_data = fetch_yfinance_data(eq_cfg["tickers"], start=eq_cfg.get("start_date"), end=eq_cfg.get("end_date"))
    eq_data.columns = eq_data.columns.str.lower()
    save_data(eq_data, "data/raw_equities.csv")

    # --- Crypto ---
    crypto_cfg = config["crypto"]
    crypto_data = fetch_yfinance_data(crypto_cfg["tickers"], start=crypto_cfg.get("start_date"), end=crypto_cfg.get("end_date"))
    crypto_data.columns = crypto_data.columns.str.lower()
    save_data(crypto_data, "data/raw_crypto.csv")
