# 📊 Quant Portfolio Analytics

A multi-asset **quant research dashboard** for equities and crypto.  
It fetches market data, computes factor exposures (momentum, volatility, value, quality), and stores everything in **DuckDB** for analysis and visualization.

---

## 🚀 Features
- 🧮 **Factor Engine:** Momentum, Volatility, Value, Quality  
- 💾 **Unified Data Store:** DuckDB with equity + crypto factors  
- 📈 **Backtesting Ready:** Clean pipeline for portfolio analytics  
- 🧠 **Extensible Design:** Modular utils + YAML configuration  
- 🧰 **Dashboard Deployment:** Streamlit or Hugging Face Spaces  

---

## 🗂️ Structure
```bash
quant-portfolio-analytics-dev/
├── app/ # Streamlit dashboard
├── data/ # CSV & DuckDB data files
├── scripts/ # CLI build scripts
├── utils/ # Factor and data modules
├── config.yml # Config (tickers, dates)
├── requirements.txt
└── README.md
```
---

## ⚙️ Setup
```bash
git clone https://github.com/leaalonzo/quant-portfolio-analytics-dev.git
cd quant-portfolio-analytics-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🧠 Usage
1. Fetch market data
```bash python -m scripts.build_data ```
2. Build factor scores
```bash python -m scripts.build_factors ```
3. Query data
```bash
-- in DuckDB CLI
SELECT * FROM factors_all LIMIT 10;
```
4. Launch dashboard
```bash streamlit run app/main.py ```

---

## 📅 Roadmap
```bash
Week	Task	Status
1	Data pipeline	✅
2	Factor construction	✅
3	Backtesting engine	✅
4	Portfolio optimization	⏳
5	Risk analytics	⏳
6	Dashboard UI	⏳
7	AI explainability	⏳
8	Deployment & docs	⏳
```

---

## 🧰 Tech Stack
Python • pandas • numpy • yfinance • DuckDB • Streamlit

## 👤 Contact
You may contact me through https://www.linkedin.com/in/leaalonzo/

---

## ⚠️ Disclaimer
This project is a personal educational initiative developed entirely outside of my employment.
It is not affiliated with, endorsed by, or representative of any current or past employer.