# QQQ Drawdown Risk & Regime Investment Dashboard
**Course Project: Investment / Trader’s Dashboard (ML4T)**

## Research Question
*Can a machine-learning model that uses market, volatility, and macro features predict whether QQQ will fall more than 7% below its current price at some point in the next 21 trading days, and does it provide a meaningful out-of-sample improvement over a simple volatility-only baseline, especially during stress periods?*

---

## Repository Contents
* `src/data_loader.py`: Automated ingestion of QQQ, VIX, and Treasury yield data via `yfinance`.
* `src/features.py`: Feature engineering pipeline computing realized volatility, technical momentum, and macro yield curve proxies, plus the 21-day forward 7% drawdown binary target.
* `src/model.py`: Time-series walk-forward training and evaluation pipeline for a Volatility-only Baseline (Logistic Regression) vs. Full Feature Model (XGBoost).
* `run_pipeline.py`: End-to-end script to generate datasets and train models.
* `app.py`: Interactive Streamlit dashboard for portfolio managers and risk officers.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-georgia-tech-username/qqq_drawdown_dashboard.git](https://github.com/your-georgia-tech-username/qqq_drawdown_dashboard.git)
   cd qqq_drawdown_dashboard