# Stock Trend Prediction

An ML-powered web application for NYSE/NASDAQ stock trend forecasting. Combines technical indicator analysis with NLP-based financial news sentiment scoring. Built as a team project for the "Artificial Intelligence Fundamentals" course at Warsaw University of Technology.

## How it works

1. Historical stock data is pulled via the yfinance API
2. Technical indicators are computed (RSI, MACD, Bollinger Bands, SMA, EMA, volatility, etc.)
3. Financial news is fetched and scored for sentiment using VADER and FinBERT
4. A Random Forest or Gradient Boosting classifier predicts the price direction (up/down) for 1, 3, or 7 days
5. A separate regressor estimates the predicted price level
6. Results are displayed on an interactive Flask dashboard with Chart.js

## Features

- **Multiple prediction horizons** — 1-day, 3-day, and 7-day forecasts
- **Two ML models** — Random Forest and Gradient Boosting, both auto-cached and retrained on demand
- **30+ engineered features** — lagged prices, returns, moving averages, RSI, MACD, Bollinger Band position, volume ratios, day of week
- **Dual sentiment analysis** — VADER (rule-based) + FinBERT (neural) on financial news headlines
- **Economic indicators** — VIX, Fed Funds Rate, CPI via FRED API
- **Model evaluation** — accuracy, precision, recall, F1, MAE, RMSE displayed in the UI
- **Auto-expiring model cache** — models retrain automatically after 1 hour

## Project structure

```
├── app.py          # Flask routes and API endpoints
├── data.py         # Stock data loading and feature engineering (yfinance)
├── model.py        # Training, prediction, caching, and evaluation
├── sentiment.py    # VADER + FinBERT sentiment scoring
├── news.py         # Financial news fetching
├── social.py       # Social media sentiment stub
├── report.py       # Report generation utilities
├── train_all.py    # Batch training script for all model/horizon combos
└── templates/
    └── index.html  # Dashboard UI with Chart.js
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env       # add your API keys (NewsAPI, FRED)
python app.py
```

Open http://localhost:5000 in your browser.

## Tech

Python, Flask, scikit-learn, yfinance, VADER, FinBERT (HuggingFace Transformers), Chart.js, FRED API
