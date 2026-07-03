# Gold-Equity Regime Switching ML

A machine learning project for analyzing regime-dependent relationships between gold and Indian sectoral equities.

## Project Summary

This project studies how gold's predictive role changes across market regimes in India using:

- Hidden Markov Models (HMM) for regime detection
- Regime-specific XGBoost models
- SHAP for explainability
- Backtesting and forward forecasting
- A Streamlit dashboard for live monitoring

## Dataset

- Frequency: Daily
- Period: 2014-01-01 to 2024-12-31
- Market: Indian equities plus macro and commodity variables
- Main assets:
  - Gold
  - Nifty 50
  - Bank Nifty
  - IT
  - India VIX
  - USD/INR
  - Oil
  - Silver

## Pipeline Structure

### Layer 1 - Data Pipeline
`layer1_data_pipeline.py`

- Fetches market data using `yfinance`
- Engineers market, volatility, momentum, correlation, and microstructure features
- Saves the master dataset

### Layer 2 - Regime Detection
`layer2_regime_detection.py`

- Fits Gaussian HMM models
- Detects hidden market regimes
- Labels data into:
  - Crisis
  - Late_Cycle
  - Expansion

### Layer 3 - Regime-Specific ML + SHAP
`layer3_models_shap.py`

- Trains regime-specific XGBoost models
- Uses compact feature selection
- Evaluates using walk-forward validation
- Computes SHAP feature importance by regime

Key finding:
Gold's predictive importance changes materially by regime.

### Layer 4 - Backtest Report
`layer4_backtest_report.py`

- Backtests regime-aware model signals
- Compares against a global model and buy-and-hold
- Generates summary reports and visualizations

### Layer 5 - Forecasting
`layer5_forecast.py`

- Generates 5-day and 30-day price forecasts
- Builds confidence bands
- Runs rolling forecast validation

### Live Prediction Script
`predict_tomorrow.py`

- Fetches recent market data
- Detects the current regime
- Predicts the next-day Nifty Bank move
- Saves text and JSON outputs

### Dashboard
`dashboard.py`

- Streamlit dashboard
- Live regime monitoring
- Tomorrow signal
- Forecast chart
- SHAP and regime visualization panels

## Important Results

### Regime-Specific SHAP

From the final improved model:

- Crisis: Gold SHAP = 0.000085, rank #11
- Late_Cycle: Gold SHAP = 0.000562, rank #1
- Expansion: Gold SHAP = 0.000249, rank #4

Interpretation:
Gold is most informative in Late_Cycle, less dominant in full Crisis, and still structurally relevant in Expansion.

### Final Honest Out-of-Sample Performance

Test window: 2023-2024

- Crisis_bank: R2 = -0.027
- Late_Cycle_bank: R2 = -0.041
- Expansion_bank: R2 = 0.000
- Ensemble: R2 = -0.004
- Global_bank: R2 = -0.072

This means the final model is stronger as a regime-aware research and explainability framework than as a production trading signal.

### Directional Threshold Result

- Best threshold: +/-0.30%
- Direction accuracy at threshold: 68.18%

## Outputs

### Data

- `data/regime_labeled_data.csv`

### Models

- `models/Crisis_bank.pkl`
- `models/Late_Cycle_bank.pkl`
- `models/Expansion_bank.pkl`
- `models/Global_bank.pkl`
- `models/hmm_model.pkl`
- `models/hmm_scaler.pkl`
- `models/ensemble_model.pkl`

### Reports

- `outputs/layer3_improved_results.txt`
- `outputs/layer4_results.txt`
- `outputs/layer5_forecast.txt`
- `outputs/prediction_latest.json`

### Visualizations

- `visualizations/regime_timeline.png`
- `visualizations/regime_boxplot.png`
- `visualizations/transition_heatmap.png`
- `visualizations/shap_summary_Crisis.png`
- `visualizations/shap_summary_Late_Cycle.png`
- `visualizations/shap_summary_Expansion.png`
- `visualizations/price_forecast.png`
- `visualizations/tomorrow_prediction.png`
- `visualizations/cumulative_returns.png`

## How to Run

### Run tomorrow prediction

```bash
python predict_tomorrow.py
```

### Run dashboard

```bash
streamlit run dashboard.py
```

## Tech Stack

- Python
- pandas
- numpy
- yfinance
- hmmlearn
- xgboost
- shap
- matplotlib
- plotly
- streamlit
- scikit-learn

## Disclaimer

This is an academic machine learning project and not financial advice.
