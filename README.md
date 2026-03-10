# XGBoost vs TabNet for Financial Time Series Classification

This script compares two powerful models for tabular financial data classification:

1. **XGBoost** (Extreme Gradient Boosting) - Tree-based ensemble method
2. **TabNet** (Deep Learning for Tabular Data) - Attention-based neural network

---

## 🎯 Task Definition

| Component | Description |
|-----------|-------------|
| **Objective** | Predict whether SPY will fall more than 5% over the next 20 trading days |
| **Target Variable** | `future_20d_drawdown_flag` (binary: 1 if drawdown < -5%, else 0) |
| **Forecast Horizon** | 20 trading days (~1 calendar month) |
| **Drawdown Threshold** | -5% (`DRAWDOWN_THRESHOLD = -0.05`) |
| **Data Period** | January 2018 – Present |
| **Test Split** | 25% of data (`TEST_SIZE = 0.25`) |

---

## ⚙️ Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `STOCK_TICKERS` | AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, BAC, GS, TSLA | Individual stock universe |
| `MARKET_TICKER` | SPY | S&P 500 ETF (target benchmark) |
| `SECTOR_TICKER` | XLK | Technology sector ETF |
| `MACRO_TICKER` | TLT | 20+ Year Treasury Bond ETF |
| `DRAWDOWN_THRESHOLD` | -0.05 | Threshold for classification target |
| `FORECAST_HORIZON` | 20 | Number of trading days to forecast |
| `RANDOM_STATE` | 42 | Random seed for reproducibility |
| `TEST_SIZE` | 0.25 | Proportion of data reserved for testing |

---

## 📊 Feature Engineering

### Core Features Table

| Feature | Category | Description |
|---------|----------|-------------|
| `mean_return` | Cross-sectional | Average daily return across all stocks |
| `volatility` | Cross-sectional | Standard deviation of returns across stocks |
| `momentum_20` | Technical | 20-day price change, averaged across stocks |
| `momentum_60` | Technical | 60-day price change, averaged across stocks |
| `drawdown` | Risk | Average drawdown from peak across stocks |
| `volume_trend` | Liquidity | Ratio of 20-day to 60-day rolling volume average |
| `sector_beta` | Factor | Rolling 60-day beta vs. XLK sector ETF |
| `macro_factor` | Macro | Average 20-day return of TLT, GLD, DBC |

### Macro Assets Downloaded

| Ticker | Asset Class | Role in Model |
|--------|-------------|---------------|
| TLT | Long-term Treasuries | Interest rate sensitivity proxy |
| GLD | Gold | Safe-haven / inflation hedge proxy |
| DBC | Commodities | Economic growth / inflation proxy |

---

## 🤖 Model Architectures

### XGBoost Configuration

| Hyperparameter | Value | Purpose |
|----------------|-------|---------|
| `n_estimators` | 400 | Number of boosting rounds |
| `max_depth` | 6 | Maximum tree depth |
| `learning_rate` | 0.05 | Step size shrinkage |
| `subsample` | 0.9 | Row sampling ratio |
| `colsample_bytree` | 0.9 | Column sampling ratio |
| `scale_pos_weight` | n_neg/n_pos | Handle class imbalance |
| `eval_metric` | logloss | Optimization objective |

### TabNet Configuration

| Hyperparameter | Value | Purpose |
|----------------|-------|---------|
| `n_d` / `n_a` | 16 | Number of decision/attention units |
| `n_steps` | 5 | Number of sequential attention steps |
| `gamma` | 1.5 | Coefficient for feature re-use penalty |
| `lambda_sparse` | 1e-4 | Sparsity regularization |
| `optimizer_fn` | Adam | Optimization algorithm |
| `lr` | 2e-2 | Learning rate |
| `mask_type` | entmax | Sparse feature selection mask |
| `max_epochs` | 1000 | Training epochs limit |
| `patience` | 15 | Early stopping patience |
| `batch_size` | 256 | Mini-batch size |
| `virtual_batch_size` | 128 | Batch size for ghost batch norm |

---

## 📈 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP+TN)/Total | Overall correctness |
| **ROC-AUC** | Area under ROC curve | Discrimination ability across thresholds |
| **Precision** | TP/(TP+FP) | Reliability of positive predictions |
| **Recall** | TP/(TP+FN) | Coverage of actual positive cases |
| **F1-Score** | 2×(Prec×Rec)/(Prec+Rec) | Harmonic mean of precision & recall |

*TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative*

---

## 📦 Installation Requirements

```bash
# Core dependencies
pip install numpy pandas scikit-learn xgboost torch

# TabNet (official implementation)
pip install pytorch-tabnet

# Data & visualization
pip install yfinance plotly matplotlib

# Optional: Jupyter support
pip install jupyter notebook
