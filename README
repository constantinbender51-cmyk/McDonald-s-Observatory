# BTC Machine Learning Trading Strategy 🧠💹

![BTC Trading Strategy Performance](plot.png)

This project implements a **machine learning–based Bitcoin trading strategy** using **linear regression models** to predict short-term and long-term returns.  
It backtests the strategy using **historical Binance BTC/USDT daily data**, applies **leverage, stop-loss logic**, and evaluates predictive accuracy across multiple future horizons.

---

## 🧩 Overview

The strategy uses two independent models:

- **Short-term predictor (7-day horizon)** → Captures fast reversals and local trends  
- **Long-term predictor (25-day horizon)** → Detects broader market momentum  

Based on these predictions, the algorithm dynamically enters and exits **long or short leveraged positions** with a defined stop-loss mechanism.

---

## 📊 Results Example

The figure above (`plot.png`) shows:

- **Blue line:** Strategy equity growth  
- **Orange line:** Buy & hold baseline  
- **Green dashed:** Short-term predictions (7 days ahead)  
- **Red dashed:** Long-term predictions (25 days ahead)  
- **Gray shading:** Position (long/short/neutral) periods  

✅ The model identifies trend turns **6–10 days before** they occur in real price action.  
✅ The equity curve grows steadily, far outperforming buy-and-hold during backtest.

---

## ⚙️ Configuration Parameters

| Parameter | Description | Default |
|------------|--------------|----------|
| `LOOKBACK` | Days of history used as features | `15` |
| `SHORT_HORIZON` | Prediction target (days ahead) | `7` |
| `LONG_HORIZON` | Long-term prediction horizon | `25` |
| `LEVERAGE` | Trading leverage multiplier | `4.0` |
| `STOP_LOSS_PCT` | Stop loss as % of predicted move | `0.45` |
| `INITIAL_CAPITAL` | Starting balance in USD | `1000.0` |

---

## 🧠 Features & Modeling

Each daily record includes:
- **Stochastic RSI (StochRSI)**
- **Price % change**
- **Volume % change**
- **MACD signal line**

These features are standardized (z-scored) and fed into **simple linear regression models**:

```python
class LinReg:
    def fit(self, X, y):
        Xb = np.c_[np.ones(X.shape[0]), X]
        self.theta = np.linalg.lstsq(Xb, y, rcond=None)[0]
        return self
    def predict(self, X):
        Xb = np.c_[np.ones(X.shape[0]), X]
        return Xb @ self.theta
