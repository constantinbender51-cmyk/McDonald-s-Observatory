import pandas as pd
import numpy as np
from pathlib import Path

# ---------- 1. load ----------
CSV_FILE = Path("btc_daily.csv")
df = pd.read_csv(CSV_FILE, parse_dates=["date"]).sort_values("date")

# ---------- 2. create yesterday-only predictors ----------
# ---------- 1. MACD (12,26,9) ----------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

fast = 12
slow = 26
sig  = 9

macd_line = ema(df["close"], fast) - ema(df["close"], slow)
signal_line = ema(macd_line, sig)

# ---------- 2. build 20-day look-back ----------
lookback = 20
macd_cols  = [f"macd_{i}"  for i in range(lookback)]
sig_cols   = [f"sig_{i}"   for i in range(lookback)]

# shift so that row t contains macd[t-19]...macd[t]  (most recent last)
for i in range(lookback):
    df[macd_cols[i]]  = macd_line.shift(lookback - i)
    df[sig_cols[i]]   = signal_line.shift(lookback - i)

FEATURES = macd_cols + sig_cols          # 40 features
df["y"]  = (macd_line - signal_line).shift(-2)   # tomorrow's distance
df = df.dropna()                         # removes rows with NaN at ends
# ---------- 4. train/test split ----------
split = int(len(df) * 0.8)
train_df = df.iloc[:split]
test_df  = df.iloc[split:]

y_train = train_df["y"].values          #  <--  was missing
y_test  = test_df["y"].values           #  <--  was missing

# ---------- 5. standardise (from scratch) ----------
def zscore_fit(X):
    return X.mean(axis=0), X.std(axis=0, ddof=0)

def zscore_transform(X, mu, sigma):
    return (X - mu) / np.where(sigma == 0, 1, sigma)

mu, sigma = zscore_fit(train_df[FEATURES].values)
X_train = zscore_transform(train_df[FEATURES].values, mu, sigma)
X_test  = zscore_transform(test_df[FEATURES].values,  mu, sigma)

# ---------- 6. linear regression from scratch ----------
class LinReg:
    def fit(self, X, y):
        Xb = np.c_[np.ones(X.shape[0]), X]          # bias column
        self.theta = np.linalg.lstsq(Xb, y, rcond=None)[0]
        return self
    def predict(self, X):
        Xb = np.c_[np.ones(X.shape[0]), X]
        return Xb @ self.theta

model = LinReg().fit(X_train, y_train)
pred = model.predict(X_test)

# ---------- 7. metrics (from scratch) ----------
# ---------- metrics feast ----------

# ---------- leak litmus: shuffle X_test rows, keep y_test ----------
np.random.seed(42)                                    # reproducible
X_shuf = X_test.copy()
np.random.shuffle(X_shuf)                             # scramble features only
pred_shuf = model.predict(X_shuf)                     # predict on garbage
shuf_dir  = (np.sign(pred_shuf) == np.sign(y_test)).mean()

print("\n==========  SHUFFLE TEST  ==========")
print(f"Original dir-acc : {dir_acc:10.1%}")
print(f"Shuffled dir-acc : {shuf_dir:10.1%}")
print(f"Difference       : {dir_acc - shuf_dir:10.1%}")

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error, median_absolute_error)

y_true = y_test

# 1. regression
mae   = mean_absolute_error(y_true, pred)
rmse = np.sqrt(mean_squared_error(y_true, pred))
r2    = r2_score(y_true, pred)
mape  = mean_absolute_percentage_error(y_true, pred) * 100
medae = median_absolute_error(y_true, pred)

# 2. directional hit-rate on the *sign* of the distance
dir_acc = (np.sign(pred) == np.sign(y_true)).mean()

# 3. pearson & spearman
pearson = np.corrcoef(pred, y_true)[0, 1]
spear   = pd.Series(pred).corr(pd.Series(y_true), method='spearman')

# 4. residual diagnostics
residual = y_true - pred
pct_within_1_sigma = (np.abs(residual) <= residual.std()).mean()

print("==========  MACD-distance forecast  ==========")
print(f"MAE                    : {mae:10.4f}")
print(f"RMSE                   : {rmse:10.4f}")
print(f"R²                     : {r2:10.4f}")
print(f"MAPE                   : {mape:8.2f} %")
print(f"MedAE                  : {medae:10.4f}")
print(f"Direction accuracy     : {dir_acc:10.1%}")
print(f"Pearson r              : {pearson:10.3f}")
print(f"Spearman ρ             : {spear:10.3f}")
print(f"Resid within ±1σ       : {pct_within_1_sigma:10.1%}")
print(f"Residual mean          : {residual.mean():10.4f}")
print(f"Residual std           : {residual.std():10.4f}")
