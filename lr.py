import pandas as pd
import numpy as np
from pathlib import Path
import time

FORECAST_HORIZON = 40          # <-- look-ahead days (change this only)

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
df["y"] = (macd_line - signal_line).shift(-FORECAST_HORIZON)   # <-- use var
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
# -------# distance between MACD and its signal line
df["macd_signal"] = macd_line - signal_line

# ---------- Stochastic RSI (14-period) ----------
def stoch_rsi(close, rsi_period=14, stoch_period=14):
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=rsi_period).mean()
    avg_loss = pd.Series(loss).rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    stoch_rsi = (rsi - rsi.rolling(window=stoch_period).min()) / \
                (rsi.rolling(window=stoch_period).max() - rsi.rolling(window=stoch_period).min())
    return stoch_rsi

df["stoch_rsi"] = stoch_rsi(df["close"])



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

print("index  actual     pred     error")
for i in range(5):
    a = y_test[i]
    p = pred[i]
    print(f"{i:5d}  {a:8.2f}  {p:8.2f}  {a-p:8.2f}")
    time.sleep(0.1)


# StochRSI strategy: buy < 0.2, sell > 0.8
df["stoch_signal"] = np.where(df["stoch_rsi"] < 0.2, 1,
                                np.where(df["stoch_rsi"] > 0.8, -1, 0))

# ---------- 8. one-day price-change % and pretty print ----------
close = df["close"].values

# first and last index of the test window
first_test_idx = split
last_test_idx  = len(close) - 1 - FORECAST_HORIZON

# next-day percentage change
pct_change = (close[first_test_idx+1 : last_test_idx+1] /
              close[first_test_idx : last_test_idx] - 1) * 100

# align all series to the same window
y_test      = y_test[:len(pct_change)]
pred        = pred[:len(pct_change)]
macd_signal = df["macd_signal"].values[split : split + len(pct_change)]

# --- grab the date column for the test window -----------------------------
test_dates = df["date"].iloc[split : split + len(pct_change)].reset_index(drop=True)

capital   = 1000.0
buy_hold  = 1000.0
macd_real = 1000.0
# ----- StochRSI strategy -----
stoch_real = 1000.0
pos_stoch  = 0
last_stoch_flip = 0


pos_model = 0
pos_real  = 0
last_model_flip = 0
last_real_flip  = 0

print("\ndate          idx    pred   pctChg%   macd-sig     model      buy&hold    real-MACD   stochRSI")
for i in range(len(pred)):
    new_model_sign = int(np.sign(pred[i]))
    new_real_sign  = int(np.sign(macd_signal[i]))
    new_stoch_sign = int(np.sign(df["stoch_signal"].iloc[split + i]))

    # ----- model strategy -----
    if new_model_sign != pos_model:
        cum_ret = (np.prod(1 + pos_model * pct_change[last_model_flip:i+1]/100) - 1)
        capital *= 1 + cum_ret
        pos_model       = new_model_sign
        last_model_flip = i + 1

    # ----- real-MACD strategy -----
    if new_real_sign != pos_real:
        cum_ret = (np.prod(1 + pos_real * pct_change[last_real_flip:i+1]/100) - 1)
        macd_real *= 1 + cum_ret
        pos_real       = new_real_sign
        last_real_flip = i + 1

    # ----- StochRSI strategy -----
    
    if new_stoch_sign != pos_stoch and new_stoch_sign != 0:
        cum_ret = (np.prod(1 + pos_stoch * pct_change[last_stoch_flip:i+1]/100) - 1)
        stoch_real *= 1 + cum_ret
        pos_stoch       = new_stoch_sign
        last_stoch_flip = i + 1
        

    # ----- buy & hold -----
    buy_hold *= 1 + pct_change[i]/100

    # pretty print with date
    print(f"{test_dates[i].strftime('%Y-%m-%d')}  {i:3d}  {pred[i]:7.2f}  "
          f"{pct_change[i]:6.2f}%  {macd_signal[i]:8.2f}   {capital:8.2f}   "
          f"{buy_hold:8.2f}   {macd_real:8.2f}   {stoch_real:8.2f}")
    time.sleep(0.01)
