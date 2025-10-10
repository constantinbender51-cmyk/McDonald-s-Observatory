import pandas as pd
import numpy as np
from pathlib import Path
import time

FORECAST_HORIZON = 40          # <-- look-ahead days (change this only)

# ---------- 1. load ----------
CSV_FILE = Path("btc_daily.csv")
df = pd.read_csv(CSV_FILE, parse_dates=["date"]).sort_values("date")


# ---------- 2. create forward-horizon target ----------
close   = df["close"].values
volume  = df["volume"].values

# ---------- 2. create forward-horizon target ----------
# keep it as a Series so we can shift
close_series = df["close"]
df["y"] = (close_series.shift(-FORECAST_HORIZON) / close_series - 1) * 100

# ---------- 3. compute raw series that we need ----------
# 3a. StochRSI (14,14)
def stoch_rsi(close, rsi_period=14, stoch_period=14):
    delta = pd.Series(close).diff()
    gain  = np.where(delta > 0, delta, 0)
    loss  = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(rsi_period).mean()
    avg_loss = pd.Series(loss).rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    stoch = (rsi - rsi.rolling(stoch_period).min()) / \
            (rsi.rolling(stoch_period).max() - rsi.rolling(stoch_period).min())
    return stoch.values

df["stoch_rsi"] = stoch_rsi(df["close"])

# 3b. 1-day % change of close and volume
df["pct_chg"]   = df["close"].pct_change() * 100
df["vol_pct_chg"] = df["volume"].pct_change() * 100

# 3c. MACD – signal
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
macd_line = ema(df["close"], 12) - ema(df["close"], 26)
signal_line = ema(macd_line, 9)
df["macd_signal"] = macd_line - signal_line

# ---------- 4. build 20-day look-back ----------
lookback = 20
stoch_cols = [f"stoch_{i}"     for i in range(lookback)]
pct_cols   = [f"pct_{i}"       for i in range(lookback)]
vol_cols   = [f"vol_{i}"       for i in range(lookback)]
macd_cols  = [f"macd_{i}"      for i in range(lookback)]

for i in range(lookback):
    df[stoch_cols[i]] = df["stoch_rsi"].shift(lookback - i)
    df[pct_cols[i]]   = df["pct_chg"].shift(lookback - i)
    df[vol_cols[i]]   = df["vol_pct_chg"].shift(lookback - i)
    df[macd_cols[i]]  = df["macd_signal"].shift(lookback - i)

FEATURES = stoch_cols + pct_cols + vol_cols + macd_cols   # 80 features
df = df.dropna()   # drops rows with NaN at both ends

# ---------- 5. train/test split ----------
split = int(len(df) * 0.8)
train_df = df.iloc[:split]
test_df  = df.iloc[split:]

y_train = train_df["y"].values
y_test  = test_df["y"].values

# ---------- 6. standardise ----------
def zscore_fit(X):
    return X.mean(axis=0), X.std(axis=0, ddof=0)

def zscore_transform(X, mu, sigma):
    return (X - mu) / np.where(sigma == 0, 1, sigma)

mu, sigma = zscore_fit(train_df[FEATURES].values)
X_train = zscore_transform(train_df[FEATURES].values, mu, sigma)
X_test  = zscore_transform(test_df[FEATURES].values,  mu, sigma)

# ---------- 7. linear regression ----------
class LinReg:
    def fit(self, X, y):
        Xb = np.c_[np.ones(X.shape[0]), X]
        self.theta = np.linalg.lstsq(Xb, y, rcond=None)[0]
        return self
    def predict(self, X):
        Xb = np.c_[np.ones(X.shape[0]), X]
        return Xb @ self.theta

model = LinReg().fit(X_train, y_train)
pred = model.predict(X_test)

# ---------- 8. metrics ----------
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error, median_absolute_error)

y_true = y_test
mae   = mean_absolute_error(y_true, pred)
rmse  = np.sqrt(mean_squared_error(y_true, pred))
r2    = r2_score(y_true, pred)
mape  = mean_absolute_percentage_error(y_true, pred) * 100
medae = median_absolute_error(y_true, pred)
dir_acc = (np.sign(pred) == np.sign(y_true)).mean()
pearson = np.corrcoef(pred, y_true)[0, 1]
spear   = pd.Series(pred).corr(pd.Series(y_true), method='spearman')
residual = y_true - pred
pct_within_1_sigma = (np.abs(residual) <= residual.std()).mean()

print("==========  Price-change forecast  ==========")
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

# ---------- 9. leak litmus ----------
np.random.seed(42)
X_shuf = X_test.copy()
np.random.shuffle(X_shuf)
pred_shuf = model.predict(X_shuf)
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

# --- align StochRSI to the same window ---
stoch_values = df["stoch_rsi"].iloc[split : split + len(pct_change)].reset_index(drop=True)
stoch_signals = df["stoch_signal"].iloc[split : split + len(pct_change)].reset_index(drop=True)


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

print("\ndate          idx    pred   pctChg%   macd-sig  stoch-rsi  stoch-sig   model      buy&hold    real-MACD   stochRSI")
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
      f"{pct_change[i]:6.2f}%  {macd_signal[i]:8.2f}  "
      f"{stoch_values[i]:8.3f}  {stoch_signals[i]:8.0f}   "
      f"{capital:8.2f}   {buy_hold:8.2f}   {macd_real:8.2f}   {stoch_real:8.2f}")
    time.sleep(0.01)
