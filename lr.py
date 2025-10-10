import pandas as pd
import numpy as np
from pathlib import Path

# ---------- 1. load ----------
CSV_FILE = Path("btc_daily.csv")
df = pd.read_csv(CSV_FILE, parse_dates=["date"]).sort_values("date")

# ---------- 2. create yesterday-only predictors ----------
# ---------- 1. create the columns ----------
today_hl   = ["high", "low"]
yest_ohlcv = ["yest_open", "yest_high", "yest_low", "yest_close", "yest_volume"]

# ---- today’s high/low (already in the file, just ensure float) ----
df[today_hl] = df[today_hl].astype(float)

# ---- yesterday’s OHLCV ----
df[yest_ohlcv] = df[["open", "high", "low", "close", "volume"]].shift(1)

# ---------- 2. build feature list ----------
FEATURES = today_hl + yest_ohlcv   # 7 columns
df["y"] = df["close"]          # target is still today’s close
df.dropna(inplace=True)

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
mae  = np.mean(np.abs(y_test - pred))
rmse = np.sqrt(np.mean((y_test - pred) ** 2))

print("=== Close-price regression (pure NumPy) ===")
print(f"Test MAE : {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print("\nLast 10 predicted vs actual:")
for a, p in zip(y_test[-10:], pred[-10:]):
    print(f"actual {a:8.2f}  pred {p:8.2f}")

today_c  = test_df["yest_close"].values   # 583 elements
pred_dir = np.sign(pred - today_c)        # pred is 583
true_dir = np.sign(y_test - today_c)      # y_test is 583
acc      = (pred_dir == true_dir).mean()
print(f"Direction accuracy: {acc:.3f}")
print(f"Right: {(pred_dir == true_dir).sum()} / {len(pred)}")

mape = np.mean(np.abs(y_test - pred) / y_test) * 100
ss_res = np.sum((y_test - pred) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
mid = (test_df["high"] + test_df["low"]) / 2
mid_mae = np.mean(np.abs(y_test - mid))

print(f"MAPE        : {mape:.2f} %")
print(f"R²          : {r2:.4f}")
print(f"Midpoint MAE: {mid_mae:.2f}")
print(f"Range-norm  : {mae / (test_df['high'] - test_df['low']).mean():.2f}")

# ---------- 8. compare with 4-feature run ----------
old_mae = 604.57          # your 4-feature result from earlier log
old_r2  = 0.9984
print(f"Δ MAE  : {old_mae - mae:+.2f}  ({(old_mae - mae)/old_mae * 100:+.1f} %)")
print(f"Δ R²   : {r2 - old_r2:+.4f}")

print((pred == test_df["yest_close"].values).mean())

test_df["pred"] = pred
test_df["mid"]  = (test_df.high + test_df.low)/2
print((np.abs(test_df.pred - test_df.mid) / test_df.mid).describe())
