import pandas as pd
import numpy as np
from pathlib import Path

# ---------- 1. load ----------
CSV_FILE = Path("btc_daily.csv")
df = pd.read_csv(CSV_FILE, parse_dates=["date"]).sort_values("date")

# ----- today’s OHLCV -----
today = ["open", "high", "low", "volume"]
for c in today:
    df[c] = df[c].astype(float)

# ----- yesterday’s OHLCV -----
yest = ["yest_open", "yest_high", "yest_low", "yest_volume"]
df[yest] = df[today].shift(1)          # lag = 1
FEATURES = today + yest                # 8 columns

# ----- target (same-day close) -----
df["y"] = df["close"]

# drop rows with NaNs introduced by lag
df.dropna(inplace=True)

# ---------- 4. train/test split ----------
split = int(len(df) * 0.8)
train_df = df.iloc[:split]
test_df  = df.iloc[split:]

# ---------- 5. standardise (from scratch) ----------
def zscore_fit(X):
    return X.mean(axis=0), X.std(axis=0, ddof=0)

def zscore_transform(X, mu, sigma):
    return (X - mu) / np.where(sigma == 0, 1, sigma)

mu, sigma = zscore_fit(train_df[feat].values)
X_train = zscore_transform(train_df[feat].values, mu, sigma)
y_train = train_df["y"].values
X_test  = zscore_transform(test_df[feat].values, mu, sigma)
y_test  = test_df["y"].values

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

today_c   = test_df["close"].values
dir_right = np.mean((np.sign(pred - today_c) == np.sign(y_test - today_c)))
print(f"Direction accuracy: {dir_right:.3f}")

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


