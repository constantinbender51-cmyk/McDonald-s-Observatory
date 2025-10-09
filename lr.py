import pandas as pd
import numpy as np
from pathlib import Path

# ---------- 1. load ----------
CSV_FILE = Path("btc_daily.csv")
df = pd.read_csv(CSV_FILE, parse_dates=["date"]).sort_values("date")

# ---------- 2. target ----------
df["y"] = df["close"].shift(-1)            # tomorrow's close

# ---------- 3. features ----------
feat = ["open", "high", "low", "volume"]
for c in feat:
    df[c] = df[c].astype(float)
df[feat] = df[feat].shift(1)               # no peeking
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
