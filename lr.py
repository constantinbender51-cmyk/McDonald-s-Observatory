import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ---------- 1. load ----------
CSV_FILE = Path("btc_daily.csv")
df = pd.read_csv(CSV_FILE, parse_dates=["date"]).sort_values("date")

# ---------- 2. target ----------
# tomorrow's close
df["y"] = df["close"].shift(-1)

# ---------- 3. features ----------
# today's OHLCV
feat = ["open", "high", "low", "volume"]
for c in feat:
    df[c] = df[c].astype(float)
df[feat] = df[feat].shift(1)          # no peeking
df.dropna(inplace=True)

# ---------- 4. split ----------
split = int(len(df) * 0.8)
train_df = df.iloc[:split]
test_df  = df.iloc[split:]

# ---------- 5. standardise ----------
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feat])
y_train = train_df["y"].values
X_test  = scaler.transform(test_df[feat])
y_test  = test_df["y"].values

# ---------- 6. linear regression from scratch ----------
class LinReg:
    def fit(self, X, y):
        Xb = np.c_[np.ones(X.shape[0]), X]          # add bias
        self.theta = np.linalg.lstsq(Xb, y, rcond=None)[0]
        return self
    def predict(self, X):
        Xb = np.c_[np.ones(X.shape[0]), X]
        return Xb @ self.theta

model = LinReg().fit(X_train, y_train)
pred = model.predict(X_test)

# ---------- 7. console metrics ----------
mae  = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)
print("=== Close-price regression ===")
print(f"Test MAE : {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print("\nLast 10 predicted vs actual:")
out = pd.DataFrame({"actual": y_test[-10:], "pred": pred[-10:]})
print(out.to_string(index=False, float_format="%.2f"))
