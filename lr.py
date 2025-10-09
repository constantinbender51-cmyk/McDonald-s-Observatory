import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# ---------- 1. load ----------
CSV_FILE = Path("btc_daily.csv")   # <-- same CSV you already have
df = pd.read_csv(CSV_FILE, parse_dates=["date"]).sort_values("date")
close = df["close"].astype(float)

# ---------- 2. MACD ----------
def macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast).mean()
    ema_s = series.ewm(span=slow).mean()
    macd_line = ema_f - ema_s
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line

macd_line, signal_line = macd(close)

# ---------- 3. label ----------
df["y"] = (close.shift(-1) > close).astype(int)

# ---------- 4. features ----------
df["macd"] = macd_line
df["signal"] = signal_line
df["macd_x"] = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int)  # cross-up
df["macd_o"] = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))).astype(int)  # cross-down
FEATURES = ["macd_x", "macd_o"]
df[FEATURES] = df[FEATURES].shift(1)   # no peeking
df.dropna(inplace=True)

# ---------- 5. split ----------
split = int(len(df) * 0.8)
train_df = df.iloc[:split]
test_df  = df.iloc[split:]

X_train, y_train = train_df[FEATURES].values, train_df["y"].values
X_test,  y_test  = test_df[FEATURES].values,  test_df["y"].values

# ---------- 6. logistic regression (no external libs) ----------
class LogisticRegression:
    def __init__(self, lr=0.3, n_iter=15_000, tol=1e-6):
        self.lr, self.n_iter, self.tol = lr, n_iter, tol
    def fit(self, X, y):
        Xb = np.c_[np.ones(X.shape[0]), X]
        th = np.zeros(Xb.shape[1])
        for i in range(self.n_iter):
            p = 1 / (1 + np.exp(-np.clip(Xb @ th, -500, 500)))
            g = (Xb.T @ (p - y)) / y.size
            th_new = th - self.lr * g
            if np.linalg.norm(th_new - th, ord=1) < self.tol:
                break
            th = th_new
        self._th = th
        return self
    def predict_proba(self, X):
        Xb = np.c_[np.ones(X.shape[0]), X]
        return 1 / (1 + np.exp(-np.clip(Xb @ self._th, -500, 500)))
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

model = LogisticRegression().fit(X_train, y_train)
pred = model.predict(X_test)

# ---------- 7. console output ----------
print("=== MACD crossover logistic-regression results ===")
print("Confusion matrix (test set):")
print(confusion_matrix(y_test, pred))
print("\nClassification report:")
print(classification_report(y_test, pred, digits=3))

# optional: show last few predictions vs actual
print("\nLast 10 predictions vs actual:")
out = pd.DataFrame({"actual": y_test[-10:], "pred": pred[-10:]})
print(out.to_string(index=False))
