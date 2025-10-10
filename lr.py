import pandas as pd
import numpy as np
from pathlib import Path
import time

# ------------------------------------------------------------------
# 0. 1-to-40-day sweep
# ------------------------------------------------------------------
results = []                       # (horizon, final_capital)

for FORECAST_HORIZON in range(1, 41):
    # ---------- 1. load ----------
    CSV_FILE = Path("btc_daily.csv")
    df = pd.read_csv(CSV_FILE, parse_dates=["date"]).sort_values("date")

    # ---------- 2. forward-horizon target ----------
    close_series = df["close"]
    df["y"] = (close_series.shift(-FORECAST_HORIZON) / close_series - 1) * 100

    # ---------- 3. raw series ----------
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
    df["pct_chg"]   = df["close"].pct_change() * 100
    df["vol_pct_chg"] = df["volume"].pct_change() * 100

    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    macd_line   = ema(df["close"], 12) - ema(df["close"], 26)
    signal_line = ema(macd_line, 9)
    df["macd_signal"] = macd_line - signal_line

    # ---------- 4. 20-day look-back ----------
    lookback = 20
    stoch_cols = [f"stoch_{i}" for i in range(lookback)]
    pct_cols   = [f"pct_{i}"   for i in range(lookback)]
    vol_cols   = [f"vol_{i}"   for i in range(lookback)]
    macd_cols  = [f"macd_{i}"  for i in range(lookback)]

    for i in range(lookback):
        df[stoch_cols[i]] = df["stoch_rsi"].shift(lookback - i)
        df[pct_cols[i]]   = df["pct_chg"].shift(lookback - i)
        df[vol_cols[i]]   = df["vol_pct_chg"].shift(lookback - i)
        df[macd_cols[i]]  = df["macd_signal"].shift(lookback - i)

    FEATURES = stoch_cols + pct_cols + vol_cols + macd_cols
    df = df.dropna()

    # ---------- 5. train/test split ----------
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    y_train, y_test   = train_df["y"].values, test_df["y"].values

    # ---------- 6. standardise ----------
    def zscore_fit(X): return X.mean(axis=0), X.std(axis=0, ddof=0)
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
    pred  = model.predict(X_test)

    # ---------- 8. strategy back-test (1-day position flip) ----------
    close = df["close"].values
    first_test_idx = split
    last_test_idx  = len(close) - 1 - FORECAST_HORIZON
    pct_change = (close[first_test_idx+1 : last_test_idx+1] /
                  close[first_test_idx : last_test_idx] - 1) * 100

    pred        = pred[:len(pct_change)]
    test_dates  = df["date"].iloc[split : split + len(pct_change)].reset_index(drop=True)

    capital, position, entry_i = 1000.0, 0, 0
    for i in range(len(pred)):
        new_pos  = int(np.sign(pred[i]))
        if new_pos != position:
            gross = (1 + position * pct_change[entry_i:i+1]/100).prod()
            capital *= gross
            position, entry_i = new_pos, i+1
    if position != 0:
        capital *= (1 + position * pct_change[entry_i:]/100).prod()

    results.append((FORECAST_HORIZON, capital))

# ------------------------------------------------------------------
# 9. show 5 best horizons
# ------------------------------------------------------------------
print("\nTop-5 horizons by final strategy capital:")
for h, c in sorted(results, key=lambda x: x[1], reverse=True)[:5]:
    print(f"Forecast horizon {h:2d} days → {c:8.2f}")
