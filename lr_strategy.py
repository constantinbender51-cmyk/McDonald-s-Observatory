import pandas as pd
import numpy as np
from pathlib import Path
import time

CSV_FILE = Path("btc_daily.csv")
df = pd.read_csv(CSV_FILE, parse_dates=["date"]).sort_values("date")
close = df["close"].values

# ---------- 1.  FEATURES + STANDARDISATION  ----------
lookback = 10                                      # << changed
stoch_cols = [f"stoch_{i}" for i in range(lookback)]
pct_cols   = [f"pct_{i}"   for i in range(lookback)]
vol_cols   = [f"vol_{i}"   for i in range(lookback)]
macd_cols  = [f"macd_{i}"  for i in range(lookback)]

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
macd_line = ema(df["close"], 12) - ema(df["close"], 26)
signal_line = ema(macd_line, 9)
df["macd_signal"] = macd_line - signal_line

for i in range(lookback):
    df[stoch_cols[i]] = df["stoch_rsi"].shift(lookback - i)
    df[pct_cols[i]]   = df["pct_chg"].shift(lookback - i)
    df[vol_cols[i]]   = df["vol_pct_chg"].shift(lookback - i)
    df[macd_cols[i]]  = df["macd_signal"].shift(lookback - i)

FEATURES = stoch_cols + pct_cols + vol_cols + macd_cols
df = df.dropna()

split = int(len(df) * 0.8)
mu, sigma = df[FEATURES].values[:split].mean(axis=0), df[FEATURES].values[:split].std(axis=0)

def zscore(X): return (X - mu) / np.where(sigma == 0, 1, sigma)

class LinReg:
    def fit(self, X, y):
        Xb = np.c_[np.ones(X.shape[0]), X]
        self.theta = np.linalg.lstsq(Xb, y, rcond=None)[0]
        return self
    def predict(self, X):
        Xb = np.c_[np.ones(X.shape[0]), X]
        return Xb @ self.theta

# ---------- 2.  TRAIN 6-day & 10-day MODELS ----------
def train_model(horizon):
    d = df.copy()
    d["y"] = (d["close"].shift(-horizon) / d["close"] - 1) * 100
    d = d.dropna(subset=["y"])
    X = zscore(d[FEATURES].values)
    y = d["y"].values
    s = int(len(d) * 0.8)
    model = LinReg().fit(X[:s], y[:s])
    pred  = model.predict(X[s:])
    return pred

pred6  = train_model(6)    # << changed
pred10 = train_model(10)   # << changed

# ----------  3.  TRADE  ----------
first = split
min_len = min(len(pred6), len(pred10))             # << changed
pct1d = (close[first+1 : first+min_len+1] / close[first : first+min_len] - 1)

pred6  = pred6 [:min_len]                          # << changed
pred10 = pred10[:min_len]                          # << changed

capital = 1000.0
buyhold = 1000.0
lev     = 3.0                                       # << changed
stop    = 0.80                                      # << changed

pos     = 0
entry_i = 0

max_cap = capital
worst_dd = 0.0

print("date        6d%  10d%  pos  equity   buy&hold")

for i in range(len(pct1d)):
    p6, p10 = pred6[i], pred10[i]                   # << changed
    new_pos = 0
    if   p6 > 0 and p10 > 0: new_pos =  1
    elif p6 < 0 and p10 < 0: new_pos = -1

    if pos != 0:
        realised = (close[first+i] / close[first+entry_i] - 1) * 100 * pos
        if realised <= -stop * abs(pred6[entry_i]):   # << changed
            new_pos = 0
        if pos ==  1 and p6 < 0 and p10 < 0: new_pos = 0
        if pos == -1 and p6 > 0 and p10 > 0: new_pos = 0

    if new_pos != pos:
        if pos != 0:
            gross = 1 + (close[first+i] / close[first+entry_i] - 1) * lev * pos
            capital *= gross
        if new_pos != 0:
            entry_i = i
        pos = new_pos

    buyhold *= 1 + pct1d[i]

    # --- track worst single-day equity drop ---
    if capital > max_cap:
        max_cap = capital
    dd = (capital - max_cap) / max_cap * 100
    if dd < worst_dd:
        worst_dd = dd

    print(f"{df['date'].iloc[first+i].strftime('%Y-%m-%d')}  "
          f"{p6:5.1f}  {p10:5.1f}  {pos:3d}  {capital:8.2f}  {buyhold:8.2f}")
    time.sleep(0.01)

# final close-out
if pos != 0:
    gross = 1 + (close[first+len(pct1d)-1] / close[first+entry_i] - 1) * lev * pos
    capital *= gross

print(f"\nFinal equity (3×) : {capital:8.2f}")
print(f"Buy & hold        : {buyhold:8.2f}")
print(f"Excess            : {capital - buyhold:8.2f}")
print(f"Worst trade (%)   : {worst_dd:8.2f}")
