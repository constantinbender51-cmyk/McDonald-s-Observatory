# --------------------------------------------------
# 0.  FETCH FULL BTC DAILY HISTORY FROM BINANCE
# --------------------------------------------------
import requests, math, time, pandas as pd, numpy as np
from pathlib import Path

BINANCE_CSV = Path("btc_daily_bin.csv")

def fetch_binance_daily(symbol="BTCUSDT"):
    """Return a DataFrame with columns ['date', 'close', 'volume'] since 2017-08-17."""
    if BINANCE_CSV.exists():
        print("Loading cached", BINANCE_CSV)
        return pd.read_csv(BINANCE_CSV, parse_dates=["date"])

    print("Downloading full daily history from Binance …")
    root = "https://api.binance.com/api/v3/klines"
    limit = 1000                       # max per call
    interval = "1d"
    start_time = 1502928000000         # 2017-08-17 00:00 UTC
    end_time   = int(time.time()*1000)
    data = []

    while start_time < end_time:
        params = dict(symbol=symbol, interval=interval,
                      startTime=start_time, limit=limit)
        r = requests.get(root, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        data.extend(batch)
        start_time = batch[-1][6] + 1   # next call starts after last close time
        time.sleep(0.2)                 # be polite

    df = (pd.DataFrame(data,
                       columns="open_time o h l c v close_time qav n taker_base_qav taker_quote_qav ignore".split())
            .loc[:, ["close_time", "c", "v"]]
            .rename(columns={"close_time": "date", "c": "close", "v": "volume"})
            .assign(date=lambda x: pd.to_datetime(x["date"], unit="ms"),
                    close=lambda x: x["close"].astype(float),
                    volume=lambda x: x["volume"].astype(float))
            .sort_values("date")
            .reset_index(drop=True))
    df.to_csv(BINANCE_CSV, index=False)
    print("Saved", len(df), "rows to", BINANCE_CSV)
    return df

df = fetch_binance_daily()
close = df["close"].values
# --------------------------------------------------
# 1.  FEATURES + STANDARDISATION  (your code below)
# --------------------------------------------------
lookback = 10
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

pred6  = train_model(6)
pred10 = train_model(10)

# ----------  3.  TRADE  ----------
first = split
min_len = min(len(pred6), len(pred10))
pct1d = (close[first+1 : first+min_len+1] / close[first : first+min_len] - 1)

pred6  = pred6 [:min_len]
pred10 = pred10[:min_len]

capital = 1000.0
buyhold = 1000.0
lev     = 3.0
stop    = 0.8

pos     = 0
entry_i = 0
max_cap = capital
worst_dd = 0.0

print("date        6d%  10d%  pos  equity   buy&hold")
results = []  # <-- NEW: collect rows for CSV
for i in range(len(pct1d)):
    p6, p10 = pred6[i], pred10[i]
    new_pos = 0
    if   p6 > 0 and p10 > 0: new_pos =  1
    elif p6 < 0 and p10 < 0: new_pos = -1

    if pos != 0:
        realised = (close[first+i] / close[first+entry_i] - 1) * 100 * pos
        if realised <= -stop * abs(pred6[entry_i]):
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

    if capital > max_cap:
        max_cap = capital
    dd = (capital - max_cap) / max_cap * 100
    if dd < worst_dd:
        worst_dd = dd

    date_str = df['date'].iloc[first+i].strftime('%Y-%m-%d')
    print(f"{date_str}  {p6:5.1f}  {p10:5.1f}  {pos:3d}  {capital:8.2f}  {buyhold:8.2f}")
    results.append([date_str, p6, p10, pos, capital, buyhold])
    time.sleep(0.01)

if pos != 0:
    gross = 1 + (close[first+len(pct1d)-1] / close[first+entry_i] - 1) * lev * pos
    capital *= gross

print(f"\nFinal equity (3×) : {capital:8.2f}")
print(f"Buy & hold        : {buyhold:8.2f}")
print(f"Excess            : {capital - buyhold:8.2f}")
print(f"Worst trade (%)   : {worst_dd:8.2f}")

# --------------------------------------------------
# 4.  ACCURACY & ERROR METRICS
# --------------------------------------------------
d = df.copy()
d["y6"]  = (d["close"].shift(-6)  / d["close"] - 1) * 100
d["y10"] = (d["close"].shift(-10) / d["close"] - 1) * 100
d = d.iloc[split:split+min_len]

acc6  = (np.sign(d["y6"].values)  == np.sign(pred6)).mean()
acc10 = (np.sign(d["y10"].values) == np.sign(pred10)).mean()
mae6  = np.abs(d["y6"].values  - pred6).mean()
mae10 = np.abs(d["y10"].values - pred10).mean()
mse6  = ((d["y6"].values  - pred6) ** 2).mean()
mse10 = ((d["y10"].values - pred10) ** 2).mean()

print("\nPrediction quality (out-of-sample)")
print(f"6-day  directional accuracy : {acc6:6.1%}")
print(f"10-day directional accuracy : {acc10:6.1%}")
print(f"6-day  MAE                  : {mae6:6.2f}%")
print(f"10-day MAE                  : {mae10:6.2f}%")
print(f"6-day  MSE                  : {mse6:6.2f}")
print(f"10-day MSE                  : {mse10:6.2f}")

next_day_ret = (close[first+1:first+min_len+1] / close[first:first+min_len] - 1) * 100
dir6_vs_1d  = (np.sign(pred6)  == np.sign(next_day_ret)).mean()
dir10_vs_1d = (np.sign(pred10) == np.sign(next_day_ret)).mean()

print("\nUsing long-horizon forecast to guess T+1 direction")
print(f"pred6  → next-day accuracy : {dir6_vs_1d:6.1%}")
print(f"pred10 → next-day accuracy : {dir10_vs_1d:6.1%}")

# --------------------------------------------------
# 5.  WRITE CSV + START WEB SERVER
# --------------------------------------------------
csv_path = Path("results.csv")
pd.DataFrame(results, columns=["date","pred6","pred10","pos","equity","buyhold"]).to_csv(csv_path, index=False)
print(f"\nResults saved → {csv_path.resolve()}")

# fire up the web-plotter
import subprocess, webbrowser, time, os
port = int(os.environ.get("PORT", 5000))
proc = subprocess.Popen(["python", "webplot.py"], cwd=Path(__file__).parent)
time.sleep(1.5)          # give Flask a moment to bind
webbrowser.open(f"http://127.0.0.1:{port}/")
print("Browser opened – Ctrl-C here to stop the server.")
proc.wait()
