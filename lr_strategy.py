# --------------------------------------------------
# CONFIGURATION PARAMETERS
# --------------------------------------------------
LOOKBACK = 10           # Number of historical days for features
SHORT_HORIZON = 6       # Days ahead for short-term prediction
LONG_HORIZON = 10       # Days ahead for long-term prediction
SHIFT_VARIABLE = 3      # How many days ahead to use predictions from
THRESHOLD = 1.0         # Prediction threshold (%) for taking positions
LEVERAGE = 3.0          # Position leverage multiplier
STOP_LOSS_PCT = 0.8    # Stop loss as % of predicted move (0.45 = 45%)
INITIAL_CAPITAL = 1000.0

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
# 1.  FEATURES + STANDARDISATION
# --------------------------------------------------
lookback = LOOKBACK
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

# ---------- 2.  TRAIN SHORT & LONG HORIZON MODELS ----------
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

pred_short_raw = train_model(SHORT_HORIZON)
pred_long_raw  = train_model(LONG_HORIZON)

# ----------  3.  SHIFT PREDICTIONS & TRADE  ----------
first = split

# Calculate shifts
short_shift = SHORT_HORIZON - SHIFT_VARIABLE
long_shift = LONG_HORIZON - SHIFT_VARIABLE

print(f"\n{'='*70}")
print(f"STRATEGY CONFIGURATION")
print(f"{'='*70}")
print(f"Short horizon: {SHORT_HORIZON} days | Shift back: {short_shift} days")
print(f"Long horizon:  {LONG_HORIZON} days  | Shift back: {long_shift} days")
print(f"Threshold:     ±{THRESHOLD}%")
print(f"Position logic: Both > threshold → LONG | Both < -threshold → SHORT | Else → HOLD")
print(f"{'='*70}\n")

# ----------------------------------------------------------
# 3.  ALIGN EVERYTHING TO THE SAME OUT-OF-SAMPLE LENGTH
# ----------------------------------------------------------
first = split
n_rows = len(df) - first - 1          # we need t+1 to compute pct1d

# --- truncate raw predictions to identical length ----------
pred_short_raw = pred_short_raw[:n_rows]
pred_long_raw  = pred_long_raw [:n_rows]

# --- build returns vector ----------
close_seg = close[first : first + n_rows + 1]
pct1d = close_seg[1:] / close_seg[:-1] - 1          # length = n_rows - 1

# --- horizon shift for *trading* signal ----------
sh_short = SHORT_HORIZON - SHIFT_VARIABLE
sh_long  = LONG_HORIZON  - SHIFT_VARIABLE

# --- shift backwards, pad with NaN at the end ----------
pred_short = np.full(n_rows, np.nan)
pred_long  = np.full(n_rows, np.nan)

pred_short[:-sh_short] = pred_short_raw[sh_short:]
pred_long [:-sh_long]  = pred_long_raw [sh_long:]

# --- final mask: both forecasts finite ----------
mask = ~(np.isnan(pred_short) | np.isnan(pred_long))
pred_short = pred_short[mask]
pred_long  = pred_long [mask]
pct1d      = pct1d     [mask[:len(pct1d)]]  # pct1d is 1 shorter, slice it

# --- ready to loop ----------
capital = INITIAL_CAPITAL
buyhold = INITIAL_CAPITAL
lev     = LEVERAGE
stop    = STOP_LOSS_PCT
threshold = THRESHOLD

pos = 0
entry_i = 0
entry_pred = 0
max_cap = capital
worst_dd = 0.0

print(f"date        {SHIFT_VARIABLE}dS%  {SHIFT_VARIABLE}dL%  pos  equity   buy&hold")
results = []
for i in range(len(pct1d)):
    ps, pl = pred_short[i], pred_long[i]
    ...

capital = INITIAL_CAPITAL
buyhold = INITIAL_CAPITAL
lev     = LEVERAGE
stop    = STOP_LOSS_PCT
threshold = THRESHOLD

pos     = 0
entry_i = 0
entry_pred = 0
max_cap = capital
worst_dd = 0.0

print(f"date        {SHIFT_VARIABLE}dS%  {SHIFT_VARIABLE}dL%  pos  equity   buy&hold")
results = []
for i in range(len(pct1d)):
    ps, pl = pred_short[i], pred_long[i]
    
    # NEW STRATEGY LOGIC
    new_pos = 0
    if ps > threshold and pl > threshold:
        new_pos = 1   # Both bullish → LONG
    elif ps < -threshold and pl < -threshold:
        new_pos = -1  # Both bearish → SHORT
    # else: new_pos = 0 (HOLD when they disagree)

    # Stop loss check
    if pos != 0:
        realised = (close[first+i] / close[first+entry_i] - 1) * 100 * pos
        if realised <= -stop * abs(entry_pred):
            new_pos = 0

    # Execute trade
    if new_pos != pos:
        if pos != 0:
            gross = 1 + (close[first+i] / close[first+entry_i] - 1) * lev * pos
            capital *= gross
        if new_pos != 0:
            entry_i = i
            entry_pred = ps if new_pos == 1 else -ps  # Store entry prediction
        pos = new_pos

    buyhold *= 1 + pct1d[i]

    if capital > max_cap:
        max_cap = capital
    dd = (capital - max_cap) / max_cap * 100
    if dd < worst_dd:
        worst_dd = dd

    date_str = df['date'].iloc[first+i].strftime('%Y-%m-%d')
    print(f"{date_str}  {ps:5.1f}  {pl:5.1f}  {pos:3d}  {capital:8.2f}  {buyhold:8.2f}")
    results.append([date_str, ps, pl, pos, capital, buyhold])
    time.sleep(0.01)

# Close final position
if pos != 0:
    gross = 1 + (close[first+len(pct1d)-1] / close[first+entry_i] - 1) * lev * pos
    capital *= gross

print(f"\nFinal equity ({LEVERAGE}×) : {capital:8.2f}")
print(f"Buy & hold        : {buyhold:8.2f}")
print(f"Excess            : {capital - buyhold:8.2f}")
print(f"Worst drawdown (%) : {worst_dd:8.2f}")

# --------------------------------------------------
# 4.  ACCURACY METRICS FOR SHIFT_VARIABLE DAY AHEAD
# --------------------------------------------------
d = df.copy()
d["y_shift"] = (d["close"].shift(-SHIFT_VARIABLE) / d["close"] - 1) * 100
d = d.iloc[split:split+min_len]

# Use the shifted predictions to evaluate
acc_short = (np.sign(d["y_shift"].values) == np.sign(pred_short)).mean()
acc_long  = (np.sign(d["y_shift"].values) == np.sign(pred_long)).mean()
mae_short = np.abs(d["y_shift"].values - pred_short).mean()
mae_long  = np.abs(d["y_shift"].values - pred_long).mean()
mse_short = ((d["y_shift"].values - pred_short) ** 2).mean()
mse_long  = ((d["y_shift"].values - pred_long) ** 2).mean()

print("\n" + "="*70)
print(f"PREDICTION QUALITY ({SHIFT_VARIABLE}-day ahead, out-of-sample)")
print("="*70)
print(f"Short model directional accuracy : {acc_short:6.1%}")
print(f"Long model directional accuracy  : {acc_long:6.1%}")
print(f"Short model MAE                  : {mae_short:6.2f}%")
print(f"Long model MAE                   : {mae_long:6.2f}%")
print(f"Short model MSE                  : {mse_short:6.2f}")
print(f"Long model MSE                   : {mse_long:6.2f}")

# --------------------------------------------------
# 5.  WRITE CSV + START WEB SERVER
# --------------------------------------------------
csv_path = Path("results.csv")

results_df = pd.DataFrame(results, 
             columns=["date","pred_short","pred_long","pos","equity","buyhold"])
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved → {csv_path.resolve()}")

# fire up the web-plotter
import subprocess, webbrowser, time, os
port = int(os.environ.get("PORT", 5000))
proc = subprocess.Popen(["python", "webplot.py"], cwd=Path(__file__).parent)
time.sleep(1.5)          # give Flask a moment to bind
webbrowser.open(f"http://127.0.0.1:{port}/")
print("Browser opened – Ctrl-C here to stop the server.")
proc.wait()
