# grid_search_btc.py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm               # pip install tqdm

CSV_FILE = Path("btc_daily.csv")
df_full = pd.read_csv(CSV_FILE, parse_dates=["date"]).sort_values("date")
close_full = df_full["close"].values

# ---------- unchanged helper code ----------
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

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

class LinReg:
    def fit(self, X, y):
        Xb = np.c_[np.ones(X.shape[0]), X]
        self.theta = np.linalg.lstsq(Xb, y, rcond=None)[0]
        return self
    def predict(self, X):
        Xb = np.c_[np.ones(X.shape[0]), X]
        return Xb @ self.theta
# -------------------------------------------

def run_once(lookback, short_h, long_h, stop_pct, lev):
    """
    Returns (final_equity, buyhold, max_drawdown)
    Everything else is identical to your original logic.
    """
    # 1. FEATURE ENGINEERING
    df = df_full.copy()
    df["stoch_rsi"] = stoch_rsi(df["close"])
    df["pct_chg"]   = df["close"].pct_change() * 100
    df["vol_pct_chg"] = df["volume"].pct_change() * 100
    macd_line = ema(df["close"], 12) - ema(df["close"], 26)
    signal_line = ema(macd_line, 9)
    df["macd_signal"] = macd_line - signal_line

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
    split = int(len(df) * 0.8)
    mu, sigma = df[FEATURES].values[:split].mean(axis=0), df[FEATURES].values[:split].std(axis=0)
    def zscore(X): return (X - mu) / np.where(sigma == 0, 1, sigma)

    # 2. TRAIN
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

    pred5  = train_model(short_h)
    pred27 = train_model(long_h)

    # 3. TRADE
    first = split
    min_len = min(len(pred5), len(pred27))
    pct1d = (close_full[first+1 : first+min_len+1] / close_full[first : first+min_len] - 1)

    pred5  = pred5 [:min_len]
    pred27 = pred27[:min_len]

    capital = 1000.0
    buyhold = 1000.0
    stop    = stop_pct / 100.0
    pos = 0
    entry_i = 0
    equity_curve = [capital]

    for i in range(len(pct1d)):
        p5, p27 = pred5[i], pred27[i]
        new_pos = 0
        if   p5 > 0 and p27 > 0: new_pos =  1
        elif p5 < 0 and p27 < 0: new_pos = -1

        if pos != 0:
            realised = (close_full[first+i] / close_full[first+entry_i] - 1) * 100 * pos
            if realised <= -stop * abs(pred5[entry_i]):
                new_pos = 0
            if pos ==  1 and p5 < 0 and p27 < 0: new_pos = 0
            if pos == -1 and p5 > 0 and p27 > 0: new_pos = 0

        if new_pos != pos:
            if pos != 0:
                gross = 1 + (close_full[first+i] / close_full[first+entry_i] - 1) * lev * pos
                capital *= gross
            if new_pos != 0:
                entry_i = i
            pos = new_pos

        buyhold *= 1 + pct1d[i]
        equity_curve.append(capital)

    if pos != 0:
        gross = 1 + (close_full[first+len(pct1d)-1] / close_full[first+entry_i] - 1) * lev * pos
        capital *= gross

    # max drawdown
    equity_curve = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()

    return capital, buyhold, max_dd

# ---------- GRID ----------
lookbacks = range(5, 105, 5)
shorts    = range(1, 21)
longs     = range(5, 101)
stops     = range(10, 205, 5)
levs      = np.arange(1, 5.5, 0.5)

total = len(lookbacks)*len(shorts)*len(longs)*len(stops)*len(levs)
best = []   # keep 5 best (final_equity, params, buyhold, max_dd)

for lb in tqdm(lookbacks, desc="lookback"):
    for sh in shorts:
        for lo in longs:
            if sh >= lo:              # meaningless
                continue
            for st in stops:
                for lv in levs:
                    fe, bh, dd = run_once(lb, sh, lo, st, lv)
                    best.append((fe, (lb, sh, lo, st, lv), bh, dd))
                    best = sorted(best, key=lambda x: x[0], reverse=True)[:5]

# ---------- FINAL SUMMARY ----------
print("\nTop-5 parameter sets (highest final equity):")
print("lookback | short | long | stop% | lev || final_eq | buy&hold | max_dd%")
for fe, p, bh, dd in best:
    lb, sh, lo, st, lv = p
    print(f"{lb:7d}  | {sh:4d}  | {lo:3d} | {st:4d} | {lv:4.1f} || {fe:8.2f} | {bh:8.2f} | {dd*100:7.2f}")
