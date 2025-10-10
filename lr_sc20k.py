#!/usr/bin/env python3
import numpy as np
import pandas as pd
from numba import njit, prange
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

CSV_FILE = Path("btc_daily.csv")
df_full  = pd.read_csv(CSV_FILE, parse_dates=["date"]).sort_values("date")
close_full = df_full["close"].values
volume_full = df_full["volume"].values

# ------------------------------------------------------------------
# 1.  feature engineering  –  ONCE
# ------------------------------------------------------------------
def stoch_rsi(close, rsi_p=14, stoch_p=14):
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    roll  = lambda x, w: np.convolve(x, np.ones(w)/w, mode='valid')
    avg_gain = roll(gain, rsi_p)
    avg_loss = roll(loss, rsi_p)
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    # stochastic of rsi
    min_rsi, max_rsi = rolling_minmax(rsi, stoch_p)
    stoch = (rsi[stoch_p-1:] - min_rsi) / (max_rsi - min_rsi + 1e-12)
    # pad to original length
    out = np.empty_like(close, dtype=float)
    out[:rsi_p+stoch_p-2] = np.nan
    out[rsi_p+stoch_p-2:] = stoch
    return out

def ema(arr, n):
    alpha = 2/(n+1)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha*arr[i] + (1-alpha)*out[i-1]
    return out
@njit
def rolling_minmax(x, window):
    n = len(x)
    out_min = np.empty(n - window + 1, dtype=x.dtype)
    out_max = np.empty(n - window + 1, dtype=x.dtype)
    for i in range(n - window + 1):
        w = x[i:i + window]
        out_min[i] = w.min()
        out_max[i] = w.max()
    return out_min, out_max
    

# pre-compute everything
stoch = stoch_rsi(close_full)
pct   = np.empty_like(close_full)
pct[1:] = (close_full[1:]/close_full[:-1] - 1)*100
pct[0]  = 0.0
volpct= np.empty_like(close_full)
volpct[1:] = (volume_full[1:]/volume_full[:-1] - 1)*100
volpct[0]  = 0.0
macd_line = ema(close_full,12) - ema(close_full,26)
macd_sig  = macd_line - ema(macd_line,9)

MAX_LB = 60
N = len(close_full)

# build 4D tensor (T, 4, MAX_LB)  –  we will slice inside the loop
lags = np.full((N, 4, MAX_LB), np.nan, dtype=np.float32)
for i in range(MAX_LB):
    lags[i+1:, 0, i] = stoch[MAX_LB-1-i : -i-1]   # stoch
    lags[i+1:, 1, i] = pct[MAX_LB-1-i : -i-1]     # pct
    lags[i+1:, 2, i] = volpct[MAX_LB-1-i: -i-1]   # vol
    lags[i+1:, 3, i] = macd_sig[MAX_LB-1-i:-i-1]  # macd

# ------------------------------------------------------------------
# 2.  jit-compiled back-test  –  ONE parameter vector
# ------------------------------------------------------------------
@njit(fastmath=True, nogil=True)
def backtest(close, pred5, pred27, lev, stop_pct, cost):
    capital = 1000.0
    pos, entry_i = 0, 0
    stop = stop_pct / 100.0
    equity = [capital]
    for i in range(len(close)-1):
        p5, p27 = pred5[i], pred27[i]
        new_pos = 0
        if p5 > 0 and p27 > 0:   new_pos =  1
        elif p5 < 0 and p27 < 0: new_pos = -1

        if pos != 0:
            realised = (close[i] / close[entry_i] - 1)*100*pos
            if realised <= -stop*abs(pred5[entry_i]):
                new_pos = 0
            if pos==1  and p5<0 and p27<0: new_pos=0
            if pos==-1 and p5>0 and p27>0: new_pos=0

        if new_pos != pos:
            if pos != 0:
                gross = 1 + (close[i]/close[entry_i] - 1)*lev*pos
                capital *= gross*(1 - cost)
            if new_pos != 0:
                entry_i = i
            pos = new_pos
        equity.append(capital)

    if pos != 0:
        gross = 1 + (close[-1]/close[entry_i] - 1)*lev*pos
        capital *= gross*(1 - cost)
        equity[-1] = capital

    eq = np.array(equity)
    running_max = np.maximum.accumulate(eq)
    max_dd = ((eq - running_max)/(running_max + 1e-12)).min()
    return capital, max_dd

# ------------------------------------------------------------------
# 3.  single parameter vector  –  wrapper
# ------------------------------------------------------------------
def run_once(lb, sh, lo, st, lv):
    if sh >= lo:
        return None
    # slice lags
    X = lags[MAX_LB:, :, :lb].reshape(-1, 4*lb)  # (T, 4*lb)
    y5  = (close_full[MAX_LB+sh:]/close_full[MAX_LB:-sh] - 1)*100
    y27 = (close_full[MAX_LB+lo:]/close_full[MAX_LB:-lo] - 1)*100

    # align lengths
    min_len = min(len(X), len(y5), len(y27))
    X, y5, y27 = X[:min_len], y5[:min_len], y27[:min_len]
    split = int(0.8*min_len)

    # z-score on train
    mu = X[:split].mean(axis=0)
    sg = X[:split].std(axis=0) + 1e-12
    X = (X - mu) / sg

    # linear regression  (Xb = add constant)
    Xb = np.c_[np.ones(split), X[:split]]
    theta5  = np.linalg.lstsq(Xb, y5[:split],  rcond=None)[0]
    theta27 = np.linalg.lstsq(Xb, y27[:split], rcond=None)[0]

    Xb_test = np.c_[np.ones(min_len-split), X[split:]]
    pred5  = Xb_test @ theta5
    pred27 = Xb_test @ theta27

    # back-test
    close_seg = close_full[MAX_LB+split : MAX_LB+min_len]
    final, dd = backtest(close_seg, pred5, pred27, lv, st, 0.0015)
    buyhold = 1000.0 * close_seg[-1] / close_seg[0]
    return final, (lb,sh,lo,st,lv), buyhold, dd

# ------------------------------------------------------------------
# 4.  grid + multiprocessing
# ------------------------------------------------------------------
lookbacks = range(5, 61, 5)
shorts    = range(1, 20, 2)
longs     = range(5, 50, 5)
stops     = range(15, 61, 15)
levs      = np.arange(1, 6, 1)

param_grid = list(np.array(t) for t in
                  np.meshgrid(lookbacks, shorts, longs, stops, levs))
param_grid = np.vstack([p.ravel() for p in param_grid]).T.astype(int)

best = []
with mp.Pool(processes=mp.cpu_count()) as pool:
    for res in tqdm(pool.imap_unordered(run_once, param_grid, chunksize=100),
                    total=len(param_grid)):
        if res is None: continue
        best.append(res)
        best = sorted(best, key=lambda x: x[0], reverse=True)[:5]

# ------------------------------------------------------------------
# 5.  pretty print
# ------------------------------------------------------------------
print("\nTop-5 (final equity):")
print("lb | sh | lo | stop | lev || final_eq | buy&hold | max_dd%")
for fe, p, bh, dd in best:
    lb,sh,lo,st,lv = p
    print(f"{lb:2d} | {sh:2d} | {lo:2d} | {st:4d} | {lv:3.1f} || "
          f"{fe:8.2f} | {bh:8.2f} | {dd*100:7.2f}")
