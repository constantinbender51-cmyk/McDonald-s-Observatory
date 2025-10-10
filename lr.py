import pandas as pd
import numpy as np
from pathlib import Path
import time

# ----------------------------------------------------------
# 9.  TWO-HORIZON SET-UP  (5-day and 27-day)
# ----------------------------------------------------------
H_short = 5
H_long  = 27

def train_model(horizon):
    """clone of the LinReg pipeline for any horizon"""
    df_ = df.copy()
    df_["y"] = (df_["close"].shift(-horizon) / df_["close"] - 1) * 100
    df_ = df_.dropna(subset=["y"])          # keep aligned

    # same 80-feature matrix we already built
    X_ = zscore_transform(df_[FEATURES].values, mu, sigma)
    y_ = df_["y"].values

    split_ = int(len(df_) * 0.8)
    X_tr, y_tr = X_[:split_], y_[:split_]
    X_te, y_te = X_[split_:], y_[split_:]

    m = LinReg().fit(X_tr, y_tr)
    pred_te = m.predict(X_te)
    return m, pred_te, df_

model5,  pred5,  df5  = train_model(H_short)
model27, pred27, df27 = train_model(H_long)

# ----------------------------------------------------------
# 10.  TRADE LOGIC
# ----------------------------------------------------------
capital      = 1000.0
buy_hold     = 1000.0
lev          = 2.0
stop_pct     = 0.50          # 50 % of |5-day pred|

# --- align everything to the common test window -------------
first_idx = split
last_idx  = len(df) - max(H_short, H_long)   # keep room for both targets

# 1-day % changes for the whole out-of-sample period
pct_1d = (close[first_idx+1 : last_idx+1] /
          close[first_idx : last_idx] - 1)

# slice forecasts to same window
pred5  = pred5 [:len(pct_1d)]
pred27 = pred27[:len(pct_1d)]

position   = 0               # +1 long / -1 short / 0 out
entry_val  = 0.0             # equity at entry
entry_i    = 0
max_dd     = 0.0             # running max dd inside current trade
high_since_entry = 0.0

print("\ndate       idx  5d%  27d%  pos  equity   buy&hold  note")
for i in range(len(pct_1d)):
    p5, p27 = pred5[i], pred27[i]
    new_pos = 0
    note    = ""

    # ----- signal agreement -----
    if   p5 > 0 and p27 > 0: new_pos =  1
    elif p5 < 0 and p27 < 0: new_pos = -1
    else:                    new_pos =  0

    ret = pct_1d[i] * lev          # 1-day leveraged return

    # ----- exit conditions if we are in a trade -----
    if position != 0:
        # 1. stop-loss: price has moved against us by >50 % of |5-day pred|
        stop_thresh = stop_pct * abs(pred5[entry_i])
        realised_pct = (close[first_idx+i] / close[first_idx+entry_i] - 1) * 100 * position
        if realised_pct <= -stop_thresh:
            new_pos, note = 0, "STOP"

        # 2. both forecasts flipped to opposite side
        if position == 1 and p5 < 0 and p27 < 0:
            new_pos, note = 0, "FLIP-S"
        if position == -1 and p5 > 0 and p27 > 0:
            new_pos, note = 0, "FLIP-L"

        # 3. signals no longer agree
        if new_pos == 0 and note == "":
            note = "EXIT"

    # ----- position change -----
    if new_pos != position:
        if position != 0:               # close existing trade
            gross = (1 + (close[first_idx+i] / close[first_idx+entry_i] - 1) * lev * position)
            capital *= gross
            print(f"  --- close trade  dd:{max_dd:5.2%}  multiplier:{gross-1:6.2%}")
            max_dd = 0.0
        if new_pos != 0:                # open new trade
            entry_val  = capital
            entry_i    = i
            high_since_entry = capital

    # ----- update running max draw-down inside trade -----
    if new_pos != 0:
        high_since_entry = max(high_since_entry, capital * (1 + (close[first_idx+i] / close[first_idx+entry_i] - 1) * lev * new_pos))
        dd = (high_since_entry - capital * (1 + (close[first_idx+i] / close[first_idx+entry_i] - 1) * lev * new_pos)) / high_since_entry
        max_dd = max(max_dd, dd)

    position = new_pos

    # ----- buy & hold -----
    buy_hold *= 1 + pct_1d[i]

    # ----- pretty print -----
    print(f"{df['date'].iloc[first_idx+i].strftime('%Y-%m-%d')}  "
          f"{i:3d}  {p5:5.1f}  {p27:5.1f}  "
          f"{position:3d}  {capital:8.2f}  {buy_hold:8.2f}  {note}")

# ----- final open position -----
if position != 0:
    gross = (1 + (close[first_idx+len(pct_1d)-1] / close[first_idx+entry_i] - 1) * lev * position)
    capital *= gross
    print(f"  --- final close  dd:{max_dd:5.2%}  multiplier:{gross-1:6.2%}")

print(f"\nFinal equity (2× lev) : {capital:10.2f}")
print(f"Buy & hold             : {buy_hold:10.2f}")
print(f"Excess return          : {capital - buy_hold:10.2f}")
