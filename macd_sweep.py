#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD + 3.3× lever BTC strategy – full single-run script
Prints every-day equity curve, trade list, performance stats,
and the DCA projection (50 € deposited every month).
"""

import pandas as pd
import numpy as np
import time

# ------------------------------------------------ read data ---------------------
df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

# ---- MACD (proper warm-up) ----------------------------------------------------
ema12 = df['close'].ewm(span=12, min_periods=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, min_periods=26, adjust=False).mean()
macd  = ema12 - ema26
signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()

# 1/-1 on cross
cross = np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
                np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, 0))
pos = pd.Series(cross, index=df.index).replace(0, np.nan).ffill().fillna(0)

# =====================  GRID: LEVERAGE 1–5 × STOP 0.1–8 % =====================
from itertools import product

LEV_GRID   = [round(x*0.1,1) for x in range(10, 51)]      # 1.0 … 5.0
STOP_GRID  = [round(x*0.001,3) for x in range(1, 81)]     # 0.001 … 0.080

results = []          # list of dicts, one per (lev,stop) pair

for LEVERAGE, stp_pct in product(LEV_GRID, STOP_GRID):

    # ----------  fresh state for every simulation  --------------------------
    curve      = [10000]
    in_pos     = 0
    entry_p    = None
    entry_d    = None
    trades     = []
    stp        = False
    days_stp   = 0
    stp_cnt    = 0
    stp_cnt_max= 0
    just_entered= False

    # ----------  identical loop as in your original code  --------------------
    for i in range(1, len(df)):
        p_prev = df['close'].iloc[i-1]
        p_now  = df['close'].iloc[i]
        pos_i  = pos.iloc[i]
        stp_ret = 0

        # ----- stop-loss check (intraday) -----------------------------------
        if stp != True and in_pos != 0:
            r_hi = (p_prev / df['high'].iloc[i] - 1) * in_pos
            r_lo = (p_prev / df['low'].iloc[i]  - 1) * in_pos
            if  r_hi >= stp_pct or r_lo >= stp_pct:
                stp = True
                stp_price = curve[-1] * (1 - stp_pct * LEVERAGE)
                stp_cnt += 1
                stp_cnt_max = max(stp_cnt_max, stp_cnt)
                stp_ret = -stp_pct*LEVERAGE 
        # ----- entry logic ----------------------------------------------------
        if in_pos == 0 and pos_i != 0:
            in_pos       = pos_i
            entry_p      = p_prev
            entry_d      = df['date'].iloc[i]
            just_entered = True
            stp          = False
            r_hi = (p_prev / df['high'].iloc[i] - 1) * in_pos
            r_lo = (p_prev / df['low'].iloc[i]  - 1) * in_pos
            if  r_hi >= stp_pct or r_lo >= stp_pct:
                stp = True
                stp_price = curve[-1] * (1 - stp_pct * LEVERAGE)
                stp_cnt += 1
                stp_cnt_max = max(stp_cnt_max, stp_cnt)
                curve.append(stp_price)
            else: 
                curve.append(curve[-1] * (1 + (p_now/p_prev - 1) * in_pos * LEVERAGE))    
              
            trades.append((entry_d, df['date'].iloc[i],
                stp_ret if stp else daily_ret))
            stp_ret = 0
      continue

        # ----- exit on opposite cross ----------------------------------------
        if in_pos != 0 and pos_i == -in_pos:
            trades.append((entry_d, df['date'].iloc[i],
                stp_ret if stp else daily_ret))
            stp_ret = 0            
            if not stp and daily_ret >= 0:
                stp_cnt = 0
            else:
                stp_cnt += 1
                stp_cnt_max = max(stp_cnt_max, stp_cnt)

            if stp:
                curve.append(stp_price)
                days_stp += 1
            else:
                curve.append(curve[-1] * (1 + (p_now/p_prev - 1) * in_pos * LEVERAGE))
            in_pos = 0
            stp    = False
            continue

        # ----- equity update --------------------------------------------------
        if stp:
            curve.append(stp_price)
            days_stp += 1
        else:
            curve.append(curve[-1] * (1 + (p_now/p_prev - 1) * in_pos * LEVERAGE))
        trades.append((entry_d, df['date'].iloc[i],
            stp_ret if stp else daily_ret))
        stp_ret = 0

    curve = pd.Series(curve, index=df.index)

    # ---------------------------  METRICS  -----------------------------------
    daily_ret = curve.pct_change().dropna()
    trades_ret = pd.Series([t[2] for t in trades])
    n_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25

    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / n_years) - 1
    vol  = daily_ret.std() * np.sqrt(252)
    sharpe = cagr / vol if vol else np.nan
    drawdown = curve / curve.cummax() - 1
    maxdd = drawdown.min()
    calmar = cagr / abs(maxdd) if maxdd else np.nan

    wins   = trades_ret[trades_ret > 0]
    losses = trades_ret[trades_ret < 0]
    win_rate = len(wins) / len(trades_ret) if trades_ret.size else 0
    avg_win  = wins.mean()   if len(wins)   else 0
    avg_loss = losses.mean() if len(losses) else 0
    payoff   = abs(avg_win / avg_loss) if avg_loss else np.nan
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() else np.nan
    expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)

    kelly = expectancy / trades_ret.var() if trades_ret.var() > 0 else np.nan
    time_in_mkt = 1-((1-(pos != 0).mean())*len(df)+days_stp)/len(df)
    tail_ratio = (np.percentile(daily_ret, 95) /
                  abs(np.percentile(daily_ret, 5))) if daily_ret.size else np.nan
    trades_per_year = len(trades) / n_years
    lose_streak = (trades_ret < 0).astype(int)
    max_lose_streak = lose_streak.groupby(
                          lose_streak.diff().ne(0).cumsum()).sum().max()

    # store row
    results.append(dict(
        lev=LEVERAGE,
        stop_pct=stp_pct,
        final=curve.iloc[-1],
        cagr=cagr,
        vol=vol,
        sharpe=sharpe,
        maxdd=maxdd,
        calmar=calmar,
        trades_py=trades_per_year,
        win_rate=win_rate,
        payoff=payoff,
        pf=profit_factor,
        exp=expectancy,
        kelly=kelly,
        time_mkt=time_in_mkt,
        tail=tail_ratio,
        max_streak=max_lose_streak
    ))

# ==========================  PRINT GRID  ======================================
results.sort(key=lambda x: x['calmar'], reverse=True)
print('lev,stop,final,cagr,vol,sharpe,maxdd,calmar,trades_py,win_rate,payoff,pf,exp,kelly,time_mkt,tail,max_streak')
for r in results[:10]:
    print(f"{r['lev']},{r['stop_pct']},{r['final']:.0f},"
          f"{r['cagr']*100:.2f},{r['vol']*100:.2f},{r['sharpe']:.2f},"
          f"{r['maxdd']*100:.2f},{r['calmar']:.2f},{r['trades_py']:.1f},"
          f"{r['win_rate']*100:.1f},{r['payoff']:.2f},{r['pf']:.2f},"
          f"{r['exp']*100:.2f},{r['kelly']*100:.2f},{r['time_mkt']*100:.1f},"
          f"{r['tail']:.2f},{r['max_streak']:.0f}")
