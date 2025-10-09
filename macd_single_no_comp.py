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
# --------------------  Stochastic RSI -----------------------------------------
def stoch_rsi(series, rsi_len=14, stoch_len=14, smooth_k=3, smooth_d=3):
    """Trading-View style Stochastic-RSI: returns (K, D)"""
    delta = series.diff()
    gain  = np.where(delta > 0, delta, 0)
    loss  = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(rsi_len).mean()
    avg_loss = pd.Series(loss).rolling(rsi_len).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    stoch = (rsi - rsi.rolling(stoch_len).min()) / \
            (rsi.rolling(stoch_len).max() - rsi.rolling(stoch_len).min()) * 100
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d

k, d = stoch_rsi(df['close'])          # K = blue, D = orange

# ---- Stoch-RSI position mask -------------------------------------------------
stoch_cross = np.where((k > d) & (k.shift() <= d.shift()) & (k <= 20),  1,
              np.where((k < d) & (k.shift() >= d.shift()) & (k >= 80), -1, 0))
pos_stoch = pd.Series(stoch_cross, index=df.index).replace(0, np.nan).ffill().fillna(0)

# ---- quick overlay: both signals in one frame -------------------------------
signal_df = pd.DataFrame({
    'date'      : df['date'],
    'close'     : df['close'],
    'macd_pos'  : pos,
    'stoch_pos' : pos_stoch
})

print('\n----- first 10 rows: MACD vs Stoch-RSI position -----')
print(signal_df.head(40).to_string(index=False))

# =====================  SINGLE RUN (WITH 2.9 % STOP) ==========================
LEVERAGE = 1
curve      = [10000]
in_pos     = 0
entry_p    = None
entry_d    = None
trades     = []
stp        = False
stp_pct    = 0.99
days_stp   = 0
stp_cnt    = 0
stp_cnt_max= 0
just_entered= False                 # NEW: flag to skip P&L on entry bar

for i in range(1, len(df)):
    p_prev = df['close'].iloc[i-1]
    p_now  = df['close'].iloc[i]
    pos_i  = pos.iloc[i]
    stp_ret = 0

    # ----- stop-loss check (intraday) ---------------------------------------
    if stp != True and in_pos != 0:
        r_hi = (p_prev / df['high'].iloc[i] - 1) * in_pos
        r_lo = (p_prev / df['low'].iloc[i]  - 1) * in_pos
        if  r_hi >= stp_pct or r_lo >= stp_pct:
            stp = True
            stp_price = curve[-1] * (1 - stp_pct * LEVERAGE)
            stp_cnt += 1
            stp_cnt_max = max(stp_cnt_max, stp_cnt)
            stp_ret = -stp_price*LEVERAGE


  
    # ----- entry logic --------------------------------------------------------
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
            curve.append(curve[-1])     
        print(f"{in_pos}"
          f" {df['date'].iloc[i].strftime('%Y-%m-%d')}  "
          f" ENTRY PRICE {entry_p}"
          f" POS {in_pos}"
          f" HIGH {df['high'].iloc[i]}"
          f" LOW {df['low'].iloc[i]}"
          f" CLOSE {df['close'].iloc[i]:>10.2f}  "
          f" STOP {stp}"
          f" CURVE {curve[-1]}"
          f" JUST ENTERED")
        continue                        # skip to next bar
      
  
    # ----- exit on opposite cross ---------------------------------------------
    if in_pos != 0 and pos_i == -in_pos:
        daily_ret = (p_now / entry_p - 1) * in_pos * LEVERAGE
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
        else:                               # normal bar
            curve.append(curve[-1] * (1 + (p_now/entry_p - 1) * in_pos * LEVERAGE))
        print(f"{in_pos}"
          f" {df['date'].iloc[i].strftime('%Y-%m-%d')}  "
          f" ENTRY PRICE {entry_p}"
          f" POS {in_pos}"
          f" HIGH {df['high'].iloc[i]}"
          f" LOW {df['low'].iloc[i]}"
          f" CLOSE {df['close'].iloc[i]:>10.2f}  "
          f" STOP {stp}"
          f" CURVE {curve[-1]}"
          f" CROSSING")
        in_pos = 0
        stp    = False
        
        continue 
    # ----- equity update -------------------------------------------------------
    if stp:
        curve.append(stp_price)
        days_stp += 1
    else:                               # normal bar
        curve.append(curve[-1])

    print(f"{in_pos}"
       f" {df['date'].iloc[i].strftime('%Y-%m-%d')}  "
       f" ENTRY PRICE {entry_p}"
       f" POS {in_pos}"
       f" HIGH {df['high'].iloc[i]}"
       f" LOW {df['low'].iloc[i]}"
       f" CLOSE {df['close'].iloc[i]:>10.2f}  "
       f" STOP {stp}"
       f" CURVE {curve[-1]}")
    time.sleep(0.02)

curve = pd.Series(curve, index=df.index)

# ---------------------------  FULL METRICS  -----------------------------------
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

# ---------------------------  PRINT  ------------------------------------------
final_macd = curve.iloc[-1]
final_hold = (df['close'].iloc[-1] / df['close'].iloc[0]) * 10000
worst      = min(trades, key=lambda x: x[2])

print(f'\n===== MACD (2.9 % stop, {LEVERAGE}× lev) =====')
print(f'MACD final:        €{final_macd:,.0f}')
print(f'Buy & Hold final:  €{final_hold:,.0f}')
print(f'Worst trade:       {worst[2]*100:.2f}% (exit {worst[1].strftime("%Y-%m-%d")})')
print(f'Max drawdown:      {maxdd*100:.2f}%')
time.sleep(0.01)

print('\n----- full performance stats -----')
print(f'CAGR:               {cagr*100:6.2f}%')
print(f'Ann. volatility:    {vol*100:6.2f}%')
print(f'Sharpe (rf=0):      {sharpe:6.2f}')
print(f'Max drawdown:       {maxdd*100:6.2f}%')
print(f'Calmar:             {calmar:6.2f}')
print(f'Trades/year:        {trades_per_year:6.1f}')
print(f'Win-rate:           {win_rate*100:6.1f}%')
print(f'Average win:        {avg_win*100:6.2f}%')
time.sleep(1)
print(f'Average loss:       {avg_loss*100:6.2f}%')
print(f'Payoff ratio:       {payoff:6.2f}')
print(f'Profit factor:      {profit_factor:6.2f}')
print(f'Expectancy/trade:   {expectancy*100:6.2f}%')
print(f'Kelly fraction:     {kelly*100:6.2f}%')
print(f'Time in market:     {time_in_mkt*100:6.1f}%')
print(f'Tail ratio (95/5):  {tail_ratio:6.2f}')
print(f'Max lose streak:    {stp_cnt_max:6.0f}')
time.sleep(1)

# ---------------------  DAY-BY-DAY EQUITY CURVE (first 10 rows) ---------------
print('\n----- equity curve (day-by-day) -----')
print('date       close      equity')
for idx, row in df.head(10).iterrows():
    print(f"{row['date'].strftime('%Y-%m-%d')}  "
          f"{row['close']:>10.2f}  "
          f"{curve[idx]:>10.2f}")
