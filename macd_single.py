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

# =====================  SINGLE RUN (NO STOP-LOSS) =============================
LEVERAGE = 3.3
curve    = [10000]
in_pos   = 0
entry_p  = None
entry_d  = None
trades   = []
stp = False
stp_pct = 0.029
days_stp = 0
stp_cnt = 0
stp_cnt_max = 0

for i in range(1, len(df)):
    p_prev = df['close'].iloc[i-1]
    p_now  = df['close'].iloc[i]
    pos_i  = pos.iloc[i]
    exit = False
    
    if stp != True and in_pos != 0 and ((entry_p/df['high'].iloc[i]-1)*in_pos>=stp_pct or (entry_p/df['low'].iloc[i]-1)*in_pos>=stp_pct):
      stp = True
      stp_price=curve[-1] * (1 - stp_pct * LEVERAGE)
      stp_cnt=stp_cnt+1
      if stp_cnt_max<stp_cnt:
              stp_cnt_max=stp_cnt_max
      exit = True
      
    # ----- entry logic --------------------------------------------------------
    if in_pos == 0 and pos_i != 0:
        in_pos  = pos_i
        entry_p = p_now
        entry_d = df['date'].iloc[i]
        stp = False
        print(f"ENTRY {in_pos}  ")


    # ----- exit on opposite cross ---------------------------------------------
    if in_pos != 0 and pos_i == -in_pos:
        ret = (p_now / entry_p - 1) * in_pos * LEVERAGE
        if stp == True:
          trades.append((entry_d, df['date'].iloc[i], -stp_pct*LEVERAGE))
        else:
          trades.append((entry_d, df['date'].iloc[i], ret))
          if ret >= 0:
            stp_cnt=0
          else:
            stp_cnt=stp_cnt+1
            if stp_cnt_max<stp_cnt:
              stp_cnt_max=stp_cnt
          exit = True
          
              
        in_pos = 0
        stp = False
        print(f"CROSS TRADE {trades[-1]}  ")

    # ----- equity update -------------------------------------------------------
    if stp == True:
      curve.append(stp_price)
      days_stp=days_stp+1
      print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}  "
          f"STOP @ {df['close'].iloc[i]:>10.2f}  "
          f"CURVE {curve[-1]}"
          f" EXIT == {exit}")
      
    else:
      if exit: 
        curve.append(curve[-1] * (1 + (p_now/entry_p - 1) * in_pos * LEVERAGE))
      else:
        curve.append(curve[-1])
      print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}  "
          f" {df['close'].iloc[i]:>10.2f}  "
          f"CURVE {curve[-1]}"
          f" EXIT == {exit}")
    time.sleep(0.01)

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

print(f'\n===== MACD (no stop-loss, {LEVERAGE}× lev) =====')
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

# ------------------------------------------------ DCA projection --------------
curve_df = pd.DataFrame({'date': df['date'], 'equity': curve})
curve_df = curve_df.set_index('date')
monthly_eq = curve_df['equity'].resample('M').last()
monthly_ret = monthly_eq.pct_change().dropna()

g_month = monthly_ret.mean()
sigma_m = monthly_ret.std()
g_year  = (1 + g_month)**12 - 1

daily_ret = curve.pct_change().dropna()
monthly_vol_real = daily_ret.std() * np.sqrt(21)   # 21 trading days ≈ 1 month
print(f'\nRealised monthly volatility: {monthly_vol_real*100:.2f} %')



print('\n----- DCA plan (50 € every month) -----')
print(f'Strategy monthly return (mean): {g_month*100:5.2f} %')
print(f'Strategy yearly return (geom):  {g_year*100:5.2f} %')
print(f'Monthly volatility:             {sigma_m*100:5.2f} %')

for yrs in (1, 2, 5):
    n = yrs * 12
    fv = 50 * (1 + g_month) * ((1 + g_month)**n - 1) / g_month if g_month else 50 * n
    print(f'Expected value after {yrs:>2.0f} year(s): {fv:7.0f} €')

# ---------------------------  TRADE-LEVEL COMPOUNDING  -------------------------
trade_dates = [t[1] for t in trades]          # exit dates
trade_rets  = [t[2] for t in trades]          # log returns (already %)

# Compounded equity curve (reset base = 1)
trade_curve = (1 + pd.Series(trade_rets, index=trade_dates)).cumprod() * 10000

# ---------------------------  MONTHLY RETURNS  --------------------------------
monthly = trade_curve.resample('M').last().pct_change().dropna()

print('\n----- Monthly returns (compounded per trade) -----')
print(monthly.tail(12))  # last 12 months

print(f'\nExpected 1-month return (mean): {monthly.mean()*100:.2f}%')
print(f'Monthly volatility:             {monthly.std()*100:.2f}%')
