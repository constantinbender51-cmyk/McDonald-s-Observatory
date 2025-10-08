import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# ------------------------------------------------ read data ---------------------
df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

# ---- MACD (proper warm-up) ----------------------------------------------------
ema12 = df['close'].ewm(span=12, min_periods=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, min_periods=26, adjust=False).mean()
macd  = ema12 - ema26
signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()

cross = np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
                np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, 0))
pos = pd.Series(cross, index=df.index).replace(0, np.nan).ffill().fillna(0)

# =====================  GRIDS ================================================
lev_grid = np.round(np.arange(1.0, 5.1, 0.1), 2)        # 1.0 … 5.0
stp_grid = np.round(np.arange(0.001, 0.081, 0.002), 3)  # 0.001 … 0.080 step 0.002

summary = []   # list-of-dicts

for LEVERAGE in tqdm(lev_grid, desc='leverage'):
  for STP_PCT in tqdm(stp_grid, desc='stop %', leave=False):

    # ----------  identical inner loop to your original -----------------------
    curve    = [10000]
    in_pos   = 0
    entry_p  = None
    entry_d  = None
    trades   = []
    stp      = False
    days_stp = 0
    stp_cnt  = 0
    stp_cnt_max = 0

    for i in range(1, len(df)):
        p_prev = df['close'].iloc[i-1]
        p_now  = df['close'].iloc[i]
        pos_i  = pos.iloc[i]

        # ----- stop-loss check -----
        if (not stp) and in_pos != 0:
            hh = df['high'].iloc[i]
            ll = df['low'].iloc[i]
            if (entry_p/hh - 1)*in_pos >= STP_PCT or (entry_p/ll - 1)*in_pos >= STP_PCT:
                stp = True
                stp_price = curve[-1] * (1 - STP_PCT * LEVERAGE)
                stp_cnt += 1
                stp_cnt_max = max(stp_cnt_max, stp_cnt)

        # ----- entry -----
        if in_pos == 0 and pos_i != 0:
            in_pos  = pos_i
            entry_p = p_now
            entry_d = df['date'].iloc[i]
            stp     = False

        # ----- exit on opposite cross -----
        if in_pos != 0 and pos_i == -in_pos:
            ret = (p_now / entry_p - 1) * in_pos * LEVERAGE
            if stp:
                trades.append((entry_d, df['date'].iloc[i], -STP_PCT*LEVERAGE))
            else:
                trades.append((entry_d, df['date'].iloc[i], ret))
                stp_cnt = 0 if ret >= 0 else stp_cnt + 1
                stp_cnt_max = max(stp_cnt_max, stp_cnt)
            in_pos = 0
            stp    = False

        # ----- equity update -----
        if stp:
            curve.append(stp_price)
            days_stp += 1
        else:
            curve.append(curve[-1] * (1 + (p_now/p_prev - 1) * in_pos * LEVERAGE))

    # ---------------------------  METRICS  -----------------------------------
    curve = pd.Series(curve, index=df.index)
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
    time_in_mkt = 1 - ((1 - (pos != 0).mean()) * len(df) + days_stp) / len(df)
    tail_ratio = (np.percentile(daily_ret, 95) /
                  abs(np.percentile(daily_ret, 5))) if daily_ret.size else np.nan
    trades_per_year = len(trades) / n_years

    summary.append({
        'lev': LEVERAGE,
        'stp_pct': STP_PCT,
        'final': curve.iloc[-1],
        'cagr': cagr,
        'vol': vol,
        'sharpe': sharpe,
        'maxdd': maxdd,
        'calmar': calmar,
        'win_rate': win_rate,
        'trades_per_year': trades_per_year,
        'expectancy': expectancy,
        'profit_factor': profit_factor,
        'kelly': kelly,
        'time_in_mkt': time_in_mkt,
        'tail_ratio': tail_ratio,
        'max_lose_streak': stp_cnt_max
    })

# ---------------------------  RESULT TABLE  ----------------------------------
summary = pd.DataFrame(summary)
print('\n====== GRID COMPLETE ======')
print(summary.round(3).tail())   # quick peek at tail

# ---- top-10 by Calmar -------------------------------------------------------
top_calmar = summary.nlargest(10, 'calmar')[['lev', 'stp_pct', 'calmar', 'cagr', 'maxdd', 'sharpe']]
print('\nTop 10 combinations by Calmar:')
print(top_calmar.round(3))

# ---- best Sharpe -----------------------------------------------------------
best = summary.loc[summary.sharpe.idxmax()]
print('\nBest Sharpe:\n', best)

# ---- save everything -------------------------------------------------------
summary.to_csv('macd_lev_stp_grid.csv', index=False)
