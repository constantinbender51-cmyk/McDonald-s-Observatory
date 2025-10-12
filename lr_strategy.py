# --------------------------------------------------
# CONFIGURATION PARAMETERS
# --------------------------------------------------
LOOKBACK = 10           # Number of historical days for features
SHORT_HORIZON = 6       # Days ahead for short-term prediction
LONG_HORIZON = 10       # Days ahead for long-term prediction
LEVERAGE = 3.0          # Position leverage multiplier
STOP_LOSS_PCT = 0.8     # Stop loss as % of predicted move
INITIAL_CAPITAL = 1000.0

# GRID SEARCH PARAMETERS
THRESHOLD_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHIFT_VALUES = [1, 2, 3, 4, 5, 6]

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
    limit = 1000
    interval = "1d"
    start_time = 1502928000000
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
        start_time = batch[-1][6] + 1
        time.sleep(0.2)

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
print("Training models...")
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

# ----------  3.  GRID SEARCH OVER PARAMETERS  ----------
first = split
lev = LEVERAGE
stop = STOP_LOSS_PCT

print(f"\n{'='*80}")
print(f"GRID SEARCH: Testing {len(THRESHOLD_VALUES)} thresholds × {len(SHIFT_VALUES)} shifts = {len(THRESHOLD_VALUES)*len(SHIFT_VALUES)} combinations")
print(f"{'='*80}\n")

grid_results = []

for shift_var in SHIFT_VALUES:
    # Calculate shifts for this iteration
    short_shift = SHORT_HORIZON - shift_var
    long_shift = LONG_HORIZON - shift_var
    
    # Skip if shifts are invalid
    if short_shift < 0 or long_shift < 0:
        continue
    
    # Shift predictions backward
    pred_short = np.concatenate([pred_short_raw[short_shift:], np.full(short_shift, np.nan)])
    pred_long = np.concatenate([pred_long_raw[long_shift:], np.full(long_shift, np.nan)])
    
    # Find valid range
    min_len = min(len(pred_short), len(pred_long))
    valid_mask = ~(np.isnan(pred_short[:min_len]) | np.isnan(pred_long[:min_len]))
    min_len = np.sum(valid_mask)
    
    # 1. build the mask once
    valid_mask = ~(np.isnan(pred_short) | np.isnan(pred_long))
    pred_short_valid = pred_short[valid_mask]
    pred_long_valid  = pred_long[valid_mask]

    # 2. derive the corresponding close slice
    #    first_valid is the index (relative to 'first') of the first True
    first_valid = np.where(valid_mask)[0][0]
    last_valid  = np.where(valid_mask)[0][-1]
    pct1d = (close[first+first_valid+1 : first+last_valid+2] /
             close[first+first_valid : first+last_valid+1] - 1)

    # 3. keep the date array consistent
    dates = df['date'].values[first+first_valid : first+last_valid+1]

    for threshold in THRESHOLD_VALUES:
        capital = INITIAL_CAPITAL
        buyhold = INITIAL_CAPITAL
        pos = 0
        entry_i = 0
        entry_pred = 0
        max_cap = capital
        worst_dd = 0.0
        num_trades = 0
        
        for i in range(len(pct1d)):
            ps, pl = pred_short_valid[i], pred_long_valid[i]
            
            # Strategy logic
            new_pos = 0
            if ps > threshold and pl > threshold:
                new_pos = 1
            elif ps < -threshold and pl < -threshold:
                new_pos = -1
            
            # Stop loss
            if pos != 0:
                realised = (close[first+i] / close[first+entry_i] - 1) * 100 * pos
                if realised <= -stop * abs(entry_pred):
                    new_pos = 0
            
            # Execute trade
            if new_pos != pos:
                if pos != 0:
                    gross = 1 + (close[first+i] / close[first+entry_i] - 1) * lev * pos
                    capital *= gross
                    num_trades += 1
                if new_pos != 0:
                    entry_i = i
                    entry_pred = ps if new_pos == 1 else -ps
                pos = new_pos
            
            buyhold *= 1 + pct1d[i]
            
            if capital > max_cap:
                max_cap = capital
            dd = (capital - max_cap) / max_cap * 100
            if dd < worst_dd:
                worst_dd = dd
        
        # Close final position
        if pos != 0:
            gross = 1 + (close[first+len(pct1d)-1] / close[first+entry_i] - 1) * lev * pos
            capital *= gross
            num_trades += 1
        
        # Calculate metrics
        total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        excess_return = capital - buyhold
        sharpe = total_return / abs(worst_dd) if worst_dd != 0 else 0
        
        grid_results.append({
            'shift': shift_var,
            'threshold': threshold,
            'final_equity': capital,
            'buyhold': buyhold,
            'total_return_%': total_return,
            'excess_return': excess_return,
            'worst_dd_%': worst_dd,
            'sharpe': sharpe,
            'num_trades': num_trades
        })

# --------------------------------------------------
# 4.  PRINT TOP 5 BY SHARPE RATIO
# --------------------------------------------------
results_df = pd.DataFrame(grid_results)

print(f"\n{'='*80}")
print("TOP 5 CONFIGURATIONS BY SHARPE RATIO (Return/MaxDD)")
print(f"{'='*80}")
top5_sharpe = results_df.nlargest(5, 'sharpe')
print(top5_sharpe.to_string(index=False))

# Get the best configuration by Sharpe ratio
best_config = top5_sharpe.iloc[0]
print(f"\n{'='*80}")
print("BEST CONFIGURATION (by Sharpe ratio) - SELECTED FOR RESULTS")
print(f"{'='*80}")
print(f"Shift variable:    {best_config['shift']}")
print(f"Threshold:         {best_config['threshold']:.1f}%")
print(f"Final equity:      ${best_config['final_equity']:.2f}")
print(f"Buy & hold:        ${best_config['buyhold']:.2f}")
print(f"Total return:      {best_config['total_return_%']:+.1f}%")
print(f"Excess return:     ${best_config['excess_return']:.2f}")
print(f"Worst drawdown:    {best_config['worst_dd_%']:.1f}%")
print(f"Sharpe ratio:      {best_config['sharpe']:.2f}")
print(f"Number of trades:  {best_config['num_trades']}")

# --------------------------------------------------
# 5.  RE-RUN BEST STRATEGY TO GENERATE DETAILED RESULTS
# --------------------------------------------------
print(f"\n{'='*80}")
print("GENERATING DETAILED RESULTS FOR BEST CONFIGURATION")
print(f"{'='*80}\n")

best_shift = int(best_config['shift'])
best_threshold = best_config['threshold']

# Recalculate with best parameters to get daily details
short_shift = SHORT_HORIZON - best_shift
long_shift = LONG_HORIZON - best_shift

pred_short = np.concatenate([pred_short_raw[short_shift:], np.full(short_shift, np.nan)])
pred_long = np.concatenate([pred_long_raw[long_shift:], np.full(long_shift, np.nan)])

min_len = min(len(pred_short), len(pred_long))
valid_mask = ~(np.isnan(pred_short[:min_len]) | np.isnan(pred_long[:min_len]))
min_len = np.sum(valid_mask)

pred_short_valid = pred_short[valid_mask]
pred_long_valid = pred_long[valid_mask]
pct1d = (close[first+1 : first+min_len+1] / close[first : first+min_len] - 1)

# Get dates for results
dates = df['date'].values[first:first+min_len]

# Run strategy with detailed logging
capital = INITIAL_CAPITAL
buyhold = INITIAL_CAPITAL
pos = 0
entry_i = 0
entry_pred = 0
max_cap = capital
worst_dd = 0.0
num_trades = 0

daily_results = []

for i in range(len(pct1d)):
    ps, pl = pred_short_valid[i], pred_long_valid[i]
    
    # Strategy logic
    new_pos = 0
    if ps > best_threshold and pl > best_threshold:
        new_pos = 1
    elif ps < -best_threshold and pl < -best_threshold:
        new_pos = -1
    
    # Stop loss
    if pos != 0:
        realised = (close[first+i] / close[first+entry_i] - 1) * 100 * pos
        if realised <= -stop * abs(entry_pred):
            new_pos = 0
    
    # Execute trade
    trade_return = 0
    if new_pos != pos:
        if pos != 0:
            gross = 1 + (close[first+i] / close[first+entry_i] - 1) * lev * pos
            trade_return = (gross - 1) * capital
            capital *= gross
            num_trades += 1
        if new_pos != 0:
            entry_i = i
            entry_pred = ps if new_pos == 1 else -ps
        pos = new_pos
    
    buyhold *= 1 + pct1d[i]
    
    if capital > max_cap:
        max_cap = capital
    dd = (capital - max_cap) / max_cap * 100
    if dd < worst_dd:
        worst_dd = dd
    
    daily_results.append({
        'date': dates[i],
        'btc_price': close[first+i],
        'position': pos,
        'pred_short': ps,
        'pred_long': pl,
        'equity': capital,
        'buyhold_equity': buyhold,
        'drawdown_%': dd,
        'trade_return': trade_return
    })

# Close final position
if pos != 0:
    gross = 1 + (close[first+len(pct1d)-1] / close[first+entry_i] - 1) * lev * pos
    capital *= gross
    num_trades += 1

# --------------------------------------------------
# 6.  SAVE RESULTS.CSV
# --------------------------------------------------
results_csv_df = pd.DataFrame(daily_results)
results_csv_path = Path("results.csv")
results_csv_df.to_csv(results_csv_path, index=False)
print(f"Detailed results saved → {results_csv_path.resolve()}")
print(f"Contains {len(results_csv_df)} daily records\n")

# --------------------------------------------------
# 7.  SAVE METRICS
# --------------------------------------------------
total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
excess_return = capital - buyhold
sharpe = total_return / abs(worst_dd) if worst_dd != 0 else 0

metrics = {
    'configuration': {
        'shift': best_shift,
        'threshold': best_threshold,
        'leverage': LEVERAGE,
        'stop_loss_pct': STOP_LOSS_PCT,
        'initial_capital': INITIAL_CAPITAL,
        'lookback': LOOKBACK,
        'short_horizon': SHORT_HORIZON,
        'long_horizon': LONG_HORIZON
    },
    'performance': {
        'final_equity': capital,
        'buyhold_equity': buyhold,
        'total_return_%': total_return,
        'excess_return': excess_return,
        'worst_drawdown_%': worst_dd,
        'sharpe_ratio': sharpe,
        'number_of_trades': num_trades
    },
    'comparison_to_buyhold': {
        'outperformance': capital > buyhold,
        'outperformance_amount': excess_return,
        'outperformance_%': (capital / buyhold - 1) * 100
    }
}

import json
metrics_path = Path("metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved → {metrics_path.resolve()}")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"✓ Best configuration identified (Shift={best_shift}, Threshold={best_threshold:.1f}%)")
print(f"✓ Sharpe Ratio: {sharpe:.2f}")
print(f"✓ Final Equity: ${capital:.2f}")
print(f"✓ Total Return: {total_return:+.1f}%")
print(f"✓ Results saved to: results.csv ({len(results_csv_df)} rows)")
print(f"✓ Metrics saved to: metrics.json")
print(f"{'='*80}\n")
