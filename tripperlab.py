import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time
from datetime import datetime

# ==========================================
# CONFIGURATION - Optimal Parameters
# ==========================================
X_DAYS = 24           # Optimal Feature lookback window (X days)
ZETA_DAYS = 3         # Optimal Target prediction horizon (Zeta days)
LEVERAGE = 2.0        # Leverage factor (1.0 = no leverage, 5.0 = 5x)

# Constants
START_DATE = "01 Jan, 2018"
SYMBOL = "BTCUSDT"
INITIAL_CAPITAL = 10000

# Stop-Loss Range (Percentage of entry price)
SL_MIN_PCT = 0.1 / 100  # 0.1% (Original minimum, now unused for fixed SL)
SL_MAX_PCT = 5.0 / 100 # 10.0% (Used as the FIXED stop loss percentage)

# ==========================================

# --- CUSTOM SCALER CLASS ---
class CustomMinMaxScaler:
    """Scales data to a custom range [min_target, max_target]."""
    def __init__(self, min_target, max_target):
        self.min_target = min_target
        self.max_target = max_target
        self.data_min = None
        self.data_max = None

    def fit(self, X):
        """Calculates min/max from the training data (X)."""
        self.data_min = X.min()
        self.data_max = X.max()
        return self

    def transform(self, X):
        """Applies the transformation using fitted min/max."""
        # Check for zero range to avoid division by zero
        epsilon = 1e-7
        
        # Min-Max formula: (X - min) / (max - min)
        X_std = (X - self.data_min) / (self.data_max - self.data_min + epsilon)
        
        # Scale to target range: X_scaled * (target_max - target_min) + target_min
        X_scaled = X_std * (self.max_target - self.min_target) + self.min_target
        
        return X_scaled

# --- DATA & INDICATOR FUNCTIONS ---

def get_binance_data(symbol, start_date):
    """Fetches daily klines from Binance starting from start_date."""
    base_url = "https://api.binance.com/api/v3/klines"
    
    dt_obj = datetime.strptime(start_date, "%d %b, %Y")
    start_ts = int(dt_obj.timestamp() * 1000)
    
    klines = []
    print(f"Fetching data for {symbol} from {start_date}...")
    
    while True:
        params = {
            'symbol': symbol,
            'interval': '1d',
            'startTime': start_ts,
            'limit': 1000
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status() # Raise HTTPError for bad responses
            data = response.json()
            
            if not data or len(data) == 0:
                break
                
            klines.extend(data)
            
            last_kline_time = data[-1][0]
            start_ts = last_kline_time + 86400000
            
            time.sleep(0.1)
            
            if last_kline_time >= int(time.time() * 1000):
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
            
    print(f"Total days fetched: {len(klines)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # We now need high/low for the backtest
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('date', inplace=True)
    
    return df[['open', 'high', 'low', 'close', 'volume']]

def calculate_indicators(df):
    """
    Calculates MACD Diff, Stoch RSI, Price Change (Simple PCT), Volume Change (Simple PCT).
    """
    df = df.copy()
    
    # 1. MACD (Standard 12, 26, 9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_diff'] = macd_line - signal_line
    
    # 2. Stochastic RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    min_rsi = rsi.rolling(window=14).min()
    max_rsi = rsi.rolling(window=14).max()
    df['stoch_rsi'] = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-7)
    
    # 3. Price Change (Simple Percentage Change)
    df['price_change'] = df['close'].pct_change()
    
    # 4. Volume Change (Simple Percentage Change)
    df['volume_change'] = df['volume'].pct_change()
    
    df.dropna(inplace=True)
    return df

def prepare_features(df, x_days, zeta_days):
    """
    Creates feature columns for X days of history and the Target.
    """
    data = df.copy()
    feature_cols = []
    
    # Create lag features for X days
    for i in range(x_days):
        for metric in ['macd_diff', 'stoch_rsi', 'price_change', 'volume_change']:
            col_name = f'{metric}_lag_{i}'
            data[col_name] = data[metric].shift(i)
            feature_cols.append(col_name)
            
    # Create Target: Direction ZETA_DAYS ahead
    data['future_close'] = data['close'].shift(-zeta_days)
    data['target'] = (data['future_close'] > data['close']).astype(int)
    
    data.dropna(inplace=True)
    
    return data, feature_cols

def calculate_position_size(probability):
    """
    Calculates position size based on conviction (distance from 0.5).
    Scales probability from [0.0, 1.0] to [-1.0, 1.0].
    """
    # Conviction is the distance from 0.5 (max 0.5), scaled to max 1.0
    conviction = (probability - 0.5) * 2
    return conviction

def get_dynamic_stop_loss(conviction):
    """
    Returns a fixed stop loss percentage, independent of conviction.
    Uses the maximum defined SL percentage for a fixed risk value.
    """
    # *** CHANGE APPLIED HERE: Stop loss is now fixed and does not depend on conviction ***
    return SL_MAX_PCT

# --- MAIN EXECUTION ---

def main():
    # 1. Initial Data Fetch
    df = get_binance_data(SYMBOL, START_DATE)
    df_indicators = calculate_indicators(df)
    
    # 2. Prepare Features
    data, features = prepare_features(df_indicators, X_DAYS, ZETA_DAYS)
    
    print("-" * 50)
    print(f"Running Strategy (X={X_DAYS}, ZETA={ZETA_DAYS}, Leverage={LEVERAGE:.1f}x)")
    print(f"Dataset size after processing: {len(data)} rows")
    print("-" * 50)

    if len(data) < 100:
        print("Not enough data to run the backtest after cleanup.")
        return

    # 3. Train / Test Split (70% Train, 15% Test1, 15% Test2)
    N = len(data)
    split_idx_1 = int(N * 0.70)
    split_idx_2 = int(N * 0.85)

    train_data = data.iloc[:split_idx_1]
    test_data_1 = data.iloc[split_idx_1:split_idx_2]
    
    # Prepare sets for ML
    X_train = train_data[features]
    y_train = train_data['target']
    X_test_1 = test_data_1[features]
    y_test_1 = test_data_1['target']
    
    print(f"Train size: {len(train_data)} | Test 1 size: {len(test_data_1)}")
    
    # 4. Feature Scaling (All features to [-1, 1])
    print(f"Scaling {len(features)} features to the [-1, 1] range, based on training data...")
    
    all_features_scaler = CustomMinMaxScaler(-1, 1).fit(X_train)
    X_train_scaled = all_features_scaler.transform(X_train)
    X_test_1_scaled = all_features_scaler.transform(X_test_1)
    
    # 5. Train Model (Max Iterations set to 10,000)
    print("\nTraining Logistic Regression...")
    
    model = LogisticRegression(max_iter=10000, solver='lbfgs', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 6. Prediction (ONLY on Test Set 1)
    probs_1 = model.predict_proba(X_test_1_scaled)[:, 1]
    preds_1 = model.predict(X_test_1_scaled)
    
    # 7. Accuracy Report
    acc = accuracy_score(y_test_1, preds_1)
    print(f"\n--- Prediction Accuracy on Test Set 1: {acc:.4f} ---")
    
    # 8. Capital Development Backtest (ONLY on Test Set 1)
    capital = INITIAL_CAPITAL
    capital_history = [capital]
    
    # Combine predictions and necessary data for backtest
    backtest_df = test_data_1[['close', 'high', 'low']].copy()
    # Shift high/low back by 1 day so that index i has entry price (close[i]) and exit info (high[i+1], low[i+1], close[i+1])
    backtest_df['next_close'] = backtest_df['close'].shift(-1)
    backtest_df['next_high'] = backtest_df['high'].shift(-1)
    backtest_df['next_low'] = backtest_df['low'].shift(-1)
    backtest_df['model_prob'] = probs_1
    
    # Remove the last row which has no next day data
    backtest_df = backtest_df.iloc[:-1]

    # Backtest simulation
    print("Starting backtest simulation with FIXED stop-loss...")
    for index, row in backtest_df.iterrows():
        entry_price = row['close']
        next_high = row['next_high']
        next_low = row['next_low']
        exit_price = row['next_close']
        prob = row['model_prob']
        
        # 1. Determine Position and Size
        conviction = calculate_position_size(prob)
        position_size = abs(conviction)
        direction = 1 if conviction > 0 else -1 # 1 for Long, -1 for Short
        
        # 2. Determine FIXED Stop Loss
        sl_pct = get_dynamic_stop_loss(conviction) # This now returns SL_MAX_PCT
        
        # Calculate actual Stop Loss price level
        if direction == 1: # Long position: Stop loss is BELOW entry price
            sl_price = entry_price * (1 - sl_pct)
        else: # Short position: Stop loss is ABOVE entry price
            sl_price = entry_price * (1 + sl_pct)
            
        # 3. Simulate Daily PnL
        trade_pnl_pct = 0.0
        
        if direction == 1: # Long
            # Check if stop loss was hit (next day low went below SL price)
            if next_low <= sl_price:
                # Loss is capped at SL_PCT
                trade_pnl_pct = -sl_pct
            else:
                # PnL is based on closing price change
                trade_pnl_pct = (exit_price - entry_price) / entry_price
        
        else: # Short
            # Check if stop loss was hit (next day high went above SL price)
            if next_high >= sl_price:
                # Loss is capped at SL_PCT
                trade_pnl_pct = -sl_pct
            else:
                # PnL is based on closing price change (inverted for short)
                trade_pnl_pct = (entry_price - exit_price) / entry_price
        
        # 4. Apply Position Size and Leverage to Capital
        # Total daily return: (Trade PnL % * Leverage * Position Size)
        daily_return = trade_pnl_pct * LEVERAGE * position_size
        daily_pnl = capital * daily_return
        
        capital += daily_pnl
        capital_history.append(capital)

    final_capital = capital_history[-1]
    strategy_return_percent = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    # 9. Visualization
    dates = backtest_df.index
    print("-" * 30)
    print(f"--- Final Strategy Performance (Test Set 1) ---")
    print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Final Capital:   ${final_capital:.2f}")
    print(f"Strategy Return: {strategy_return_percent:.2f}%")
    print(f"Fixed SL: {SL_MAX_PCT*100:.1f}%, Leverage: {LEVERAGE:.1f}x")
    print("-" * 30)
    
    plt.figure(figsize=(12, 6))
    
    # Calculate Buy & Hold for comparison
    buy_hold_data = test_data_1.copy()
    buy_hold_return = buy_hold_data['close'] / buy_hold_data['close'].iloc[0] * INITIAL_CAPITAL
    
    # Plot strategy equity (using dates aligned with the backtest results)
    plt.plot(dates[:len(capital_history)-1], capital_history[:-1], label='Strategy Equity')
    
    # Plot Buy & Hold (using original test set dates)
    plt.plot(buy_hold_data.index, buy_hold_return, label='Buy & Hold BTC (Test Set 1)', alpha=0.5, linestyle='--')
    
    plt.title(f"Capital Development for ML Strategy (X={X_DAYS}, Z={ZETA_DAYS})")
    plt.xlabel("Date")
    plt.ylabel("Capital ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
