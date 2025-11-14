import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
X_DAYS = 7            # Number of past days to use as features
ZETA_DAYS = 7         # Target horizon: Predict price direction ZETA days from now
START_DATE = "01 Jan, 2018"
SYMBOL = "BTCUSDT"
INITIAL_CAPITAL = 10000
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
        # Add a small epsilon to data_min and data_max for robustness
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
    
    # Convert start_date to milliseconds timestamp
    dt_obj = datetime.strptime(start_date, "%d %b, %Y")
    start_ts = int(dt_obj.timestamp() * 1000)
    
    klines = []
    print(f"Fetching data for {symbol} from {start_date}...")
    
    # Binance limit is 1000 candles per request
    while True:
        params = {
            'symbol': symbol,
            'interval': '1d',
            'startTime': start_ts,
            'limit': 1000
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if not data or len(data) == 0:
                break
                
            klines.extend(data)
            
            # Update start_ts to the timestamp of the last kline + 1 day
            # 86400000 ms = 1 day
            last_kline_time = data[-1][0]
            start_ts = last_kline_time + 86400000
            
            # Rate limit guard
            time.sleep(0.1)
            
            if last_kline_time >= int(time.time() * 1000):
                break
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    print(f"Total days fetched: {len(klines)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # Type conversion
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
    # Handle the max_rsi - min_rsi == 0 case (Stoch RSI is bounded 0 to 1)
    df['stoch_rsi'] = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-7)
    
    # 3. Price Change (Simple Percentage Change)
    df['price_change'] = df['close'].pct_change()
    
    # 4. Volume Change (Simple Percentage Change)
    df['volume_change'] = df['volume'].pct_change()
    
    # Drop NaNs created by rolling windows
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
            # Shift data back i days to represent history
            data[col_name] = data[metric].shift(i)
            feature_cols.append(col_name)
            
    # Create Target: Direction ZETA_DAYS ahead
    data['future_close'] = data['close'].shift(-zeta_days)
    data['target'] = (data['future_close'] > data['close']).astype(int)
    
    # Drop NaNs created by lagging features and shifting target
    data.dropna(inplace=True)
    
    return data, feature_cols

def calculate_position_size(probability):
    """
    Calculates position size based on conviction (distance from 0.5).
    Scales probability from [0.0, 1.0] to [-1.0, 1.0].
    """
    conviction = (probability - 0.5) * 2
    return conviction

# --- MAIN EXECUTION ---

def main():
    # 1. Fetch Data
    df = get_binance_data(SYMBOL, START_DATE)
    
    # 2. Calculate Base Indicators
    df = calculate_indicators(df)
    
    # 3. Prepare Features (X Days) and Target (Zeta Days)
    data, features = prepare_features(df, X_DAYS, ZETA_DAYS)
    
    print(f"Dataset size after processing: {len(data)} rows")
    print(f"Number of features: {len(features)}")
    
    # 4. Train / Test Split (70% Train, 15% Test1, 15% Test2)
    N = len(data)
    split_idx_1 = int(N * 0.70)
    split_idx_2 = int(N * 0.85)

    train_data = data.iloc[:split_idx_1]
    test_data_1 = data.iloc[split_idx_1:split_idx_2]
    test_data_2 = data.iloc[split_idx_2:] # The second test set (unused for final output)
    
    # Use Test Set 1 for model evaluation and backtest
    X_train = train_data[features]
    y_train = train_data['target']
    
    X_test_1 = test_data_1[features]
    y_test_1 = test_data_1['target']
    
    X_test_2 = test_data_2[features]
    y_test_2 = test_data_2['target']
    
    print(f"Train size: {len(train_data)} | Test 1 size: {len(test_data_1)} | Test 2 size: {len(test_data_2)}")
    
    # --- 5. Feature Scaling (All features to [-1, 1]) ---
    print(f"Scaling all {len(features)} features to the [-1, 1] range, based on training data...")
    
    # Initialize and fit the scaler ONLY on the training data
    all_features_scaler = CustomMinMaxScaler(-1, 1).fit(X_train[features])
    
    # Transform all three sets using the fitted scaler
    X_train[features] = all_features_scaler.transform(X_train[features])
    X_test_1[features] = all_features_scaler.transform(X_test_1[features])
    X_test_2[features] = all_features_scaler.transform(X_test_2[features]) # Scaling Test 2 as well
    
    # 6. Train Model (Max Iterations set to 10,000)
    print("\nTraining Logistic Regression with Max Iterations 10,000...")
    
    model = LogisticRegression(max_iter=10000, solver='lbfgs')
    model.fit(X_train, y_train)
    
    # 7. Prediction (ONLY on Test Set 1)
    # We use prob_class_1 (Probability of Price Going UP)
    probs_1 = model.predict_proba(X_test_1)[:, 1]
    preds_1 = model.predict(X_test_1)
    
    # Accuracy Report
    acc = accuracy_score(y_test_1, preds_1)
    print(f"\n--- Results on Test Set 1 (15% of data) ---")
    print(f"Prediction Accuracy on Test Data 1: {acc:.4f}")
    print("\nClassification Report (Test Set 1):")
    print(classification_report(y_test_1, preds_1))
    
    # 8. Capital Development Backtest (ONLY on Test Set 1)
    capital = INITIAL_CAPITAL
    capital_history = [capital]
    
    test_data_1 = test_data_1.copy()
    test_data_1['model_prob'] = probs_1
    
    closes = test_data_1['close'].values
    dates = test_data_1.index
    model_probs = test_data_1['model_prob'].values
    
    for i in range(len(test_data_1) - 1):
        current_price = closes[i]
        next_price = closes[i+1]
        prob = model_probs[i]
        
        # Calculate conviction/position size (-1.0 to 1.0)
        position_size = calculate_position_size(prob)
        
        # Calculate daily market return (simple percentage change)
        market_return = (next_price - current_price) / current_price
        
        # Strategy PnL: Capital * Position Size * Market Return
        daily_pnl = capital * position_size * market_return
        
        capital += daily_pnl
        capital_history.append(capital)
        
    # 9. Visualization
    final_capital = capital_history[-1]
    print("-" * 30)
    print(f"--- Strategy Performance (Test Set 1) ---")
    print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Final Capital:   ${final_capital:.2f}")
    print(f"Strategy Return: {((final_capital - INITIAL_CAPITAL)/INITIAL_CAPITAL)*100:.2f}%")
    print("-" * 30)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates[:len(capital_history)], capital_history, label='Strategy Equity (Test Set 1)')
    
    # Add a Buy & Hold comparison for context (only on the Test Set 1 period)
    buy_hold_return = test_data_1['close'] / test_data_1['close'].iloc[0] * INITIAL_CAPITAL
    plt.plot(test_data_1.index, buy_hold_return, label='Buy & Hold BTC (Test Set 1)', alpha=0.5, linestyle='--')
    
    plt.title(f"Capital Development (Logistic Regression Backtest on Test Set 1)\nX={X_DAYS}, Target={ZETA_DAYS} Days Ahead")
    plt.xlabel("Date")
    plt.ylabel("Capital ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
