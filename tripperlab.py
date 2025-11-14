import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time
from datetime import datetime
import itertools

# ==========================================
# CONFIGURATION
# ==========================================
# Optimization Ranges for Brute-Force Search
X_RANGE = [3, 5, 7, 10, 14]       # Feature lookback window (X days)
ZETA_RANGE = [3, 7, 10, 14]      # Target prediction horizon (Zeta days)

# Constants
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
    Requires df to already contain the indicators.
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
    conviction = (probability - 0.5) * 2
    return conviction

def evaluate_model(df_indicators, x_days, zeta_days):
    """
    Runs the full ML pipeline and backtest for a specific (X, Zeta) pair.
    """
    try:
        data, features = prepare_features(df_indicators, x_days, zeta_days)

        if len(data) < 100: # Ensure minimum data points for splitting
            return 0.0, 0.0, 0.0

        # --- 4. Train / Test Split (70% Train, 15% Test1, 15% Test2) ---
        N = len(data)
        split_idx_1 = int(N * 0.70)
        split_idx_2 = int(N * 0.85)

        train_data = data.iloc[:split_idx_1]
        test_data_1 = data.iloc[split_idx_1:split_idx_2]
        # test_data_2 is created but not used for performance tracking
        test_data_2 = data.iloc[split_idx_2:] 
        
        # Prepare sets for ML
        X_train = train_data[features]
        y_train = train_data['target']
        X_test_1 = test_data_1[features]
        y_test_1 = test_data_1['target']
        
        # --- 5. Feature Scaling (All features to [-1, 1]) ---
        all_features_scaler = CustomMinMaxScaler(-1, 1).fit(X_train)
        
        # Transform all sets
        X_train_scaled = all_features_scaler.transform(X_train)
        X_test_1_scaled = all_features_scaler.transform(X_test_1)
        
        # --- 6. Train Model ---
        model = LogisticRegression(max_iter=10000, solver='lbfgs', random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # --- 7. Prediction (Test Set 1) ---
        probs_1 = model.predict_proba(X_test_1_scaled)[:, 1]
        preds_1 = model.predict(X_test_1_scaled)
        
        # Metrics
        acc = accuracy_score(y_test_1, preds_1)
        
        # --- 8. Capital Development Backtest (Test Set 1) ---
        capital = INITIAL_CAPITAL
        
        closes = test_data_1['close'].values
        model_probs = probs_1
        capital_history = [capital]
        
        for i in range(len(test_data_1) - 1):
            current_price = closes[i]
            next_price = closes[i+1]
            prob = model_probs[i]
            
            position_size = calculate_position_size(prob)
            market_return = (next_price - current_price) / current_price
            
            daily_pnl = capital * position_size * market_return
            
            capital += daily_pnl
            capital_history.append(capital)

        final_capital = capital_history[-1]
        strategy_return_percent = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        return strategy_return_percent, acc, capital_history, test_data_1.index

    except Exception as e:
        print(f"Error evaluating X={x_days}, ZETA={zeta_days}: {e}")
        return -100.0, 0.0, [], [] # Return a penalty for failed runs


# --- MAIN EXECUTION FOR OPTIMIZATION ---

def main():
    # 1. Initial Data Fetch (Run once)
    df = get_binance_data(SYMBOL, START_DATE)
    df_indicators = calculate_indicators(df)
    
    print("-" * 50)
    print("Starting Brute-Force Optimization (Grid Search)")
    print(f"X range: {X_RANGE} | ZETA range: {ZETA_RANGE}")
    print("-" * 50)

    # 2. Optimization Loop Setup
    results = []
    best_return = -np.inf
    best_params = None
    best_equity = None
    best_dates = None
    
    total_runs = len(X_RANGE) * len(ZETA_RANGE)
    run_count = 0

    for x, zeta in itertools.product(X_RANGE, ZETA_RANGE):
        run_count += 1
        print(f"Running {run_count}/{total_runs}: X={x} | ZETA={zeta}...", end='\r')
        
        ret, acc, equity_history, dates = evaluate_model(df_indicators, x, zeta)
        
        # Store results
        results.append({
            'X': x,
            'ZETA': zeta,
            'Accuracy': acc,
            'Return (%)': ret
        })
        
        # Check for best result based on Return
        if ret > best_return:
            best_return = ret
            best_params = (x, zeta)
            best_equity = equity_history
            best_dates = dates
            
    print("\n" + "=" * 50)
    print("Optimization Complete")
    print("=" * 50)
    
    # 3. Print Results Table
    results_df = pd.DataFrame(results).sort_values(by='Return (%)', ascending=False)
    
    print("\nTop 5 Parameter Combinations (Ranked by Test Set 1 Return):")
    print(results_df.head())
    
    # 4. Final Output and Plot
    if best_params:
        best_x, best_zeta = best_params
        
        print("\n" + "=" * 50)
        print(f"OPTIMAL PARAMETERS (Maximized Return on Test Set 1):")
        print(f"X_DAYS (Feature Lookback): {best_x}")
        print(f"ZETA_DAYS (Prediction Horizon): {best_zeta}")
        print(f"Achieved Return: {best_return:.2f}%")
        print("=" * 50)
        
        # Plot the optimal strategy's equity curve
        plt.figure(figsize=(12, 6))
        
        # Calculate Buy & Hold for comparison
        buy_hold_data = df_indicators.loc[best_dates.min():best_dates.max()]
        buy_hold_return = buy_hold_data['close'] / buy_hold_data['close'].iloc[0] * INITIAL_CAPITAL
        
        plt.plot(best_dates[:len(best_equity)], best_equity, label=f'Optimal Strategy Equity (X={best_x}, Z={best_zeta})')
        plt.plot(buy_hold_data.index, buy_hold_return, label='Buy & Hold BTC (Test Set 1)', alpha=0.5, linestyle='--')
        
        plt.title(f"Optimal Capital Development on Test Set 1 (X={best_x}, Z={best_zeta})")
        plt.xlabel("Date")
        plt.ylabel("Capital ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    main()
