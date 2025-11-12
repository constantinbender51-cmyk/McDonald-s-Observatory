import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
START_DATE = '2018-01-01'  # Data start date
LOOKBACK_DAYS = 10  # Number of days for input features
PREDICTION_HORIZONS = [6, 10]  # Days ahead to predict
STOP_LOSS_MULTIPLIER = 0.8  # Stop loss as fraction of predicted return (80%)
TRAIN_TEST_SPLIT = 0.8  # Train/test split ratio
INITIAL_CAPITAL = 1000  # Starting capital in USD

# MACD parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Stochastic RSI parameters
STOCH_RSI_PERIOD = 14
STOCH_RSI_SMOOTH_K = 3
STOCH_RSI_SMOOTH_D = 3

# Logistic Regression parameters
LR_MAX_ITER = 1000
LR_RANDOM_STATE = 42

# Binance API parameters
BINANCE_SYMBOL = 'BTCUSDT'
BINANCE_INTERVAL = '1d'
# ============================================================================

def fetch_binance_data(symbol=BINANCE_SYMBOL, interval=BINANCE_INTERVAL, start_date=START_DATE):
    """Fetch OHLCV data from Binance"""
    url = 'https://api.binance.com/api/v3/klines'
    
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000)
    
    all_data = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_ts,
            'limit': 1000
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        current_ts = data[-1][0] + 1
        
        if len(data) < 1000:
            break
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                          'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                          'taker_buy_quote', 'ignore'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].set_index('timestamp')

def calculate_macd(prices, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """Calculate MACD signal line"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return signal_line

def calculate_stoch_rsi(prices, period=STOCH_RSI_PERIOD, smooth_k=STOCH_RSI_SMOOTH_K, smooth_d=STOCH_RSI_SMOOTH_D):
    """Calculate Stochastic RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    stoch_rsi = (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min())
    stoch_rsi_k = stoch_rsi.rolling(window=smooth_k).mean()
    
    return stoch_rsi_k

def create_features(df, lookback=LOOKBACK_DAYS):
    """Create features for the model"""
    features = pd.DataFrame(index=df.index)
    
    # Price percentage changes for last 10 days
    for i in range(1, lookback + 1):
        features[f'close_pct_{i}'] = df['close'].pct_change(i)
    
    # Volume percentage changes for last 10 days
    for i in range(1, lookback + 1):
        features[f'volume_pct_{i}'] = df['volume'].pct_change(i)
    
    # MACD signal line
    features['macd_signal'] = calculate_macd(df['close'])
    
    # Stochastic RSI
    features['stoch_rsi'] = calculate_stoch_rsi(df['close'])
    
    return features

def create_targets(df, horizons=PREDICTION_HORIZONS):
    """Create target variables (future returns)"""
    targets = pd.DataFrame(index=df.index)
    
    for h in horizons:
        targets[f'return_{h}d'] = df['close'].pct_change(h).shift(-h)
        targets[f'direction_{h}d'] = (targets[f'return_{h}d'] > 0).astype(int)
    
    return targets

def prepare_data(df):
    """Prepare features and targets"""
    features = create_features(df)
    targets = create_targets(df)
    
    data = pd.concat([features, targets], axis=1).dropna()
    
    return data

def train_models(data):
    """Train logistic regression models"""
    feature_cols = [col for col in data.columns if col.startswith(('close_pct', 'volume_pct', 'macd', 'stoch'))]
    
    X = data[feature_cols]
    y_6d = data['direction_6d']
    y_10d = data['direction_10d']
    
    # Split into train and test (80/20)
    split_idx = int(len(X) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_6d_train, y_6d_test = y_6d.iloc[:split_idx], y_6d.iloc[split_idx:]
    y_10d_train, y_10d_test = y_10d.iloc[:split_idx], y_10d.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    model_6d = LogisticRegression(max_iter=1000, random_state=42)
    model_10d = LogisticRegression(max_iter=1000, random_state=42)
    
    model_6d.fit(X_train_scaled, y_6d_train)
    model_10d.fit(X_train_scaled, y_10d_train)
    
    return model_6d, model_10d, scaler, X_test, y_6d_test, y_10d_test, data.iloc[split_idx:], split_idx

def backtest_strategy(df, data, model_6d, model_10d, scaler, test_start_idx, initial_capital=1000):
    """Backtest the trading strategy"""
    feature_cols = [col for col in data.columns if col.startswith(('close_pct', 'volume_pct', 'macd', 'stoch'))]
    
    test_data = data.iloc[test_start_idx:].copy()
    test_prices = df.loc[test_data.index, 'close']
    
    X_test = test_data[feature_cols]
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions
    pred_6d = model_6d.predict(X_test_scaled)
    pred_10d = model_10d.predict(X_test_scaled)
    
    # Get predicted returns for stop loss calculation
    returns_6d = test_data['return_6d'].values
    
    capital = initial_capital
    position = 0  # 0: no position, 1: long, -1: short
    entry_price = 0
    entry_idx = 0
    stop_loss = 0
    
    trades = []
    
    for i in range(len(test_data)):
        current_price = test_prices.iloc[i]
        
        # Check stop loss if in position
        if position != 0:
            if position == 1:  # Long position
                if current_price <= stop_loss:
                    # Close position
                    pnl_pct = (current_price - entry_price) / entry_price
                    capital = capital * (1 + pnl_pct)
                    trades.append({'entry': entry_price, 'exit': current_price, 'pnl': pnl_pct, 'type': 'long', 'reason': 'stop_loss'})
                    position = 0
            elif position == -1:  # Short position
                if current_price >= stop_loss:
                    # Close position
                    pnl_pct = (entry_price - current_price) / entry_price
                    capital = capital * (1 + pnl_pct)
                    trades.append({'entry': entry_price, 'exit': current_price, 'pnl': pnl_pct, 'type': 'short', 'reason': 'stop_loss'})
                    position = 0
        
        # Generate signals
        signal = 0
        if pred_6d[i] == 1 and pred_10d[i] == 1:
            signal = 1  # Both predict up
        elif pred_6d[i] == 0 and pred_10d[i] == 0:
            signal = -1  # Both predict down
        
        # Execute trades based on signals
        if signal != 0 and signal != position:
            # Close existing position if any
            if position != 0:
                if position == 1:
                    pnl_pct = (current_price - entry_price) / entry_price
                elif position == -1:
                    pnl_pct = (entry_price - current_price) / entry_price
                capital = capital * (1 + pnl_pct)
                trades.append({'entry': entry_price, 'exit': current_price, 'pnl': pnl_pct, 'type': 'long' if position == 1 else 'short', 'reason': 'signal_change'})
            
            # Open new position
            position = signal
            entry_price = current_price
            entry_idx = i
            
            # Set stop loss based on 6-day prediction magnitude
            predicted_return_6d = returns_6d[i]
            if not np.isnan(predicted_return_6d):
                stop_loss_pct = -0.8 * abs(predicted_return_6d)
                if position == 1:
                    stop_loss = entry_price * (1 + stop_loss_pct)
                else:
                    stop_loss = entry_price * (1 - stop_loss_pct)
            else:
                # Default 5% stop loss if prediction not available
                if position == 1:
                    stop_loss = entry_price * 0.95
                else:
                    stop_loss = entry_price * 1.05
    
    # Close final position if any
    if position != 0:
        final_price = test_prices.iloc[-1]
        if position == 1:
            pnl_pct = (final_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - final_price) / entry_price
        capital = capital * (1 + pnl_pct)
        trades.append({'entry': entry_price, 'exit': final_price, 'pnl': pnl_pct, 'type': 'long' if position == 1 else 'short', 'reason': 'end'})
    
    return capital, trades

# Main execution
print("Fetching Bitcoin data from Binance...")
df = fetch_binance_data()

print("Preparing features and targets...")
data = prepare_data(df)

print("Training models...")
model_6d, model_10d, scaler, X_test, y_6d_test, y_10d_test, test_data, split_idx = train_models(data)

print("Running backtest...")
final_capital, trades = backtest_strategy(df, data, model_6d, model_10d, scaler, split_idx)

# Calculate benchmark (buy and hold)
test_prices = df.loc[test_data.index, 'close']
initial_price = test_prices.iloc[0]
final_price = test_prices.iloc[-1]
benchmark_capital = 1000 * (final_price / initial_price)

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Final Capital (Strategy): ${final_capital:.2f}")
print(f"Benchmark (Buy & Hold):   ${benchmark_capital:.2f}")
print(f"Strategy Return:          {((final_capital / 1000) - 1) * 100:.2f}%")
print(f"Benchmark Return:         {((benchmark_capital / 1000) - 1) * 100:.2f}%")
print(f"Number of Trades:         {len(trades)}")
print("="*50)
