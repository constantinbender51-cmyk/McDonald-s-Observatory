import ccxt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time

def fetch_binance_data(symbol='BTC/USDT', timeframe='1d', start_date='2018-01-01T00:00:00Z'):
    """Fetches historical OHLCV data from Binance, handling pagination."""
    print(f"Fetching {timeframe} data for {symbol} starting from {start_date}...")
    exchange = ccxt.binance({'rateLimit': 1200, 'enableRateLimit': True})
    since = exchange.parse8601(start_date)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            if last_timestamp >= exchange.milliseconds() - 24 * 60 * 60 * 1000: break
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def calculate_indicators(df):
    """Calculates requested technical indicators and price/volume changes."""
    df['pct_change'] = df['close'].pct_change()
    df['vol_change'] = df['volume'].pct_change()
    
    # MACD and Signal Line
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # MACD Histogram (Difference between MACD and Signal Line)
    df['macd_histogram'] = df['macd'] - df['signal_line']
    
    # Stochastic RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    min_rsi = rsi.rolling(window=14).min()
    max_rsi = rsi.rolling(window=14).max()
    df['stoch_rsi'] = (rsi - min_rsi) / (max_rsi - min_rsi)
    
    return df

def create_features_and_target(df, window=28, prediction_horizon=7):
    """Creates lagged features and two separate binary targets (UP and DOWN)."""
    features = []
    for i in range(1, window + 1):
        col_name_p = f'pct_change_lag_{i}'
        df[col_name_p] = df['pct_change'].shift(i)
        features.append(col_name_p)
        col_name_v = f'vol_change_lag_{i}'
        df[col_name_v] = df['vol_change'].shift(i)
        features.append(col_name_v)
    
    features.extend(['macd_histogram', 'stoch_rsi'])
    
    future_close = df['close'].shift(-prediction_horizon)
    
    # Target 1: Price Increase (1 if future close > current close, 0 otherwise)
    df['target_up'] = (future_close > df['close']).astype(int)
    
    # Target 2: Price Decrease (1 if future close < current close, 0 otherwise)
    df['target_down'] = (future_close < df['close']).astype(int)
    
    df_clean = df.dropna()
    return df_clean, features

def train_and_evaluate_two_models(df, feature_cols):
    """Trains and evaluates two Logistic Regression models (UP and DOWN)."""
    
    X = df[feature_cols]
    
    # Split
    split_point = int(len(df) * 0.80)
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================================================================
    # 1. MODEL UP (Predicting Price Increase)
    # =========================================================================
    y_up = df['target_up']
    y_up_train = y_up.iloc[:split_point]
    y_up_test = y_up.iloc[split_point:]
    
    model_up = LogisticRegression(random_state=42, max_iter=1000)
    model_up.fit(X_train_scaled, y_up_train)
    y_pred_up = model_up.predict(X_test_scaled)
    acc_up = accuracy_score(y_up_test, y_pred_up)
    
    print(f"\n{'='*60}")
    print("MODEL 1: PREDICTING PRICE INCREASE (Target: future price > current price)")
    print(f"{'='*60}")
    print(f"Overall Test Accuracy: {acc_up:.4f} ({acc_up*100:.2f}%)")
    print("\nClassification Report (0: Not Up, 1: Up):")
    print(classification_report(y_up_test, y_pred_up, zero_division=0))

    # =========================================================================
    # 2. MODEL DOWN (Predicting Price Decrease)
    # =========================================================================
    y_down = df['target_down']
    y_down_train = y_down.iloc[:split_point]
    y_down_test = y_down.iloc[split_point:]
    
    model_down = LogisticRegression(random_state=42, max_iter=1000)
    model_down.fit(X_train_scaled, y_down_train)
    y_pred_down = model_down.predict(X_test_scaled)
    acc_down = accuracy_score(y_down_test, y_pred_down)
    
    print(f"\n{'='*60}")
    print("MODEL 2: PREDICTING PRICE DECREASE (Target: future price < current price)")
    print(f"{'='*60}")
    print(f"Overall Test Accuracy: {acc_down:.4f} ({acc_down*100:.2f}%)")
    print("\nClassification Report (0: Not Down, 1: Down):")
    print(classification_report(y_down_test, y_pred_down, zero_division=0))

if __name__ == "__main__":
    df = fetch_binance_data(symbol='BTC/USDT', start_date='2018-01-01T00:00:00Z')
    if not df.empty:
        df = calculate_indicators(df)
        df_model, feature_columns = create_features_and_target(df, window=28, prediction_horizon=7)
        train_and_evaluate_two_models(df_model, feature_columns)
    else:
        print("No data fetched.")
