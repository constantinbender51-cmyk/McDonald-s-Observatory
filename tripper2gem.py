import ccxt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time

def fetch_binance_data(symbol='BTC/USDT', timeframe='1d', start_date='2018-01-01T00:00:00Z'):
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
            # print(f"Fetched {len(all_ohlcv)} candles so far...") # Commented out to reduce clutter
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def calculate_indicators(df):
    df['pct_change'] = df['close'].pct_change()
    df['vol_change'] = df['volume'].pct_change()
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    
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
    features = []
    for i in range(1, window + 1):
        col_name_p = f'pct_change_lag_{i}'
        df[col_name_p] = df['pct_change'].shift(i)
        features.append(col_name_p)
        col_name_v = f'vol_change_lag_{i}'
        df[col_name_v] = df['vol_change'].shift(i)
        features.append(col_name_v)
    
    features.extend(['macd', 'signal_line', 'stoch_rsi'])
    
    # Target: 1 if future close > current close, else 0
    future_close = df['close'].shift(-prediction_horizon)
    df['target'] = (future_close > df['close']).astype(int)
    df_clean = df.dropna()
    return df_clean, features

def train_and_evaluate_with_threshold(df, feature_cols):
    X = df[feature_cols]
    y = df['target']
    
    # Split (Shuffle=False)
    split_point = int(len(df) * 0.80)
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # --- STANDARD EVALUATION (Threshold 0.5) ---
    y_pred_std = model.predict(X_test_scaled)
    acc_std = accuracy_score(y_test, y_pred_std)
    
    print(f"\n{'='*50}")
    print(f"STANDARD MODEL (Threshold 0.5)")
    print(f"{'='*50}")
    print(f"Accuracy: {acc_std:.4f} ({acc_std*100:.2f}%)")
    print(f"Total Trades Taken: {len(y_test)}")
    
    # --- CUSTOM THRESHOLD EVALUATION (0.4 / 0.6) ---
    # Get raw probabilities for Class 1 (Price Increase)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Define High Confidence Mask
    # Logic: Trade if prob > 0.6 OR prob < 0.4
    high_conf_mask = (probs > 0.6) | (probs < 0.4)
    
    # Filter the test set to only include high confidence rows
    y_test_filtered = y_test[high_conf_mask]
    probs_filtered = probs[high_conf_mask]
    
    # Convert probabilities to predictions based on the threshold
    # If prob > 0.6, prediction is 1. If prob < 0.4, prediction is 0.
    y_pred_filtered = (probs_filtered > 0.6).astype(int)
    
    print(f"\n{'='*50}")
    print(f"HIGH CONFIDENCE MODEL (Thresholds < 0.4 and > 0.6)")
    print(f"{'='*50}")
    
    if len(y_test_filtered) > 0:
        acc_conf = accuracy_score(y_test_filtered, y_pred_filtered)
        coverage = len(y_test_filtered) / len(y_test) * 100
        
        print(f"Accuracy: {acc_conf:.4f} ({acc_conf*100:.2f}%)")
        print(f"Trades Taken: {len(y_test_filtered)} out of {len(y_test)}")
        print(f"Coverage (Trades / Total Days): {coverage:.2f}%")
        
        print("\nBreakdown of High Confidence Trades:")
        print(classification_report(y_test_filtered, y_pred_filtered))
    else:
        print("No predictions met the confidence threshold criteria.")

if __name__ == "__main__":
    df = fetch_binance_data(symbol='BTC/USDT', start_date='2018-01-01T00:00:00Z')
    if not df.empty:
        df = calculate_indicators(df)
        df_model, feature_columns = create_features_and_target(df, window=28, prediction_horizon=7)
        train_and_evaluate_with_threshold(df_model, feature_columns)
    else:
        print("No data fetched.")
