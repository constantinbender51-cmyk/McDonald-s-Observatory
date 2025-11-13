import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def fetch_bitcoin_data():
    """Fetch daily Bitcoin price data from Binance starting Jan 1, 2018"""
    client = Client()
    
    # Get daily BTCUSDT data from January 1, 2018
    klines = client.get_historical_klines(
        "BTCUSDT",
        Client.KLINE_INTERVAL_1DAY,
        "1 January, 2018"
    )
    
    # Create DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    
    df = pd.DataFrame(klines, columns=columns)
    
    # Convert to proper data types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    
    df.set_index('timestamp', inplace=True)
    
    return df[['open', 'high', 'low', 'close', 'volume']]

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD line and signal line"""
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    return macd, signal_line

def calculate_stochastic_rsi(df, rsi_period=14, stoch_period=14, k=3, d=3):
    """Calculate Stochastic RSI"""
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic RSI
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
    
    return stoch_rsi

def create_features_and_target(df):
    """Create features and target variable"""
    # Calculate price change percentages (10 days)
    df['price_pct_change'] = df['close'].pct_change()
    
    # Calculate volume change percentages (10 days)
    df['volume_pct_change'] = df['volume'].pct_change()
    
    # Calculate MACD and signal line difference (10 days)
    macd, signal_line = calculate_macd(df)
    df['macd_signal_diff'] = macd - signal_line
    
    # Calculate Stochastic RSI (10 days)
    df['stoch_rsi'] = calculate_stochastic_rsi(df)
    
    # Create target: price change direction after 7 days
    df['future_price'] = df['close'].shift(-7)
    df['price_change_direction'] = (df['future_price'] > df['close']).astype(int)
    
    # Create feature columns for last 10 days of each indicator
    feature_columns = []
    
    # 10 days of price change percentages
    for i in range(1, 11):
        df[f'price_pct_change_lag_{i}'] = df['price_pct_change'].shift(i)
        feature_columns.append(f'price_pct_change_lag_{i}')
    
    # 10 days of volume change percentages
    for i in range(1, 11):
        df[f'volume_pct_change_lag_{i}'] = df['volume_pct_change'].shift(i)
        feature_columns.append(f'volume_pct_change_lag_{i}')
    
    # 10 days of MACD - Signal differences
    for i in range(1, 11):
        df[f'macd_signal_diff_lag_{i}'] = df['macd_signal_diff'].shift(i)
        feature_columns.append(f'macd_signal_diff_lag_{i}')
    
    # 10 days of Stochastic RSI values
    for i in range(1, 11):
        df[f'stoch_rsi_lag_{i}'] = df['stoch_rsi'].shift(i)
        feature_columns.append(f'stoch_rsi_lag_{i}')
    
    # Drop rows with NaN values (from lag features and indicator calculations)
    df_clean = df.dropna(subset=feature_columns + ['price_change_direction'])
    
    return df_clean, feature_columns

def main():
    print("Fetching Bitcoin data from Binance...")
    df = fetch_bitcoin_data()
    print(f"Fetched {len(df)} days of data")
    
    print("\nCreating features and target variable...")
    df_clean, feature_columns = create_features_and_target(df)
    print(f"After cleaning: {len(df_clean)} samples with {len(feature_columns)} features")
    
    # Prepare data for training
    X = df_clean[feature_columns]
    y = df_clean['price_change_direction']
    
    # Split data (chronological split - earlier data for training, later for testing)
    split_index = int(len(X) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train logistic regression model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"Prediction Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Additional metrics
    baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
    print(f"Baseline Accuracy (majority class): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"Model Improvement: {(accuracy - baseline_accuracy)*100:.2f}%")
    
    return model, X, y, df_clean

if __name__ == "__main__":
    model, X, y, df_clean = main()
