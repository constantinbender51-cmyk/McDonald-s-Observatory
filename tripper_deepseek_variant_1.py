import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def fetch_bitcoin_data_binance():
    """Fetch Bitcoin daily price data from Binance"""
    print("Fetching Bitcoin data from Binance...")
    
    # Initialize Binance client (no API key needed for public data)
    client = Client()
    
    # Fetch daily BTC/USDT data from January 1, 2018
    klines = client.get_historical_klines(
        symbol="BTCUSDT",
        interval=Client.KLINE_INTERVAL_1DAY,
        start_str="1 Jan, 2018"
    )
    
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore']
    
    df = pd.DataFrame(klines, columns=columns)
    
    # Convert types and set index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Convert price and volume to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # Keep only necessary columns and rename to match previous structure
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    print(f"Fetched {len(df)} days of data")
    return df

def calculate_technical_indicators(df):
    """Calculate technical indicators for the dataset"""
    # Calculate daily percentage changes (for the lookback window)
    df['daily_price_pct'] = df['Close'].pct_change()
    df['daily_volume_pct'] = df['Volume'].pct_change()
    
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Stochastic RSI (14, 14, 3, 3)
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic RSI
    rsi_min = df['rsi'].rolling(window=14).min()
    rsi_max = df['rsi'].rolling(window=14).max()
    df['stoch_rsi'] = (df['rsi'] - rsi_min) / (rsi_max - rsi_min)
    df['stoch_rsi_k'] = df['stoch_rsi'].rolling(window=3).mean()
    
    return df

def create_target_variable(df, period=7):
    """Create target variable: direction of price change over next 7 days"""
    # Calculate future price change
    future_return = df['Close'].pct_change(period).shift(-period)
    
    # Create binary target (1 if price goes up, 0 if down)
    df['target'] = (future_return > 0).astype(int)
    return df

def create_3d_features(df, lookback=10):
    """
    Create 3D feature matrix with shape (n_samples, 4, lookback)
    Each sample has 4 features over 10 days lookback period
    """
    features_list = []
    targets_list = []
    
    # Features to include in the lookback window
    feature_columns = ['daily_price_pct', 'daily_volume_pct', 'macd_histogram', 'stoch_rsi_k']
    
    for i in range(lookback, len(df) - 7):  # -7 for the 7-day target
        # Extract lookback window for each feature
        feature_matrix = []
        for feature in feature_columns:
            feature_values = df[feature].iloc[i-lookback:i].values
            feature_matrix.append(feature_values)
        
        # Stack to create (4, lookback) matrix
        feature_matrix = np.array(feature_matrix)
        
        # Only include if no NaN values
        if not np.any(np.isnan(feature_matrix)) and not np.isnan(df['target'].iloc[i]):
            features_list.append(feature_matrix)
            targets_list.append(df['target'].iloc[i])
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(targets_list)
    
    return X, y

def main():
    # Step 1: Fetch data from Binance
    btc_data = fetch_bitcoin_data_binance()
    
    # Step 2: Calculate indicators
    btc_data = calculate_technical_indicators(btc_data)
    
    # Step 3: Create target variable
    btc_data = create_target_variable(btc_data, period=7)
    
    # Step 4: Create 3D features with 10-day lookback
    X, y = create_3d_features(btc_data, lookback=10)
    
    print(f"\nDataset shape: {X.shape}")  # Should be (n_samples, 4, 10)
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{pd.Series(y).value_counts(normalize=True)}")
    
    # Reshape X from 3D to 2D for logistic regression
    # From (n_samples, 4, 10) to (n_samples, 4 * 10) = (n_samples, 40)
    X_2d = X.reshape(X.shape[0], -1)
    
    # Step 5: Split data (80% train, 20% test) - without shuffling to maintain time order
    split_idx = int(0.8 * len(X_2d))
    X_train, X_test = X_2d[:split_idx], X_2d[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Step 6: Train logistic regression
    print("\nTraining logistic regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Step 7: Make predictions and calculate accuracy
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Feature matrix shape: {X.shape}")
    
    # Baseline accuracy (always predicting the majority class)
    baseline_accuracy = max(np.mean(y_test), 1 - np.mean(y_test))
    print(f"Baseline accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()
