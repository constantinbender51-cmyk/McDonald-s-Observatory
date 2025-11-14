import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
# CCXT is the preferred library for fetching OHLCV data from exchanges like Binance
# Note: Network calls will fail in this sandboxed environment, but the logic is correct.
try:
    import ccxt
except ImportError:
    print("Warning: 'ccxt' library not found. Please install it with 'pip install ccxt' in a real environment.")

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
START_DATE = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d %H:%M:%S')
WINDOW_SIZE = 10  # Look-back window for features
EPOCHS = 5
BATCH_SIZE = 32
LAG = 1 # Predict the return 1 step ahead (next candle)

def fetch_binance_data(symbol, timeframe, start_date):
    """
    Attempts to fetch historical OHLCV data from Binance using CCXT.
    Falls back to generating mock data if the API call fails or CCXT is unavailable.
    """
    print(f"Attempting to fetch real data for {symbol} starting from {start_date}...")
    
    # Check if ccxt is available (it may be imported but still not callable in sandbox)
    if 'ccxt' not in globals():
        print("CCXT not available. Falling back to mock data.")
        return generate_mock_data()

    try:
        # Initialize the exchange client
        binance = ccxt.binance({
            'enableRateLimit': True,
            # In a real setup, you might need 'apiKey' and 'secret' for higher limits
        })

        # Convert human-readable date to epoch timestamp (milliseconds)
        since_timestamp = binance.parse8601(start_date)

        # Fetch data in batches (Binance limit is 1000 per call)
        all_ohlcv = []
        limit = 1000
        
        while True:
            # fetch_ohlcv returns [timestamp, open, high, low, close, volume]
            ohlcv = binance.fetch_ohlcv(symbol, timeframe, since_timestamp, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            
            # Update the start time for the next batch
            since_timestamp = ohlcv[-1][0] + 1  # Start from the next millisecond

            # Simple rate limit control (adjust as needed for real environment)
            # time.sleep(binance.rateLimit / 1000)
            
            # Stop after collecting a large sample for demonstration
            if len(all_ohlcv) > 5000:
                 break
        
        if not all_ohlcv:
            print(f"API call successful, but no data returned for {symbol}. Falling back to mock data.")
            return generate_mock_data()

        # Convert to Pandas DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"Successfully fetched {len(df)} candles.")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    except Exception as e:
        # This branch is expected to run in the sandbox environment
        print(f"API call failed (expected in sandbox): {e}. Falling back to mock data.")
        return generate_mock_data()

def generate_mock_data(n_rows=5000):
    """Generates synthetic OHLCV data for testing the backtesting logic."""
    np.random.seed(42)  # For reproducibility
    
    # Generate prices with some trend
    base_price = 10000
    price_movements = np.cumsum(np.random.normal(0, 0.5, n_rows))
    close = base_price + price_movements + np.random.normal(0, 10, n_rows)
    
    # Generate OHLC based on close
    open_price = close * (1 + np.random.uniform(-0.001, 0.001, n_rows))
    high = np.maximum(open_price, close) * (1 + np.random.uniform(0, 0.001, n_rows))
    low = np.minimum(open_price, close) * (1 - np.random.uniform(0, 0.001, n_rows))
    volume = np.random.randint(1000, 50000, n_rows)
    
    data = {
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }
    df = pd.DataFrame(data, index=pd.date_range(end=datetime.now(), periods=n_rows, freq='H'))
    
    # Introduce NaN and zero-volume points for robust data cleaning test
    df.iloc[100:102, df.columns.get_loc('Volume')] = 0 
    df.iloc[200:202, df.columns.get_loc('Close')] = np.nan
    
    return df.dropna()

def preprocess_and_feature_engineer(df, window_size=WINDOW_SIZE, lag=LAG):
    """
    Cleans data, generates technical indicators, normalizes, and creates sequences.
    """
    print("\n--- Data Preprocessing & Feature Engineering ---")

    # 1. Cleaning and Consistency Check
    df = df.copy()
    # Fill zero volume/price by taking the mean of previous/next non-zero value
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    if len(df) < window_size + lag + 1:
        print("Error: Not enough data after cleaning for feature engineering.")
        return None, None
        
    print(f"Data length after cleaning: {len(df)}")
    
    # 2. Feature Engineering (Technical Indicators)
    
    # Calculate Returns (Target Variable) - Log Return
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1)).shift(-lag)
    
    # Momentum: Relative Strength Index (RSI - 14 periods)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))

    # Volatility: Average True Range (ATR - 14 periods)
    high_low = df['High'] - df['Low']
    high_prev_close = np.abs(df['High'] - df['Close'].shift(1))
    low_prev_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # Trend: Simple Moving Average (SMA) Crossovers (50-period, 200-period)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['SMA_Diff'] = df['SMA_50'] - df['SMA_200']
    
    # Volume Change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Finalize features and target
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'ATR', 'SMA_Diff', 'Volume_Change']
    target = 'Log_Return'

    df.dropna(inplace=True) 
    print(f"Data length after indicator calculation (ready for train/test): {len(df)}")

    # 3. Normalization (MinMaxScaler is often used for LSTM inputs)
    scalers = {}
    
    # We only scale the features, not the log return target
    feature_data = df[features].values
    
    scalers['features'] = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scalers['features'].fit_transform(feature_data)

    target_data = df[target].values
    
    # 4. Sequence Creation for LSTM
    X, Y = [], []
    for i in range(len(scaled_features) - window_size - 1):
        # Input sequence (lookback window)
        X.append(scaled_features[i:i + window_size])
        # Target (the log return 'lag' steps after the sequence)
        Y.append(target_data[i + window_size])

    X, Y = np.array(X), np.array(Y)
    print(f"Shape of X (features sequence): {X.shape}")
    print(f"Shape of Y (target returns): {Y.shape}")
    
    # The 'df' used for backtesting needs to be aligned with X and Y length
    df_aligned = df.iloc[window_size:-1]
    
    return X, Y, df_aligned, scalers['features']

def build_model(input_shape):
    """Creates and compiles the LSTM Neural Network model."""
    print("\n--- Model Definition ---")
    model = Sequential([
        # First LSTM layer with a high number of units and return_sequences=True
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        
        # Second LSTM layer
        LSTM(units=100, return_sequences=False),
        Dropout(0.2),
        
        # Output layer (predicts a single continuous value: the log return)
        Dense(units=1) 
    ])

    # Using Adam optimizer and Mean Squared Error (MSE) loss for regression
    model.compile(optimizer='adam', loss='mse')
    print("Model compiled with MSE loss and Adam optimizer.")
    return model

def backtest_strategy(df_aligned, predictions):
    """
    Simulates a simple trading strategy: Go long if prediction > 0, short if prediction < 0.
    """
    print("\n--- Backtesting Strategy ---")
    
    # 1. Create Prediction Signal (1 for Long, -1 for Short)
    # This strategy is based on the predicted sign of the log return
    df_aligned['Signal'] = np.where(predictions.flatten() > 0, 1, -1)
    
    # 2. Calculate Strategy Returns
    # Strategy Return = Position * Actual Next Return
    df_aligned['Strategy_Return'] = df_aligned['Signal'] * df_aligned['Log_Return']
    
    # 3. Calculate Cumulative Returns
    # Total return if we just held the asset (Buy & Hold)
    df_aligned['Buy_Hold_Cumulative'] = (1 + df_aligned['Log_Return']).cumprod()
    
    # Total return for the NN strategy
    df_aligned['Strategy_Cumulative'] = (1 + df_aligned['Strategy_Return']).cumprod()

    # Metrics
    total_market_return = df_aligned['Buy_Hold_Cumulative'].iloc[-1] - 1
    total_strategy_return = df_aligned['Strategy_Cumulative'].iloc[-1] - 1
    
    # Calculate Annualized Volatility (assuming hourly data: 24*365 = 8760 periods/year)
    periods_per_year = 8760
    strategy_volatility = df_aligned['Strategy_Return'].std() * np.sqrt(periods_per_year)
    
    # Calculate Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
    # Sharpe Ratio = (Annualized Return - Risk-Free Rate) / Annualized Volatility
    annualized_strategy_return = (1 + total_strategy_return) ** (periods_per_year / len(df_aligned)) - 1
    sharpe_ratio = annualized_strategy_return / strategy_volatility if strategy_volatility != 0 else np.nan

    print("-" * 40)
    print(f"Total Market Return (Buy & Hold): {total_market_return:.2%}")
    print(f"Total Strategy Return (NN):       {total_strategy_return:.2%}")
    print(f"Strategy Sharpe Ratio (Annualized): {sharpe_ratio:.2f}")
    print("-" * 40)
    
    return df_aligned

def main():
    """Main execution function."""
    
    # 1. Data Retrieval (attempts real data, falls back to mock data)
    df = fetch_binance_data(SYMBOL, TIMEFRAME, START_DATE)
    
    if df is None or df.empty:
        print("Exiting: Failed to load data.")
        return

    # 2. Preprocessing & Feature Engineering
    X, Y, df_aligned, scaler = preprocess_and_feature_engineer(df)
    
    if X is None:
        return

    # 3. Train/Test Split (80% train, 20% test - simulating walk-forward for demonstration)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    df_test = df_aligned.iloc[train_size:]

    # 4. Model Definition and Training
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    print("\n--- Model Training ---")
    history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        shuffle=False # Crucial for time series data
    )
    
    print(f"Training completed. Final Loss: {history.history['loss'][-1]:.6f}")

    # 5. Prediction
    test_predictions = model.predict(X_test)
    
    # 6. Backtesting and Evaluation
    backtest_results = backtest_strategy(df_test, test_predictions)
    
    print("\nFinal backtest results are available in the 'backtest_results' DataFrame (not displayed here).")
    print("Check the 'Strategy_Cumulative' column to evaluate performance against the 'Buy_Hold_Cumulative'.")

if __name__ == '__main__':
    # Set logging level for TensorFlow to suppress warnings
    tf.get_logger().setLevel('ERROR')
    main()
