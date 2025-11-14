import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import time
import json
import http.server
import socketserver
import os 

# Assume CCXT is installed and functional in this production environment.
try:
    import ccxt
except ImportError:
    print("FATAL ERROR: Required library 'ccxt' not found. Please install it.")
    raise

    
# --- CORE CONFIGURATION (Based on User Requirements) ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
START_DATE = '2018-01-01 00:00:00' 
WINDOW_SIZE = 72 
EPOCHS = 5
BATCH_SIZE = 32
LAG = 48 # Predict 2 days (48 hours) ahead
RESULTS_FILE = 'backtest_results.json'
HTML_FILE = 'backtest_visualization.html'
PORT = 8080 # Corrected Port
# --- Data Retrieval & Model Functions ---

def fetch_binance_data(symbol, timeframe, start_date):
    """
    Fetches historical OHLCV data from Binance using CCXT, starting from 2018-01-01.
    If the API call fails, the function will return an empty DataFrame, halting the process.
    """
    print(f"Attempting to fetch real data for {symbol} ({timeframe}) starting from {start_date}...")
    
    try:
        if 'ccxt' not in globals():
             # This check remains as a final guard against import failure, though unlikely now.
             raise ImportError("ccxt library is not available in environment globals.")
             
        binance = ccxt.binance({'enableRateLimit': True})
        since_timestamp = binance.parse8601(start_date)

        all_ohlcv = []
        limit = 1000 
        
        while True:
            # Fetch 1000 candles at a time, moving forward from the last timestamp
            ohlcv = binance.fetch_ohlcv(symbol, timeframe, since_timestamp, limit=limit)
            
            if not ohlcv: 
                print("No more data found.")
                break
                
            all_ohlcv.extend(ohlcv)
            since_timestamp = ohlcv[-1][0] + 1 
            
            print(f"Fetched {len(all_ohlcv)} candles so far...")
            
        if not all_ohlcv:
            print(f"API call completed, but returned no data between {start_date} and current time.")
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"Successfully fetched a total of {len(df)} candles.")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    except Exception as e:
        # Production-level logging of a runtime failure
        print(f"RUNTIME ERROR: Binance API call failed during data fetch.")
        print(f"Ensure network connectivity and API stability. Exception: {e}")
        # Return an empty DataFrame, causing main() to exit cleanly.
        return pd.DataFrame()


def preprocess_and_feature_engineer(df, window_size=WINDOW_SIZE, lag=LAG):
    """
    Cleans data, generates technical indicators, normalizes, and creates sequences.
    """
    print("\n--- Data Preprocessing & Feature Engineering ---")

    df = df.copy()
    # 1. Initial cleaning of input OHLCV
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    if len(df) < window_size + lag + 1:
        print(f"Error: Not enough data ({len(df)}) after cleaning.")
        return None, None, None, None
        
    # Calculate Log Return (Target Variable)
    # --- CORRECTION: Swap numerator/denominator to ensure correct log return sign ---
    df['Log_Return'] = np.log(df['Close'].shift(-lag) / df['Close']) 
    # ---------------------------------------------------------------------------------
    df.dropna(subset=['Log_Return'], inplace=True)
    
    # Calculate Features (Indicators)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # Calculate RS. This step will generate inf if loss is 0. (As desired by the user's dropout mechanism)
    RS = gain / loss 
    df['RSI'] = 100 - (100 / (1 + RS)) # RSI will be calculated from inf if RS is inf

    high_low = df['High'] - df['Low']
    high_prev_close = np.abs(df['High'] - df['Close'].shift(1))
    low_prev_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['SMA_Diff'] = df['SMA_50'] - df['SMA_200']
    
    # Volume Change: Can generate inf if previous volume was 0.
    df['Volume_Change'] = df['Volume'].pct_change()
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'ATR', 'SMA_Diff', 'Volume_Change']
    target = 'Log_Return'

    # CRITICAL: Convert ALL infinite values (from Log_Return, RSI, Volume_Change, etc.) to NaN
    # This prepares the data for complete row dropout.
    df.replace([np.inf, -np.inf], np.nan, inplace=True) 
    
    # Drop all rows that now contain NaN. This removes the rows where inf values were generated.
    df.dropna(inplace=True) 
    print(f"Data length after indicator calculation and clean dropout: {len(df)}")

    # 3. Normalization
    feature_data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(feature_data)

    target_data = df[target].values
    
    # 4. Sequence Creation for LSTM
    X, Y = [], []
    for i in range(len(scaled_features) - window_size):
        X.append(scaled_features[i:i + window_size])
        Y.append(target_data[i + window_size]) 

    X, Y = np.array(X), np.array(Y)
    df_aligned = df.iloc[window_size:]
    
    return X, Y, df_aligned, scaler

def build_model(input_shape):
    """Creates and compiles the LSTM Neural Network model."""
    print("\n--- Model Definition ---")
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=100, return_sequences=False),
        Dropout(0.2),
        Dense(units=1) 
    ])
    model.compile(optimizer='adam', loss='mse')
    print("Model compiled.")
    return model

def backtest_strategy(df_aligned, predictions):
    """
    Simulates a simple trading strategy and returns performance metrics + cumulative series.
    """
    print("\n--- Backtesting Strategy ---")
    
    df_aligned['Signal'] = np.where(predictions.flatten() > 0, 1, -1)
    df_aligned['Strategy_Return'] = df_aligned['Signal'] * df_aligned['Log_Return']

    # Log returns are additive. Cumulative return is exp(cumsum(log_returns)), 
    # which starts at 1.0 (0% return).
    df_aligned['Buy_Hold_Cumulative'] = np.exp(df_aligned['Log_Return'].cumsum())
    df_aligned['Strategy_Cumulative'] = np.exp(df_aligned['Strategy_Return'].cumsum())

    # Metrics calculation now correctly uses the cumulative series
    total_market_return = df_aligned['Buy_Hold_Cumulative'].iloc[-1] - 1
    total_strategy_return = df_aligned['Strategy_Cumulative'].iloc[-1] - 1
    
    periods_per_year = 8760 / LAG 
    strategy_volatility = df_aligned['Strategy_Return'].std() * np.sqrt(periods_per_year)
    annualized_strategy_return = (1 + total_strategy_return) ** (periods_per_year / len(df_aligned)) - 1
    sharpe_ratio = annualized_strategy_return / strategy_volatility if strategy_volatility != 0 else np.nan
    
    cumulative_max = df_aligned['Strategy_Cumulative'].cummax()
    drawdown = (cumulative_max - df_aligned['Strategy_Cumulative']) / cumulative_max
    max_drawdown = drawdown.max()
    
    print("-" * 40)
    print(f"Timeframe: {TIMEFRAME} | Lookback: {WINDOW_SIZE}h | Prediction: {LAG}h ahead")
    print(f"Total Strategy Return (NN):       {total_strategy_return:.2%}")
    print(f"Strategy Sharpe Ratio (Annualized): {sharpe_ratio:.2f}")
    print("-" * 40)
    
    results_data = {
        'metadata': {
            'symbol': SYMBOL,
            'timeframe': TIMEFRAME,
            'lookback_h': WINDOW_SIZE,
            'prediction_h': LAG,
            'epochs': EPOCHS,
            'start_date': df_aligned.index[0].strftime('%Y-%m-%d %H:%M:%S'),
            'end_date': df_aligned.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            'total_periods': len(df_aligned)
        },
        'metrics': {
            'total_market_return': total_market_return,
            'total_strategy_return': total_strategy_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        },
        'chart_data': {
            'timestamps': [ts.isoformat() for ts in df_aligned.index],
            'buy_hold': df_aligned['Buy_Hold_Cumulative'].tolist(),
            'strategy_return': df_aligned['Strategy_Cumulative'].tolist()
        }
    }
    
    return results_data

# --- Server Function ---

def start_web_server():
    """Starts a simple HTTP server to serve the visualization page."""
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    # We serve the current directory where the HTML and JSON files are
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("-" * 50)
        print(f"🚀 Server started successfully on port {PORT}")
        print(f"Access the backtest report at: http://localhost:{PORT}/{HTML_FILE}")
        print("-" * 50)
        # Keep the server running indefinitely
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

def main():
    """Main execution function."""
    
    # 1. Data Retrieval
    df = fetch_binance_data(SYMBOL, TIMEFRAME, START_DATE)
    if df is None or df.empty: 
        print("Script aborted due to missing or failed data retrieval.")
        return

    # 2. Preprocessing & Feature Engineering
    X, Y, df_aligned, scaler = preprocess_and_feature_engineer(df)
    if X is None or len(X) == 0: 
        print("Script aborted because not enough clean data was available after feature engineering.")
        return

    # 3. Train/Test Split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    df_test = df_aligned.iloc[train_size:]
    if len(X_test) == 0:
        print("Error: Test set is empty.")
        return

    # 4. Model Training
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    print("\n--- Model Training ---")
    model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        shuffle=False 
    )
    
    # 5. Prediction & Backtesting
    test_predictions = model.predict(X_test)
    results_data = backtest_strategy(df_test, test_predictions)
    
    # 6. Export Results to JSON for Web Visualization
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results_data, f, indent=4)
        
    print(f"\nSimulation complete. Results exported to '{RESULTS_FILE}'.")
    
    # 7. Start Web Server
    start_web_server()

if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    main()
