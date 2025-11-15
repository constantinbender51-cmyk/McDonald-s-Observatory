import os
import io
import gdown
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# --- Constants ---
# Google Drive File ID from your link
GD_FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
CSV_FILE_NAME = '1m.csv' # The file is a direct CSV

# --- Model & Data Parameters ---
# Lookback window (in hours). You mentioned 48 or 480.
# 48 is faster and uses less memory.
LOOK_BACK = 48
TRAIN_SPLIT_RATIO = 0.8
FEATURES = ['close', 'volume']
TARGET = 'close'

# --- Backtest Parameters ---
INITIAL_CAPITAL = 10000.0

def download_and_load_data(file_id, csv_name):
    """
    Downloads the CSV file from Google Drive and loads it into a pandas DataFrame.
    The file is assumed to have 6 columns: [timestamp, open, high, low, close, volume].
    """
    print(f"Downloading data from Google Drive (ID: {file_id})...")
    try:
        # Download the CSV file directly
        gdown.download(id=file_id, output=csv_name, quiet=False)
        print(f"Download complete. Reading '{csv_name}'...")

        # Read the CSV file, explicitly naming 6 columns
        # The first column is now assumed to be the full datetime stamp
        df = pd.read_csv(
            csv_name,
            names=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            header=0 # Set to 0 if headers are present, None if not
        )
        
        print("Data loaded successfully.")
        # Clean up the downloaded CSV file
        os.remove(csv_name)
        return df

    except Exception as e:
        print(f"Error during data loading: {e}")
        # Clean up if download failed partially
        if os.path.exists(csv_name):
            os.remove(csv_name)
        return None

def preprocess_data(df):
    """
    Resamples 1-minute data to 1-hour and handles missing values.
    """
    print("Preprocessing data...")
    try:
        # --- 1. Convert Timestamp and Set Index ---
        # The 'timestamp' column is converted to datetime objects
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('datetime')
        print(f"Original data range: {df.index.min()} to {df.index.max()}")
        
        # Select numeric columns for resampling and ensure they are numeric
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop the original timestamp column
        df = df.drop(columns=['timestamp'])
        
        # --- 2. Resample to 1 Hour ---
        print("Resampling data to 1-hour timeframe...")
        resample_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        df_1h = df.resample('1H').apply(resample_rules)

        # --- 3. Handle Missing Data ---
        # Resampling creates NaNs for hours with no trades (weekends, etc.)
        # We fill 'close' forward, and fill 'volume' with 0
        df_1h['close'] = df_1h['close'].ffill()
        df_1h['volume'] = df_1h['volume'].fillna(0)
        # Fill other OHLC values from the last 'close'
        for col in ['open', 'high', 'low']:
            df_1h[col] = df_1h[col].fillna(df_1h['close'])
            
        # Drop any remaining NaNs (e.g., at the very beginning)
        df_1h = df_1h.dropna()
        
        # --- 4. Use float32 to save memory ---
        df_1h = df_1h.astype(np.float32)

        print(f"Resampled data shape: {df_1h.shape}")
        print(f"Resampled data range: {df_1h.index.min()} to {df_1h.index.max()}")
        return df_1h

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("Please check your CSV columns and date/time format.")
        print("Expected columns: 'timestamp', 'open', 'high', 'low', 'close', 'volume'")
        return None

def create_sequences(data, look_back):
    """
    Creates sequences of data for the LSTM.
    X shape: [samples, look_back, features]
    y shape: [samples, 1] (predicting the next 'close')
    """
    X, y = [], []
    # We predict the 'close' price, which is the 0th column in our `data`
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back, 0]) # 0 is the index of 'close'
    return np.array(X), np.array(y)

def build_model(look_back, num_features):
    """
    Builds a simple, memory-efficient LSTM model.
    """
    print("Building LSTM model...")
    model = Sequential()
    model.add(LSTM(
        50,  # 50 units
        input_shape=(look_back, num_features),
        return_sequences=False # Only output at the end of the sequence
    ))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1)) # Output layer: 1 neuron for 'close' prediction

    model.compile(optimizer='adam', loss='mean_squared_error')
    print(model.summary())
    return model

def run_backtest(predictions, test_data_raw, initial_capital):
    """
    Runs a simple "long-only" trading simulation based on model predictions.
    """
    print("\n--- Running Backtest Simulation ---")
    capital = initial_capital
    shares = 0
    portfolio_values = []

    # Ensure predictions and test data align
    # We predict for t+1 at time t.
    
    # test_data_raw should be the 'close' prices from the test set
    if len(predictions) > len(test_data_raw):
        print("Warning: Predictions array is longer than test data. Trimming.")
        predictions = predictions[:len(test_data_raw)]
    elif len(test_data_raw) > len(predictions):
         print(f"Warning: Test data is longer than predictions. Trimming test data.")
         test_data_raw = test_data_raw[:len(predictions)]


    for i in range(len(predictions) - 1):
        current_price = test_data_raw[i]
        predicted_next_price = predictions[i][0]

        # --- Strategy Logic ---
        # If we predict price will go up and we are flat, buy.
        if predicted_next_price > current_price and shares == 0:
            shares = capital / current_price
            capital = 0
            # print(f"BUY @ {current_price:.2f}")

        # If we predict price will go down and we are long, sell.
        elif predicted_next_price < current_price and shares > 0:
            capital = shares * current_price
            shares = 0
            # print(f"SELL @ {current_price:.2f}")

        # --- Portfolio Value Calculation ---
        if shares > 0:
            current_portfolio_value = shares * current_price
        else:
            current_portfolio_value = capital
            
        portfolio_values.append(current_portfolio_value)

    # If still holding shares at the end, liquidate
    if shares > 0:
        capital = shares * test_data_raw[-1]
        
    final_capital = capital

    # --- Performance Metrics ---
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
    
    portfolio_series = pd.Series(portfolio_values, index=test_data_raw.index[:len(portfolio_values)])
    
    # Calculate Drawdown
    cumulative_max = portfolio_series.cummax()
    drawdown = (portfolio_series - cumulative_max) / cumulative_max
    max_drawdown_pct = drawdown.min() * 100

    print(f"\n--- Backtest Results ---")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital:   ${final_capital:,.2f}")
    print(f"Total Return:    {total_return_pct:,.2f}%")
    print(f"Max Drawdown:    {max_drawdown_pct:,.2f}%")
    
    # Buy and Hold comparison
    buy_hold_return = ((test_data_raw.iloc[-1] - test_data_raw.iloc[0]) / test_data_raw.iloc[0]) * 100
    print(f"Buy & Hold Return: {buy_hold_return:,.2f}%")


def main():
    # --- 1. Load Data ---
    df_raw = download_and_load_data(GD_FILE_ID, CSV_FILE_NAME)
    if df_raw is None:
        return

    # --- 2. Preprocess Data ---
    df_1h = preprocess_data(df_raw)
    if df_1h is None:
        return
        
    df_features = df_1h[FEATURES]

    # --- 3. Scale Data ---
    print("Scaling data...")
    # Scale all features
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df_features)

    # We need a separate scaler for 'close' to inverse-transform predictions
    scaler_close = MinMaxScaler(feature_range=(0, 1))
    scaler_close.fit(df_features[['close']])

    # --- 4. Create Sequences ---
    print(f"Creating sequences with look-back of {LOOK_BACK} hours...")
    X, y = create_sequences(data_scaled, LOOK_BACK)
    
    if len(X) == 0:
        print("Error: No sequences created. Is data shorter than look_back?")
        return

    # --- 5. Train/Test Split ---
    split_index = int(len(X) * TRAIN_SPLIT_RATIO)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Get the raw 'close' prices for the test set for backtesting
    # We need the 'close' column from the *original* 1H dataframe
    # The test data starts after the split_index AND the look_back
    test_start_index = split_index + LOOK_BACK
    test_closes_raw = df_1h['close'].iloc[test_start_index:]
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes:  X={X_test.shape}, y={y_test.shape}")
    print(f"Raw test closes for backtest: {test_closes_raw.shape}")

    # --- 6. Build & Train Model ---
    num_features = len(FEATURES)
    model = build_model(LOOK_BACK, num_features)

    print("Training model (this may take a while)...")
    history = model.fit(
        X_train,
        y_train,
        epochs=20,       # Keep low for speed/memory
        batch_size=64,   # 64 is a good balance
        validation_data=(X_test, y_test),
        shuffle=False,   # DO NOT shuffle time-series data
        verbose=2
    )

    # --- 7. Evaluate Model ---
    print("\n--- Model Evaluation ---")
    predictions_scaled = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    predictions_inv = scaler_close.inverse_transform(predictions_scaled)
    y_test_inv = scaler_close.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
    mae = mean_absolute_error(y_test_inv, predictions_inv)
    
    print(f"Test Set RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"Test Set MAE (Mean Absolute Error):     {mae:.4f}")
    
    # --- 8. Run Backtest ---
    run_backtest(predictions_inv, test_closes_raw, INITIAL_CAPITAL)


if __name__ == "__main__":
    main()
