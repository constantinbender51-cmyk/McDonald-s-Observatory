import os
import time # NEW: For logging delay
import io
import gdown
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# --- Constants ---
GD_FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
CSV_FILE_NAME = '1m.csv' 

# --- Model & Data Parameters ---
LOOK_BACK = 48
# NEW: Adjusted train/test split
TRAIN_SPLIT_RATIO = 0.7 
FEATURES = ['close', 'volume']
TARGET = 'close'
# NEW: Use only the last 5000 hours of data
MAX_SAMPLES = 5000

# --- Backtest Parameters ---
INITIAL_CAPITAL = 10000.0
MIN_PRICE_CHANGE_PCT = 0.001 # 0.1% threshold

def download_and_load_data(file_id, csv_name):
    """
    Downloads the CSV file from Google Drive and loads it into a pandas DataFrame.
    """
    print(f"Downloading data from Google Drive (ID: {file_id})...")
    try:
        gdown.download(id=file_id, output=csv_name, quiet=False)
        print(f"Download complete. Reading '{csv_name}'...")

        df = pd.read_csv(
            csv_name,
            names=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            header=0 
        )
        
        print("Data loaded successfully.")
        os.remove(csv_name)
        return df

    except Exception as e:
        print(f"Error during data loading: {e}")
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
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('datetime')
        print(f"Original data range: {df.index.min()} to {df.index.max()}")
        
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

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
        df_1h['close'] = df_1h['close'].ffill()
        df_1h['volume'] = df_1h['volume'].fillna(0)
        for col in ['open', 'high', 'low']:
            df_1h[col] = df_1h[col].fillna(df_1h['close'])
            
        df_1h = df_1h.dropna()
        
        # --- 4. Use float32 to save memory ---
        df_1h = df_1h.astype(np.float32)

        print(f"Resampled data shape: {df_1h.shape}")
        print(f"Resampled data range: {df_1h.index.min()} to {df_1h.index.max()}")
        return df_1h

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("Please check your CSV columns and date/time format.")
        return None

def create_sequences(data, look_back):
    """
    Creates sequences of data for the LSTM.
    """
    X, y = [], []
    # We predict the 'close' price, which is the 0th column in our `data`
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back, 0]) 
    return np.array(X), np.array(y)

def build_model(look_back, num_features):
    """
    Builds an LSTM model with Dropout to prevent overfitting.
    """
    print("Building LSTM model...")
    model = Sequential()
    model.add(LSTM(
        50,  # 50 units
        input_shape=(look_back, num_features),
        return_sequences=False
    ))
    model.add(Dropout(0.2)) # Drop 20% of neurons to combat overfitting
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1)) 

    model.compile(optimizer='adam', loss='mean_squared_error')
    print(model.summary())
    return model

def run_backtest(predictions, test_data_raw, initial_capital, min_change_pct):
    """
    Runs a simple "long-only" trading simulation based on model predictions.
    Now includes a minimum change threshold and prediction logging.
    """
    print(f"\n--- Running Backtest Simulation (Min Change: {min_change_pct * 100}%) ---")
    capital = initial_capital
    shares = 0
    portfolio_values = []
    trade_count = 0

    if len(predictions) > len(test_data_raw):
        predictions = predictions[:len(test_data_raw)]
    elif len(test_data_raw) > len(predictions):
         test_data_raw = test_data_raw[:len(predictions)]

    for i in range(len(predictions) - 1):
        current_price = test_data_raw[i]
        predicted_next_price = predictions[i][0]
        
        predicted_change_pct = (predicted_next_price - current_price) / current_price
        action = "HOLD"

        # --- Strategy Logic ---
        
        # BUY: If we are flat AND we predict a rise above the threshold
        if predicted_change_pct > min_change_pct and shares == 0:
            shares = capital / current_price
            capital = 0
            trade_count += 1
            action = "BUY"


        # SELL: If we are long AND we predict a drop below the negative threshold
        elif predicted_change_pct < -min_change_pct and shares > 0:
            capital = shares * current_price
            shares = 0
            trade_count += 1
            action = "SELL"
        
        # NEW: Print prediction log with delay for the first 1000 steps
        if i < 1000:
            print(f"Time: {test_data_raw.index[i]} | Actual: {current_price:.2f} | Pred: {predicted_next_price:.2f} | Pct Change: {predicted_change_pct * 100:.2f}% | Action: {action}")
            time.sleep(0.1)

        # --- Portfolio Value Calculation ---
        if shares > 0:
            current_portfolio_value = shares * current_price
        else:
            current_portfolio_value = capital
            
        portfolio_values.append(current_portfolio_value)

    # If still holding shares at the end, liquidate
    if shares > 0:
        capital = shares * test_data_raw.iloc[-1]
        
    final_capital = capital

    # --- Performance Metrics ---
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
    
    # Check if we have portfolio values before calculating drawdown
    if portfolio_values:
        # Use the index from the raw test data for the portfolio series
        portfolio_series = pd.Series(portfolio_values, index=test_data_raw.index[:len(portfolio_values)])
        cumulative_max = portfolio_series.cummax()
        # Handle division by zero
        drawdown = (portfolio_series - cumulative_max) / cumulative_max.replace(0, 1) 
        max_drawdown_pct = drawdown.min() * 100
    else:
        max_drawdown_pct = 0.0

    print(f"\n--- Final Backtest Results ---")
    print(f"Total Trades Executed: {trade_count}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital:   ${final_capital:,.2f}")
    print(f"Total Return:    {total_return_pct:,.2f}%")
    print(f"Max Drawdown:    {max_drawdown_pct:,.2f}%")
    
    # Buy and Hold comparison
    if not test_data_raw.empty and test_data_raw.iloc[0] != 0:
        buy_hold_return = ((test_data_raw.iloc[-1] - test_data_raw.iloc[0]) / test_data_raw.iloc[0]) * 100
        print(f"Buy & Hold Return: {buy_hold_return:,.2f}%")
    else:
        print("Buy & Hold Return: N/A (Insufficient test data or zero price)")


def main():
    # --- 1. Load Data ---
    df_raw = download_and_load_data(GD_FILE_ID, CSV_FILE_NAME)
    if df_raw is None:
        return

    # --- 2. Preprocess Data ---
    df_1h = preprocess_data(df_raw)
    if df_1h is None:
        return
        
    # --- 3. Slice Data to MAX_SAMPLES ---
    # We take the last 5000 hours of the clean, 1H data
    print(f"Slicing to the last {MAX_SAMPLES} hours of data...")
    df_features_sliced = df_1h[FEATURES].tail(MAX_SAMPLES)

    # --- 4. Scale Data ---
    print("Scaling data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df_features_sliced)

    scaler_close = MinMaxScaler(feature_range=(0, 1))
    scaler_close.fit(df_features_sliced[['close']]) # Use sliced data for scaler fit

    # --- 5. Create Sequences ---
    print(f"Creating sequences with look-back of {LOOK_BACK} hours...")
    X, y = create_sequences(data_scaled, LOOK_BACK)
    
    if len(X) == 0:
        print("Error: No sequences created. Is data shorter than look_back?")
        return

    # --- 6. Train/Test Split ---
    split_index = int(len(X) * TRAIN_SPLIT_RATIO)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Get the raw 'close' prices for the test set for backtesting
    # We take the raw prices corresponding to the X_test indices
    raw_close_prices_all = df_features_sliced['close']
    test_start_index = len(raw_close_prices_all) - len(y_test)
    test_closes_raw = raw_close_prices_all.iloc[test_start_index:]
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes:  X={X_test.shape}, y={y_test.shape}")
    print(f"Raw test closes for backtest: {test_closes_raw.shape}")

    # --- 7. Build & Train Model ---
    num_features = len(FEATURES)
    model = build_model(LOOK_BACK, num_features)

    print("Training model (this may take a while)...")
    history = model.fit(
        X_train,
        y_train,
        epochs=20,       
        batch_size=64,   
        validation_data=(X_test, y_test),
        shuffle=False,   
        verbose=2
    )

    # --- 8. Evaluate Model ---
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
    
    # --- 9. Run Backtest ---
    run_backtest(predictions_inv, test_closes_raw, INITIAL_CAPITAL, MIN_PRICE_CHANGE_PCT)


if __name__ == "__main__":
    main()
