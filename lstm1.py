import os
import time # For logging delay
import io
import gdown
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
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
TRAIN_SPLIT_RATIO = 0.7 
FEATURES = ['close', 'volume', 'macd', 'macd_signal']
TARGET = 'direction' 
MAX_SAMPLES = 5000
MIN_DIRECTION_CHANGE_PCT = 0.001 

# --- Backtest Parameters ---
INITIAL_CAPITAL = 10000.0
PREDICTION_THRESHOLD = 0.55 

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
    Resamples 1-minute data to 1-hour, calculates technical indicators, and handles missing values.
    """
    print("Preprocessing data and calculating technical indicators...")
    try:
        # --- 1. Convert Timestamp and Set Index ---
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('datetime')
        
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.drop(columns=['timestamp'])
        
        # --- 2. Resample to 1 Hour ---
        resample_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        df_1h = df.resample('1H').apply(resample_rules)

        # --- 3. Handle Missing Data (Intermediate) ---
        df_1h['close'] = df_1h['close'].ffill()
        df_1h['volume'] = df_1h['volume'].fillna(0)
        for col in ['open', 'high', 'low']:
            df_1h[col] = df_1h[col].fillna(df_1h['close'])
            
        # --- 4. Calculate MACD (12, 26, 9) ---
        df_1h['EMA_12'] = df_1h['close'].ewm(span=12, adjust=False).mean()
        df_1h['EMA_26'] = df_1h['close'].ewm(span=26, adjust=False).mean()
        df_1h['macd'] = df_1h['EMA_12'] - df_1h['EMA_26']
        df_1h['macd_signal'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
        
        df_1h = df_1h.drop(columns=['EMA_12', 'EMA_26'])
        
        # --- 5. Final Data Cleaning and Type Conversion ---
        df_1h = df_1h.dropna()
        df_1h = df_1h.astype(np.float32)

        print(f"Resampled and feature-engineered data shape: {df_1h.shape}")
        return df_1h

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("Please check your CSV columns and date/time format.")
        return None

def create_sequences(data, raw_prices, look_back, min_change_pct):
    """
    Creates sequences of data (X) and directional targets (y) for the LSTM.
    y = 1 if price moves up by more than min_change_pct, else 0.
    """
    X, y = [], []
    
    for i in range(len(data) - look_back - 1): 
        
        # X: The sequence of scaled feature data for the last 'look_back' periods
        X.append(data[i:(i + look_back), :])
        
        # --- Calculate the price change percentage for the target (y) ---
        current_price = raw_prices[i + look_back - 1]
        next_price = raw_prices[i + look_back] 
        
        price_change_pct = (next_price - current_price) / current_price
        
        # y: Directional label (1 for UP, 0 for DOWN/STAY)
        if price_change_pct > min_change_pct:
            y.append(1) # UP
        else:
            y.append(0) # DOWN or SIDWAYS
            
    return np.array(X), np.array(y)

def build_model(look_back, num_features):
    """
    Builds an LSTM model for Binary Classification (predicting direction).
    """
    print("Building LSTM model for DIRECTIONAL CLASSIFICATION...")
    print(f"Input Shape: (Lookback={look_back}, Features={num_features})")
    model = Sequential()
    model.add(LSTM(
        50,  
        input_shape=(look_back, num_features),
        return_sequences=False
    ))
    model.add(Dropout(0.2)) # Drop 20% of neurons
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) 

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def run_backtest(predictions_prob, test_data_raw, initial_capital, pred_threshold):
    """
    UPDATED: Runs a fixed 1-hour trade simulation: 
    If P(UP) > threshold, BUY at current close and SELL at next close.
    """
    print(f"\n--- Running Backtest Simulation (FIXED 1-HOUR TRADE: Threshold={pred_threshold}) ---")
    capital = initial_capital
    portfolio_values = []
    trade_count = 0
    
    # trade_prices is aligned so trade_prices.iloc[i] is the entry price,
    # and trade_prices.iloc[i+1] is the liquidation price one hour later.
    trade_prices = test_data_raw.iloc[1:len(predictions_prob) + 1]
    
    if len(predictions_prob) > len(trade_prices):
        predictions_prob = predictions_prob[:len(trade_prices)]
    elif len(trade_prices) > len(predictions_prob):
         trade_prices = trade_prices.iloc[:len(predictions_prob)]

    for i in range(len(predictions_prob) - 1):
        
        # predicted_prob_up is the probability that price will go up over the next period
        predicted_prob_up = predictions_prob[i][0]
        entry_price = trade_prices.iloc[i] 
        exit_price = trade_prices.iloc[i+1] # Price at the close of the next bar

        action = "SKIP"
        trade_return_pct = 0.0

        # --- Strategy Logic (Trade at the Close/Execution Price) ---
        
        # BUY and immediately calculate return for the next hour
        if predicted_prob_up > pred_threshold:
            # Calculate the return achieved over the next hour (entry at entry_price, exit at exit_price)
            trade_return_pct = (exit_price - entry_price) / entry_price
            
            # Apply the return (positive or negative) to the entire capital base
            capital *= (1.0 + trade_return_pct)
            trade_count += 1
            action = "BUY & SELL (1H)"

        # Print prediction log with delay for the first 1000 steps
        if i < 1000:
            
            print(f"Time: {trade_prices.index[i]} | Entry Price: {entry_price:.2f} | Exit Price: {exit_price:.2f} | Pred Prob UP: {predicted_prob_up:.4f} | Trade Return: {trade_return_pct * 100:+.2f}% | Action: {action}")
            time.sleep(0.1)

        # Update portfolio value at the end of the bar (after the 1H trade is complete or skipped)
        portfolio_values.append(capital)
        
    final_capital = capital
        
    # --- Performance Metrics ---
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
    
    if portfolio_values:
        portfolio_series = pd.Series(portfolio_values, index=trade_prices.index[:len(portfolio_values)])
        cumulative_max = portfolio_series.cummax()
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

    # --- 2. Preprocess Data and Feature Engineer ---
    df_1h = preprocess_data(df_raw)
    if df_1h is None:
        return
        
    # --- 3. Slice Data to MAX_SAMPLES ---
    print(f"Slicing to the last {MAX_SAMPLES} hours of data.")
    df_features_sliced = df_1h.tail(MAX_SAMPLES)

    # --- 4. Scale Data ---
    print(f"Scaling data using features: {FEATURES}...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df_features_sliced[FEATURES])

    # Get the raw 'close' prices for target creation and backtesting
    raw_close_prices_all = df_features_sliced['close'].values

    # --- 5. Create Sequences ---
    print(f"Creating sequences with look-back of {LOOK_BACK} hours and directional target...")
    X, y = create_sequences(data_scaled, raw_close_prices_all, LOOK_BACK, MIN_DIRECTION_CHANGE_PCT)
    
    if len(X) == 0:
        print("Error: No sequences created. Is data shorter than look_back?")
        return

    # --- 6. Train/Test Split ---
    split_index = int(len(X) * TRAIN_SPLIT_RATIO)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Calculate Class Weights for Imbalance Correction
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(weights))
    print(f"Calculated Class Weights: {class_weights}")
    
    # Get the raw 'close' prices for the test set for backtesting
    test_start_index = len(raw_close_prices_all) - len(y) + split_index
    test_closes_raw = df_features_sliced['close'].iloc[test_start_index:]
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes:  X={X_test.shape}, y={y_test.shape}")
    print(f"Raw test closes for backtest: {test_closes_raw.shape}")

    # --- 7. Build & Train Model (Classification) ---
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
        verbose=2,
        class_weight=class_weights
    )

    # --- 8. Evaluate Model (Classification) ---
    print("\n--- Model Evaluation ---")
    
    predictions_prob = model.predict(X_test)
    predictions_class = (predictions_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, predictions_class)
    
    print(f"Test Set Binary Cross-Entropy Loss (Lower is better): {history.history['val_loss'][-1]:.4f}")
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    
    # --- 9. Run Backtest ---
    run_backtest(predictions_prob, test_closes_raw, INITIAL_CAPITAL, PREDICTION_THRESHOLD)


if __name__ == "__main__":
    main()
