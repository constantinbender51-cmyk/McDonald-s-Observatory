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
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import warnings

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# --- Constants ---
GD_FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
CSV_FILE_NAME = '1m.csv' 

# --- Model & Data Parameters ---
LOOK_BACK = 24
TRAIN_SPLIT_RATIO = 0.7 
FEATURES = ['close', 'volume']
TARGET = 'direction' # NEW: Target is now directional
MAX_SAMPLES = 5000
# NEW: Threshold for defining an 'Up' movement (Classification Target)
MIN_DIRECTION_CHANGE_PCT = 0.001 

# --- Backtest Parameters ---
INITIAL_CAPITAL = 10000.0
# NEW: Probability threshold for making a trade (1=UP, 0=DOWN/STAY)
PREDICTION_THRESHOLD = 0.35

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

def create_sequences(data, raw_prices, look_back, min_change_pct):
    """
    Creates sequences of data (X) and directional targets (y) for the LSTM.
    y = 1 if price moves up by more than min_change_pct, else 0.
    """
    X, y = [], []
    
    # Raw prices are needed to calculate the direction for the target (y)
    # The prices must align with the scaled data used in X
    
    # We iterate over the scaled data (X)
    # We use -1 because we need the price at i + look_back to calculate the change.
    for i in range(len(data) - look_back - 1): 
        
        # X: The sequence of scaled feature data
        X.append(data[i:(i + look_back), :])
        
        # --- Calculate the price change percentage for the target (y) ---
        # Price at the beginning of the prediction period (i + look_back - 1)
        current_price = raw_prices[i + look_back - 1]
        # Price at the end of the prediction period (i + look_back)
        next_price = raw_prices[i + look_back] 
        
        price_change_pct = (next_price - current_price) / current_price
        
        # y: Directional label (1 for UP, 0 for DOWN/STAY)
        # We classify UP only if the change exceeds the threshold
        if price_change_pct > min_change_pct:
            # We predict the direction of the candle at index i + look_back
            y.append(1) # UP
        else:
            y.append(0) # DOWN or SIDWAYS
            
    return np.array(X), np.array(y)

def build_model(look_back, num_features):
    """
    Builds an LSTM model for Binary Classification (predicting direction).
    """
    print("Building LSTM model for DIRECTIONAL CLASSIFICATION...")
    model = Sequential()
    model.add(LSTM(
        50,  # 50 units
        input_shape=(look_back, num_features),
        return_sequences=False
    ))
    model.add(Dropout(0.2)) # Drop 20% of neurons to combat overfitting
    model.add(Dense(25, activation='relu'))
    # NEW: Single output neuron with sigmoid for binary classification probability
    model.add(Dense(1, activation='sigmoid')) 

    # NEW: Use binary_crossentropy loss and track accuracy
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def run_backtest(predictions_prob, test_data_raw, initial_capital, pred_threshold):
    """
    Runs a simple "long-only" trading simulation based on model's predicted probability.
    """
    print(f"\n--- Running Backtest Simulation (Prediction Threshold: {pred_threshold}) ---")
    capital = initial_capital
    shares = 0
    portfolio_values = []
    trade_count = 0
    
    # Prices aligned with the trading events. 
    # trade_prices[i] is the price at which the decision (based on prediction[i]) is executed.
    trade_prices = test_data_raw.iloc[1:len(predictions_prob) + 1]
    
    if len(predictions_prob) > len(trade_prices):
        predictions_prob = predictions_prob[:len(trade_prices)]
    elif len(trade_prices) > len(predictions_prob):
         trade_prices = trade_prices.iloc[:len(predictions_prob)]

    for i in range(len(predictions_prob) - 1):
        
        # predicted_prob_up is the probability that price will go up over the next period
        predicted_prob_up = predictions_prob[i][0]
        execution_price = trade_prices.iloc[i] 
        next_price = trade_prices.iloc[i+1] # Price at the end of the holding period

        action = "HOLD"

        # --- Strategy Logic (Trade at the Close/Execution Price) ---
        
        # BUY: If we are flat AND we predict a high probability of UP
        if predicted_prob_up > pred_threshold and shares == 0:
            shares = capital / execution_price
            capital = 0
            trade_count += 1
            action = "BUY"


        # SELL: If we are long AND we predict a low probability of UP (i.e., high prob of DOWN/SIDEWAYS)
        # We use 1 - pred_prob for conviction in a down move
        elif predicted_prob_up < (1.0 - pred_threshold) and shares > 0:
            # Liquidate position at the current execution price
            capital = shares * execution_price
            shares = 0
            trade_count += 1
            action = "SELL"
        
        # Print prediction log with delay for the first 1000 steps
        if i < 1000:
            # Calculate the actual return of holding for the next period for comparison
            actual_change_pct = (next_price - execution_price) / execution_price
            
            print(f"Time: {trade_prices.index[i]} | Exec Price: {execution_price:.2f} | Next Price: {next_price:.2f} | Pred Prob UP: {predicted_prob_up:.4f} | Actual Change: {actual_change_pct * 100:.2f}% | Action: {action}")
            time.sleep(0.1)

        # --- Portfolio Value Calculation ---
        # If long, the value changes with the price of the next bar
        if shares > 0:
            current_portfolio_value = shares * next_price
        else:
            current_portfolio_value = capital
            
        portfolio_values.append(current_portfolio_value)
        
    # Final liquidation
    final_price = trade_prices.iloc[-1]
    if shares > 0:
        final_capital = shares * final_price
    else:
        final_capital = capital
        
    # --- Performance Metrics ---
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
    
    # Check if we have portfolio values before calculating drawdown
    if portfolio_values:
        # Use the index from the trade prices for the portfolio series
        portfolio_series = pd.Series(portfolio_values, index=trade_prices.index[:len(portfolio_values)])
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
    print(f"Slicing to the last {MAX_SAMPLES} hours of data...")
    df_features_sliced = df_1h[FEATURES].tail(MAX_SAMPLES)

    # --- 4. Scale Data ---
    print("Scaling data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df_features_sliced)

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

    # Get the raw 'close' prices for the test set for backtesting
    # The first index of the raw prices used for the first prediction in X_test
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
        verbose=2
    )

    # --- 8. Evaluate Model (Classification) ---
    print("\n--- Model Evaluation ---")
    
    # Predict probabilities
    predictions_prob = model.predict(X_test)
    
    # Convert probabilities to classes (0 or 1) for accuracy calculation
    predictions_class = (predictions_prob > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions_class)
    
    print(f"Test Set Binary Cross-Entropy Loss (Lower is better): {history.history['val_loss'][-1]:.4f}")
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    
    # --- 9. Run Backtest ---
    run_backtest(predictions_prob, test_closes_raw, INITIAL_CAPITAL, PREDICTION_THRESHOLD)


if __name__ == "__main__":
    main()
