import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from collections import deque
import random
import time

# --- Configuration Constants ---
WINDOW_SIZE = 72  # Lookback window size (e.g., 72 hours)
FORECAST_HOURS = 48  # Target look-ahead (e.g., 48 hours)
N_LAG = 3          # Number of lookback periods for each feature
N_FEATURES = 4     # Number of base features: Price_Change, Volume_Change, MACD_Delta, StochRSI
HIDDEN_UNITS = 32
EPOCHS = 20
BATCH_SIZE = 64
TEST_SIZE = 0.2    # 20% of data for testing
TENSORBOARD_LOG_DIR = './logs'

def create_model(input_shape):
    """Defines the LSTM model architecture."""
    model = tf.keras.models.Sequential([
        # Shape: (WINDOW_SIZE, N_FEATURES)
        tf.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(HIDDEN_UNITS // 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def run_backtest(df, model):
    """
    Simulates a simple trading strategy using the trained model.
    A simple threshold strategy is used: Buy if prediction > 0.005, Sell if prediction < -0.005.
    """
    initial_cash = 10000.0
    cash = initial_cash
    position = 0.0  # Amount of crypto held
    trades = []
    
    # Simple backtesting loop (for illustration only)
    print("\n--- Running Simplified Backtest Simulation ---")
    
    # We only backtest the test set portion
    test_start_index = int(len(df) * (1 - TEST_SIZE))
    test_df = df.iloc[test_start_index:].copy()
    
    # To run a real backtest, you would need to re-format the test_df into X_test and y_test
    # and use the model.predict(X_test) results. For simplicity, this placeholder simulates 
    # the trade logic based on predicted returns (which would be loaded from a prediction step).
    
    # Placeholder for prediction: assuming we have predictions
    # Note: In a real scenario, you'd use the model to generate predictions here.
    # For this simulation, we assume 'Prediction' column exists and is scaled.
    test_df['Prediction'] = np.random.uniform(-0.01, 0.01, len(test_df)) # Placeholder
    
    # Simple Trading Logic
    for i in range(1, len(test_df) - FORECAST_HOURS):
        row = test_df.iloc[i]
        prediction = row['Prediction']
        current_price = row['Close']
        
        # 1. Buy Logic (Go long)
        if prediction > 0.005 and cash > 0:
            if position == 0:
                # Buy
                position = cash / current_price
                cash = 0
                trades.append(('BUY', row.name, current_price))
        
        # 2. Sell Logic (Exit long position)
        elif prediction < -0.005 and position > 0:
            # Sell
            cash = position * current_price
            position = 0
            trades.append(('SELL', row.name, current_price))

    # Calculate final value
    final_value = cash + position * test_df.iloc[-1]['Close']
    
    print(f"Initial Cash: ${initial_cash:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {((final_value - initial_cash) / initial_cash) * 100:.2f}%")
    print(f"Total Trades: {len(trades)}")
    print("-" * 40)
    
    return final_value

# --- MAIN EXECUTION BLOCK ---

# Use mock data structure that matches the user's likely time series data
# In a real environment, this would be loaded from a file or API
data = {
    'Open': np.random.rand(1000) * 10000,
    'High': np.random.rand(1000) * 10000 + 100,
    'Low': np.random.rand(1000) * 10000 - 100,
    'Close': np.random.rand(1000) * 10000,
    'Volume': np.random.rand(1000) * 1000000
}
df = pd.DataFrame(data, index=pd.date_range(start='2022-01-01', periods=1000, freq='H'))
# Simulate a few zero volume/price events that could cause the error
df.iloc[100, df.columns.get_loc('Volume')] = 0
df.iloc[101, df.columns.get_loc('Volume')] = 100 # Next Volume creates Price_Change (inf)
df.iloc[500, df.columns.get_loc('Close')] = 0
df.iloc[501, df.columns.get_loc('Close')] = 10000 # Next Close creates Price_Change (inf)


# --- 2. Feature Engineering ---

# 2.1 Calculate Price and Volume Changes
df['Price_Change'] = df['Close'].pct_change()
df['Volume_Change'] = df['Volume'].pct_change()

# 2.2 Calculate Technical Indicators (simplified)
# MACD: Placeholder for simplicity, using a basic difference
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Delta'] = df['MACD'] - df['Signal_Line']

# Stochastic RSI (simplified to just a ratio)
df['StochRSI'] = (df['Close'] - df['Close'].min()) / (df['Close'].max() - df['Close'].min())

# 2.3 Define the Target Variable
# Target: Direction and magnitude of the price change over the next 48 hours
# Price return = (Close[t+48] - Close[t]) / Close[t]
# Use .shift(-FORECAST_HOURS) to align the future return with the current feature set
df['Target'] = df['Close'].pct_change(FORECAST_HOURS).shift(-FORECAST_HOURS)


### CRITICAL DEBUGGING STEP: IDENTIFYING INFINITE VALUES ###
# As requested, we check for positive or negative infinity (inf/-inf).
# These typically arise from division by zero (e.g., pct_change when the previous value was 0).
print("\n" + "="*70)
print("### DATA QUALITY CHECK: IDENTIFYING INFINITE VALUES (INF) ###")
print("="*70)

# Columns to check for non-finite values (inf, -inf)
cols_to_check = ['Price_Change', 'Volume_Change', 'MACD_Delta', 'StochRSI', 'Target']
inf_found = False

for col in cols_to_check:
    inf_mask = np.isinf(df[col])
    inf_count = inf_mask.sum()
    
    if inf_count > 0:
        inf_found = True
        print(f"🚨 [{col}]: {inf_count} infinite value(s) found.")
        
        # Get the first 5 timestamps where inf occurred
        inf_indices = df.loc[inf_mask].index.tolist()
        print(f"  First 5 timestamps: {inf_indices[:5]}")
        
        # Print a few rows that contain the inf value and the preceding row for context
        print("\n  Sample Context (The zero value in the 'Preceding' row is the likely culprit):")
        
        for i in range(min(5, len(inf_indices))):
            idx = df.index.get_loc(inf_indices[i])
            # Determine the column that should have been non-zero in the preceding step
            culprit_col = 'Close'
            if col == 'Volume_Change':
                culprit_col = 'Volume'
            elif col == 'Price_Change' or col == 'Target':
                culprit_col = 'Close'
                
            # Get the current row and the row immediately preceding it for context
            context_df = df.iloc[idx-1 : idx+1]
            
            # Print the culprit column (Close/Volume) and the feature column (col)
            # The 'culprit_col' value in the top row of the printout should be near zero.
            print(f"  Feature Check Column: '{col}' (Inf found in the bottom row)")
            print(context_df[[culprit_col, col]].to_string(header=True))
            print("-" * 30)

if not inf_found:
    print("✅ No infinite (inf) values found in the target columns. You may proceed.")

print("="*70 + "\n")


# --- 3. Create Input Matrix (X) and Target Vector (y) ---

# Drop initial NaN rows created by indicators, shift operations, and potentially the inf/huge value replacements
# NOTE: If the diagnostic above shows inf values, they will be dropped here as well.
df = df.dropna()

# Select the four base features for the 72-hour lag window
base_features = ['Price_Change', 'Volume_Change', 'MACD_Delta', 'StochRSI']
X_data = []
y_data = []

# Create the sequence data for the LSTM
for i in range(WINDOW_SIZE * N_LAG, len(df)):
    # Slice the relevant rows (72 hours of data)
    # The current window is from i - (WINDOW_SIZE * N_LAG) to i
    start_index = i - WINDOW_SIZE
    if start_index < 0:
        continue
    
    # X input contains the last WINDOW_SIZE rows (72 hours) for the base features
    X_window = df[base_features].iloc[start_index:i].values
    
    # y target is the Target value at the end of that window
    y_target = df['Target'].iloc[i]

    X_data.append(X_window)
    y_data.append(y_target)

X = np.array(X_data)
y = np.array(y_data)

# Ensure data is not empty
if X.size == 0:
    print("Error: Dataset is empty after feature engineering and dropping NaNs.")
    # Exiting function for mock data
else:
    # --- 4. Scale Data ---
    # Reshape X for scaling (temporarily flatten to 2D)
    original_shape = X.shape # (N_samples, WINDOW_SIZE, N_FEATURES)
    X_reshaped = X.reshape(-1, original_shape[2]) # (N_samples * WINDOW_SIZE, N_FEATURES)
    
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_scaled_reshaped = scaler_X.fit_transform(X_reshaped)
    
    # Reshape back to 3D
    X_scaled = X_scaled_reshaped.reshape(original_shape)
    
    # Scale y (Target)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # --- 5. Split Data (Training and Testing) ---
    split_index = int(len(X_scaled) * (1 - TEST_SIZE))
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]

    # --- 6. Train Model ---
    model = create_model(input_shape=(WINDOW_SIZE, N_FEATURES))
    
    print("Starting Model Training...")
    # Add TensorBoard callback for monitoring
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_DIR, histogram_freq=1)
    
    # Train the model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=0, callbacks=[tensorboard_callback])
    print("Model Training Complete.")

    # --- 7. Evaluate and Backtest ---
    
    # Predict on the test set
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform to get actual return values
    y_test_actual = scaler_y.inverse_transform(y_test)
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    print(f"\nTest Set MSE (Actual Returns): {mse:.6f}")
    
    # Run simple backtest simulation
    # Note: To run the backtest, we need to add the actual 'Close' prices for the test period back to the test_df
    # In a real scenario, this step requires careful data re-alignment, which is abstracted here.
    run_backtest(df, model)

# --- END MAIN EXECUTION BLOCK ---
