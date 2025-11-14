import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from binance.client import Client, BinanceAPIException
from binance.enums import KLINE_INTERVAL_1HOUR

# --- Configuration ---
# Since this is a public endpoint, no API keys are required for market data access
TICKER = 'BTCUSDT'
START_DATE = '1 Jan, 2018'
INTERVAL = KLINE_INTERVAL_1HOUR
LOOKBACK_HOURS = 72    # Window size for features (L)
FORECAST_HOURS = 48    # Prediction window for target (F)
INPUT_FEATURES = 4     # Price change, Volume change, MACD delta, StochRSI
TOTAL_FEATURES = LOOKBACK_HOURS * INPUT_FEATURES # 72 * 4 = 288

# --- 1. Data Fetching (Binance API) ---
def fetch_binance_klines(symbol, interval, start_str):
    """
    Fetches historical OHLCV data directly from Binance using the python-binance library.
    No API key is needed for this public endpoint.
    """
    print(f"Connecting to Binance to fetch {symbol} {interval} data from {start_str}...")
    try:
        # Initialize client without API key/secret
        client = Client("", "")
        
        # get_historical_klines handles the necessary pagination for long date ranges
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str
        )
        
        # Convert klines to a Pandas DataFrame
        df = pd.DataFrame(klines, columns=[
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
            'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
            'Taker Buy Quote Asset Volume', 'Ignore'
        ])

        # Drop unnecessary columns and set the index
        df = df.drop(columns=['Close Time', 'Quote Asset Volume', 'Number of Trades', 
                              'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
        
        # Convert Open Time from milliseconds to datetime object
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df = df.set_index('Open Time')
        df.index.name = 'timestamp'
        
        # Convert OHLCV columns to float
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except BinanceAPIException as e:
        print(f"Binance API Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during data fetch: {e}")
        return pd.DataFrame()


df = fetch_binance_klines(TICKER, INTERVAL, START_DATE)

if df.empty or len(df) < LOOKBACK_HOURS + FORECAST_HOURS:
    print(f"Cannot proceed without sufficient data. Fetched {len(df)} rows.")
    exit()

print(f"Successfully fetched {len(df)} rows for {TICKER}. Starting feature engineering.")

# --- 2. Feature Engineering and Target Definition ---

# 2.1 Technical Indicators using pandas_ta
# Indicator 3: MACD Line minus Signal Line
df.ta.macd(append=True) 
# Indicator 4: Stochastic RSI
df.ta.stochrsi(append=True) 

# 2.2 Calculate instantaneous changes and deltas (L=1)
# Feature 1: Hourly Price Change
df['Price_Change'] = df['Close'].pct_change()

# Feature 2: Hourly Volume Change
df['Volume_Change'] = df['Volume'].pct_change()

# Feature 3: MACD Delta (MACD Line - Signal Line)
df['MACD_Delta'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']

# Feature 4: Stochastic RSI (%K line)
# STOCHRSIk_14_14_3_3 is the %K line with default parameters
df['StochRSI'] = df['STOCHRSIk_14_14_3_3']

# 2.3 Define the Target Variable
# Target: Direction and magnitude of the price change over the next 48 hours
# Price return = (Close[t+48] - Close[t]) / Close[t]
# Use .shift(-FORECAST_HOURS) to align the future return with the current feature set
df['Target'] = df['Close'].pct_change(FORECAST_HOURS).shift(-FORECAST_HOURS)


# --- 3. Create Input Matrix (X) and Target Vector (y) ---

# Drop initial NaN rows created by indicators and shift operations
df = df.dropna()

# Select the four base features for the 72-hour lag window
base_features = ['Price_Change', 'Volume_Change', 'MACD_Delta', 'StochRSI']
X_data = []

# Loop to create the 72-hour lagged features (288 total features)
# We must iterate from LOOKBACK_HOURS to ensure the window is full
for i in range(LOOKBACK_HOURS, len(df)):
    # Slice the data for the 72-hour lookback window
    window = df.iloc[i - LOOKBACK_HOURS : i][base_features].values
    
    # Flatten the 72x4 window into a 288-element vector
    X_data.append(window.flatten())

# Align the target vector with the start of the feature matrix (from index LOOKBACK_HOURS)
X = np.array(X_data)
y = df['Target'].iloc[LOOKBACK_HOURS:].values.reshape(-1, 1)

print(f"Final Feature matrix X shape: {X.shape}")
print(f"Final Target vector y shape: {y.shape}")

# --- 4. Pre-processing and Train/Test Split ---

# Split the data chronologically (no shuffling)
# Using 80/20 split for training/testing
split_index = int(X.shape[0] * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Feature Scaling: Pre-process features to be in the range [-1, 1]
# Scaler fitted ONLY on the training data to avoid data leakage
feature_scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# Scale the target as well for stable NN training
target_scaler = MinMaxScaler(feature_range=(-1, 1))
y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled = target_scaler.transform(y_test)

print(f"Training set size: {X_train_scaled.shape[0]}, Test set size: {X_test_scaled.shape[0]}")


# --- 5. Neural Network Architecture and Training ---

# Design the Neural Network: 2 hidden layers with 8 neurons each
model = Sequential([
    # Input Layer (288 features) -> Hidden Layer 1
    Dense(8, activation='relu', input_shape=(TOTAL_FEATURES,)),
    # Hidden Layer 2
    Dense(8, activation='relu'),
    # Output Layer: Single neuron for regression (direction and magnitude)
    Dense(1, activation='linear') 
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
print("\nTraining Neural Network (10 epochs)...")
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=10, 
    batch_size=32,
    verbose=0, # Set to 1 or 2 to see training progress
    validation_data=(X_test_scaled, y_test_scaled)
)
print("Training complete.")

# --- 6. Prediction and Accuracy Calculation ---

# Predict on the test set (scaled predictions)
scaled_predictions = model.predict(X_test_scaled, verbose=0)

# Inverse transform to get actual return predictions (unscaled)
unscaled_predictions = target_scaler.inverse_transform(scaled_predictions)

# Calculate Prediction Accuracy (Correct Direction)
# The prediction is correct if the sign of the prediction matches the sign of the actual target
correct_direction = np.sign(unscaled_predictions) == np.sign(y_test)
accuracy = np.mean(correct_direction)

print(f"\nModel Prediction Accuracy (Correct Direction): {accuracy * 100:.2f}%")

# --- 7. Trading Strategy Simulation (Backtesting) ---

# Extract prediction signals and corresponding actual returns
signals = np.sign(unscaled_predictions).flatten()
actual_returns = y_test.flatten()

# Simulation Parameters
START_CAPITAL = 1000.0
EQUITY = [START_CAPITAL]
period_returns = []
current_equity = START_CAPITAL

# We need the dates corresponding to the trade entries for the equity curve
# The index of y_test corresponds to the entry date for the 48-hour return
entry_dates = df.iloc[LOOKBACK_HOURS:].index[split_index : split_index + len(signals)]

# Simulate the trading strategy
for i in range(len(signals)):
    signal = signals[i]
    future_return = actual_returns[i]
    
    # Strategy Logic: Buy/Short and Hold for 48 hours (based on prediction)
    
    # Calculate the return for this 48-hour holding period
    return_multiplier = 0 # Default to no trade (0 return)
    if signal > 0:
        # Predicted UP: Buy (Long) -> actual return is the gain/loss
        return_multiplier = future_return
    elif signal < 0:
        # Predicted DOWN: Sell (Short) -> gain if price drops, loss if price rises
        return_multiplier = -future_return 

    # Calculate the change in equity and track it
    equity_change = current_equity * return_multiplier
    current_equity += equity_change
    
    EQUITY.append(current_equity)
    period_returns.append(return_multiplier)

# Create a time series for the equity curve
# Note: EQUITY has one extra value (START_CAPITAL), so we slice [1:]
equity_curve = pd.Series(EQUITY[1:], index=entry_dates)
returns_series = pd.Series(period_returns, index=entry_dates)

# --- 8. Calculate Performance Metrics (Sharpe Ratio and Max Drawdown) ---

def calculate_max_drawdown(equity_series):
    """Calculates the Maximum Drawdown from an equity curve."""
    # Calculate the running maximum (high water mark)
    running_max = equity_series.cummax()
    # Calculate the drawdown relative to the running maximum
    drawdown = (equity_series - running_max) / running_max
    return drawdown.min()

def calculate_sharpe_ratio(returns_series, holding_period_days=2, risk_free_rate=0.02):
    """
    Calculates the Annualized Sharpe Ratio, assuming zero risk-free rate for crypto.
    We annualize based on the 48-hour return period (2 days).
    Periods per year = 365.25 / holding_period_days
    """
    periods_per_year = 365.25 / holding_period_days
    excess_returns = returns_series - risk_free_rate / periods_per_year
    
    # Calculate annualized Sharpe Ratio
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

# Calculate the metrics
max_drawdown = calculate_max_drawdown(equity_curve)
sharpe_ratio = calculate_sharpe_ratio(returns_series)

# --- 9. Output Results ---

print("\n" + "="*50)
print(f"         TRADING STRATEGY BACKTEST RESULTS ({TICKER})")
print("="*50)
print(f"Start Capital: ${START_CAPITAL:.2f}")
print(f"End Equity:    ${current_equity:.2f}")
print(f"Total Return:  {((current_equity / START_CAPITAL) - 1) * 100:.2f}%")
print("-" * 50)
print(f"Annualized Sharpe Ratio:            {sharpe_ratio:.3f}")
print(f"Maximum Drawdown:                   {max_drawdown * 100:.2f}%")
print("="*50)

# Print Equity Development (Sample)
print("\nSample of Equity Development (Start of Test Period):")
print(equity_curve.head(5).to_string())
print("\nSample of Equity Development (End of Test Period):")
print(equity_curve.tail(5).to_string())
