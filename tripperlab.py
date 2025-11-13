import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
from datetime import datetime

# --- 1. Data Fetching ---
def fetch_binance_data(symbol="BTCUSDT", interval="1d", start_str="2018-01-01"):
    """
    Fetches historical klines from Binance public API.
    Handles pagination to get data from start_date to present.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Convert start string to milliseconds timestamp
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    
    all_data = []
    limit = 1000  # Max limit per request
    
    print(f"Fetching data for {symbol} from {start_str}...")
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": limit
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if not isinstance(data, list):
                print("Error fetching data:", data)
                break
                
            if not data:
                break
                
            all_data.extend(data)
            
            # Update start_ts to the close time of the last candle + 1ms
            start_ts = data[-1][6] + 1
            
            # Check if we reached the current time (approx)
            if len(data) < limit:
                break
                
            # Rate limit respect
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Connection error: {e}")
            break
            
    print(f"Total candles fetched: {len(all_data)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        "Open_time", "Open", "High", "Low", "Close", "Volume", 
        "Close_time", "Quote_vol", "Trades", "Taker_base", "Taker_quote", "Ignore"
    ])
    
    # Convert types
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms")
    
    return df[["Open_time", "Open", "High", "Low", "Close", "Volume"]]

# --- 2. Indicator Calculation ---
def calculate_indicators(df):
    """
    Calculates MACD and Stochastic RSI.
    """
    df = df.copy()
    
    # MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd_line - signal_line
    
    # Stochastic RSI (14, 3, 3)
    # 1. Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 2. Calculate Stoch RSI
    min_rsi = rsi.rolling(window=14).min()
    max_rsi = rsi.rolling(window=14).max()
    stoch = (rsi - min_rsi) / (max_rsi - min_rsi)
    
    # 3. Smooth K (User requested "Stockistic RSI", usually implies the K line)
    # Using a simple moving average of 3 for smoothing
    df['Stoch_RSI'] = stoch.rolling(window=3).mean().fillna(0)
    
    # Handle NaNs created by rolling windows
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

# --- 3. Feature Engineering ---
def create_features(df, window_size=10):
    """
    Creates the feature set:
    - Basic Features: 10 days of Price, Volume, MACD Hist, Stoch RSI
    - Polynomials: Power 2 to 10 for MACD Hist and Stoch RSI
    - Interactions: F1*F2, F1*F3, etc.
    """
    print("Generating features...")
    features = []
    targets = []
    
    # We need to predict the direction 10 days into the future
    # Target = 1 if Price[i+10] > Price[i], else 0
    prediction_horizon = 10
    
    # Loop through data, leaving room for window at start and target at end
    for i in range(window_size, len(df) - prediction_horizon):
        
        # --- A. Basic Features (Windows) ---
        # 1. Price (Close)
        f1_price = df['Close'].iloc[i-window_size:i].values
        # 2. Volume
        f2_volume = df['Volume'].iloc[i-window_size:i].values
        # 3. MACD Histogram
        f3_macd = df['MACD_Hist'].iloc[i-window_size:i].values
        # 4. Stochastic RSI
        f4_stoch = df['Stoch_RSI'].iloc[i-window_size:i].values
        
        row_features = []
        
        # Add Basic Features
        row_features.extend(f1_price)
        row_features.extend(f2_volume)
        row_features.extend(f3_macd)
        row_features.extend(f4_stoch)
        
        # --- B. Polynomial Features (Powers 2 to 10) ---
        # "Basic feature 3 squared... to 10"
        for p in range(2, 11):
            row_features.extend(np.power(f3_macd, p))
            
        # "Basic feature 4 squared... to 10"
        for p in range(2, 11):
            row_features.extend(np.power(f4_stoch, p))
            
        # --- C. Interaction Features ---
        # Element-wise multiplication of the vectors
        row_features.extend(f1_price * f2_volume)  # F1 * F2
        row_features.extend(f1_price * f3_macd)    # F1 * F3
        row_features.extend(f1_price * f4_stoch)   # F1 * F4
        row_features.extend(f2_volume * f3_macd)   # F2 * F3
        row_features.extend(f2_volume * f4_stoch)  # F2 * F4
        
        features.append(row_features)
        
        # --- Target ---
        # 1 if price in 10 days is higher than current price (at index i-1, the last known point)
        current_price = df['Close'].iloc[i-1]
        future_price = df['Close'].iloc[i-1 + prediction_horizon]
        target = 1 if future_price > current_price else 0
        targets.append(target)
        
    return np.array(features), np.array(targets), df.iloc[window_size : len(df) - prediction_horizon]

# --- Main Execution ---

# 1. Fetch Data
df = fetch_binance_data()

# 2. Add Indicators
df = calculate_indicators(df)

# 3. Create Dataset
X, y, df_sim = create_features(df)

print(f"Feature Matrix Shape: {X.shape}")
print(f"Target Array Shape: {y.shape}")

# 4. Split Data
total_samples = len(X)
train_idx = int(total_samples * 0.70)
test1_idx = int(total_samples * 0.85)

X_train = X[:train_idx]
y_train = y[:train_idx]

X_test1 = X[train_idx:test1_idx]
y_test1 = y[train_idx:test1_idx]
# Corresponding price data for the simulation
# We need the 'Open' prices for the NEXT day to simulate entering trades
# df_sim contains the data at the time of prediction. 
# We need to align this with future price movements for backtesting.
sim_data_test1 = df_sim.iloc[train_idx:test1_idx].copy()

# Note: Test Set 2 (test1_idx to end) is ignored as requested.

# 5. Train Model
# Scaling is crucial for Logistic Regression, especially with Price (~60k) vs StochRSI (0-1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test1_scaled = scaler.transform(X_test1)

print("Training Logistic Regression...")
model = LogisticRegression(max_iter=2000) # Increased iter for convergence
model.fit(X_train_scaled, y_train)

# 6. Prediction on Test Set 1
probs = model.predict_proba(X_test1_scaled)[:, 1] # Probability of Class 1 (Up)
predictions = (probs > 0.5).astype(int)

# Accuracy
acc = accuracy_score(y_test1, predictions)
print(f"\nTest Set 1 Accuracy (10-day direction): {acc:.4f}")

# 7. Capital Development Simulation
# Strategy:
# Long if Prob > 0.5 + Threshold (0.05) -> > 0.55
# Short if Prob < 0.5 - Threshold (0.05) -> < 0.45
# Entry: Next Open.
# Exit: We re-evaluate daily. If signal persists, we hold. If signal flips or goes neutral, we close.
# Since we simulate "Daily Rebalancing", the return is simply the daily % change applied to capital if in position.

initial_capital = 1000.0
capital = [initial_capital]
threshold = 0.05

# Align simulation data
# df_sim index i corresponds to X_test1[i]
# X_test1[i] is prediction made at Close of day D
# Trade enters at Open of day D+1 and holds until Open of day D+2
# We need to access the raw dataframe to get future Open prices
original_df_idx = sim_data_test1.index
next_opens = df.loc[original_df_idx + 1, 'Open'].values
following_opens = df.loc[original_df_idx + 2, 'Open'].values

# We iterate through the test set
position = 0 # 1 for Long, -1 for Short, 0 for Cash
dates = []

for i in range(len(probs) - 2): # -2 because we look 2 days ahead for return calc
    
    prob = probs[i]
    
    # Determine Signal
    if prob > (0.5 + threshold):
        signal = 1 # Long
    elif prob < (0.5 - threshold):
        signal = -1 # Short
    else:
        signal = 0 # Neutral/Cash
        
    # Calculate Return for holding from Next Open to Following Open
    entry_price = next_opens[i]
    exit_price = following_opens[i]
    
    if signal == 1:
        # Long Return
        pct_change = (exit_price - entry_price) / entry_price
        capital.append(capital[-1] * (1 + pct_change))
    elif signal == -1:
        # Short Return (Inverse)
        # If price goes down, we make money.
        pct_change = (entry_price - exit_price) / entry_price
        capital.append(capital[-1] * (1 + pct_change))
    else:
        # No trade, capital stays same
        capital.append(capital[-1])
        
    dates.append(sim_data_test1.iloc[i]['Open_time'])

final_capital = capital[-1]
print(f"Initial Capital: ${initial_capital:.2f}")
print(f"Final Capital on Test Set 1: ${final_capital:.2f}")
print(f"Total Return: {((final_capital - initial_capital) / initial_capital) * 100:.2f}%")
