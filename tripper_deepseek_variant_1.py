import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import time
warnings.filterwarnings('ignore')

# Step 1: Fetch Bitcoin OHLCV data from Binance
print("Fetching Bitcoin data from Binance...")
time.sleep(0.1)

def fetch_bitcoin_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1d',
        'startTime': int(pd.Timestamp('2018-01-01').timestamp() * 1000),
        'limit': 1000
    }
    
    all_data = []
    while True:
        response = requests.get(url, params=params)
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        # Set next startTime
        params['startTime'] = data[-1][0] + 1
        
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert to proper data types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df[['open', 'high', 'low', 'close', 'volume']]

# Fetch data
btc_data = fetch_bitcoin_data()
print(f"Fetched {len(btc_data)} days of data")
time.sleep(0.1)

# Step 2: Calculate technical indicators
print("Calculating technical indicators...")
time.sleep(0.1)

# Price change percentages (24 days)
for i in range(1, 25):
    btc_data[f'price_change_{i}'] = btc_data['close'].pct_change(periods=i)

# Volume change percentages (24 days)
for i in range(1, 25):
    btc_data[f'volume_change_{i}'] = btc_data['volume'].pct_change(periods=i)

# MACD (12, 26, 9)
exp12 = btc_data['close'].ewm(span=12, adjust=False).mean()
exp26 = btc_data['close'].ewm(span=26, adjust=False).mean()
macd_line = exp12 - exp26
signal_line = macd_line.ewm(span=9, adjust=False).mean()
macd_histogram = macd_line - signal_line

# Add MACD histogram for 24 days
for i in range(24):
    btc_data[f'macd_hist_{i+1}'] = macd_histogram.shift(i)

# Stochastic RSI
def calculate_stoch_rsi(close, period=14):
    # RSI calculation
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic RSI
    stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    return stoch_rsi

stoch_rsi = calculate_stoch_rsi(btc_data['close'])

# Add Stochastic RSI for 24 days
for i in range(24):
    btc_data[f'stoch_rsi_{i+1}'] = stoch_rsi.shift(i)

# Step 3: Create target variable (price direction after 14 days)
btc_data['future_price'] = btc_data['close'].shift(-14)
btc_data['price_change_14d'] = (btc_data['future_price'] - btc_data['close']) / btc_data['close']
btc_data['target'] = (btc_data['price_change_14d'] > 0).astype(int)

# Step 4: Prepare features and target
feature_columns = (
    [f'price_change_{i}' for i in range(1, 25)] +
    [f'volume_change_{i}' for i in range(1, 25)] +
    [f'macd_hist_{i}' for i in range(1, 25)] +
    [f'stoch_rsi_{i}' for i in range(1, 25)]
)

# Create feature matrix and target vector
X = btc_data[feature_columns].copy()
y = btc_data['target'].copy()

# Remove rows with NaN values
valid_indices = ~(X.isna().any(axis=1) | y.isna())
X = X[valid_indices]
y = y[valid_indices]

print(f"Total samples after cleaning: {len(X)}")
time.sleep(0.1)

# Step 5: Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

print(f"Training samples: {len(X_train)}")
time.sleep(0.1)
print(f"Testing samples: {len(X_test)}")
time.sleep(0.1)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
time.sleep(0.1)

# Step 7: Trading Strategy with 0.8% Threshold (No Stop Loss)
print("\nImplementing trading strategy with 0.8% threshold...")
time.sleep(0.1)
print("="*60)
time.sleep(0.1)

def calculate_drawdown(capital_history):
    peak = capital_history[0]
    max_dd = 0
    for capital in capital_history:
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd

def simulate_strategy(y_pred_proba, test_prices):
    """Simulate trading with 0.8% threshold"""
    # Fixed 0.8% threshold
    threshold = 0.8 / 100.0
    upper_threshold = 0.5 + threshold  # 0.508
    lower_threshold = 0.5 - threshold  # 0.492
    
    capital = 10000
    capital_history = [capital]
    trades = 0
    
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(len(y_pred_proba)):
        current_price = test_prices.iloc[i]
        prediction_prob = y_pred_proba[i]
        
        # Determine position based on 0.8% threshold
        if prediction_prob >= upper_threshold:
            position = 1  # Long
            trades += 1
        elif prediction_prob <= lower_threshold:
            position = -1  # Short
            trades += 1
        else:
            position = 0  # Flat
        
        # Calculate daily return based on current position
        if i < len(y_pred_proba) - 1:
            next_price = test_prices.iloc[i + 1]
            daily_return = (next_price - current_price) / current_price
            
            # Update capital based on position
            if position == 1:  # Long
                capital *= (1 + daily_return)
            elif position == -1:  # Short
                capital *= (1 - daily_return)
            # Flat position: capital remains unchanged
        
        capital_history.append(capital)
    
    max_drawdown = calculate_drawdown(capital_history)
    total_return = (capital - 10000) / 10000 * 100
    market_exposure = trades / len(y_pred_proba) * 100
    
    return capital, max_drawdown, total_return, trades, market_exposure

# Get test prices
test_dates = X_test.index
test_prices = btc_data.loc[test_dates, 'close']

# Simulate strategy with 0.8% threshold
capital, max_dd, total_return, trades, exposure = simulate_strategy(y_pred_proba, test_prices)

print("\n" + "="*60)
time.sleep(0.1)
print("FINAL TRADING STRATEGY RESULTS")
time.sleep(0.1)
print("="*60)
time.sleep(0.1)

print(f"\n📊 MODEL PERFORMANCE:")
time.sleep(0.1)
print(f"   Accuracy: {accuracy*100:.2f}%")
time.sleep(0.1)

print(f"\n🎯 TRADING STRATEGY PARAMETERS:")
time.sleep(0.1)
print(f"   Prediction Threshold: 0.8%")
time.sleep(0.1)
print(f"   Go Long if prediction ≥ {0.5 + 0.8/100:.3f}")
time.sleep(0.1)
print(f"   Go Short if prediction ≤ {0.5 - 0.8/100:.3f}")
time.sleep(0.1)
print(f"   Stay Flat if prediction between {0.5 - 0.8/100:.3f} and {0.5 + 0.8/100:.3f}")
time.sleep(0.1)

print(f"\n💰 TRADING RESULTS:")
time.sleep(0.1)
print(f"   Start Capital: $10,000")
time.sleep(0.1)
print(f"   Final Capital: ${capital:,.2f}")
time.sleep(0.1)
print(f"   Total Return: {total_return:+.2f}%")
time.sleep(0.1)
print(f"   Maximum Drawdown: {max_dd*100:.2f}%")
time.sleep(0.1)
print(f"   Total Trades: {trades}")
time.sleep(0.1)
print(f"   Market Exposure: {exposure:.1f}%")
time.sleep(0.1)

print(f"\n📈 PERFORMANCE ANALYSIS:")
time.sleep(0.1)
if total_return > 0:
    print(f"   ✅ Strategy was profitable")
    time.sleep(0.1)
else:
    print(f"   ❌ Strategy was not profitable")
    time.sleep(0.1)

print(f"   Trading Frequency: {trades} trades in {len(y_pred_proba)} days")
time.sleep(0.1)
print(f"   Days in Market: {trades} ({exposure:.1f}% of time)")
time.sleep(0.1)
print(f"   Days Out of Market: {len(y_pred_proba) - trades} ({100-exposure:.1f}% of time)")
time.sleep(0.1)

print("="*60)
time.sleep(0.1)
