import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Step 1: Fetch Bitcoin OHLCV data from Binance
print("Fetching Bitcoin data from Binance...")
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

# Step 2: Calculate technical indicators
print("Calculating technical indicators...")

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

# Step 5: Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Step 7: Simulate trading strategies with confidence threshold
print("\nSimulating trading strategies...")

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

# Basic Strategy (100% long/short binary - no threshold)
capital_basic = 10000
capital_history_basic = [capital_basic]
trades_basic = 0

# Confidence Threshold Strategy (10% threshold)
capital_threshold = 10000
capital_history_threshold = [capital_threshold]
trades_threshold = 0

test_dates = X_test.index
test_prices = btc_data.loc[test_dates, 'close']

for i in range(len(X_test)):
    current_price = test_prices.iloc[i]
    prediction_prob = y_pred_proba[i]
    
    # BASIC STRATEGY: Binary 100% long/short (no threshold)
    if prediction_prob > 0.5:
        position_basic = 1  # 100% long
        trades_basic += 1
    else:
        position_basic = -1  # 100% short
        trades_basic += 1
    
    # CONFIDENCE THRESHOLD STRATEGY: Only trade with 10% confidence
    if prediction_prob >= 0.60:
        position_threshold = 1  # 100% long
        trades_threshold += 1
    elif prediction_prob <= 0.40:
        position_threshold = -1  # 100% short
        trades_threshold += 1
    else:
        position_threshold = 0  # Flat position (no trade)
    
    # Calculate daily return based on positions
    if i < len(X_test) - 1:
        next_price = test_prices.iloc[i + 1]
        daily_return = (next_price - current_price) / current_price
        
        # Update basic strategy capital
        if position_basic == 1:  # Long
            capital_basic *= (1 + daily_return)
        elif position_basic == -1:  # Short
            capital_basic *= (1 - daily_return)
        
        # Update threshold strategy capital
        if position_threshold == 1:  # Long
            capital_threshold *= (1 + daily_return)
        elif position_threshold == -1:  # Short
            capital_threshold *= (1 - daily_return)
        # If position_threshold == 0, capital remains unchanged (flat)
    
    # Update capital histories
    capital_history_basic.append(capital_basic)
    capital_history_threshold.append(capital_threshold)

# Calculate drawdowns and returns
max_drawdown_basic = calculate_drawdown(capital_history_basic)
max_drawdown_threshold = calculate_drawdown(capital_history_threshold)

total_return_basic = (capital_basic - 10000) / 10000 * 100
total_return_threshold = (capital_threshold - 10000) / 10000 * 100

# Calculate trade statistics
total_days = len(X_test)
days_in_market_threshold = trades_threshold
days_out_of_market = total_days - trades_threshold

print("\n" + "="*60)
print("TRADING STRATEGY COMPARISON: CONFIDENCE THRESHOLD")
print("="*60)

print(f"\n📊 MODEL PERFORMANCE:")
print(f"   Accuracy: {accuracy*100:.2f}%")

print(f"\n⚡ BASIC STRATEGY (Always Invested):")
print(f"   Final Capital: ${capital_basic:,.2f}")
print(f"   Total Return: {total_return_basic:+.2f}%")
print(f"   Maximum Drawdown: {max_drawdown_basic*100:.2f}%")
print(f"   Total Trades: {trades_basic}")
print(f"   Days in Market: {total_days} (100%)")

print(f"\n🎯 CONFIDENCE THRESHOLD STRATEGY (10% Threshold):")
print(f"   Final Capital: ${capital_threshold:,.2f}")
print(f"   Total Return: {total_return_threshold:+.2f}%")
print(f"   Maximum Drawdown: {max_drawdown_threshold*100:.2f}%")
print(f"   Total Trades: {trades_threshold}")
print(f"   Days in Market: {days_in_market_threshold} ({days_in_market_threshold/total_days*100:.1f}%)")
print(f"   Days Out of Market: {days_out_of_market} ({days_out_of_market/total_days*100:.1f}%)")

print(f"\n📈 COMPARISON:")
if capital_threshold > capital_basic:
    improvement = ((capital_threshold - capital_basic) / capital_basic) * 100
    print(f"   ✅ Threshold strategy outperformed by: +{improvement:.2f}%")
else:
    improvement = ((capital_basic - capital_threshold) / capital_threshold) * 100
    print(f"   ❌ Basic strategy outperformed by: +{improvement:.2f}%")

print(f"\n💡 STRATEGY INSIGHTS:")
print(f"   Basic: Always 100% invested, higher trading frequency")
print(f"   Threshold: Selective trading, potentially better risk-adjusted returns")
print(f"   Threshold Rules: Long ≥0.60, Short ≤0.40, Flat between 0.40-0.60")
print("="*60)

# Additional analysis: Show prediction distribution
high_conviction_long = len([p for p in y_pred_proba if p >= 0.60])
high_conviction_short = len([p for p in y_pred_proba if p <= 0.40])
neutral_zone = len([p for p in y_pred_proba if 0.40 < p < 0.60])

print(f"\n🔍 PREDICTION CONFIDENCE DISTRIBUTION:")
print(f"   High Conviction Long (≥0.60): {high_conviction_long} trades ({high_conviction_long/len(y_pred_proba)*100:.1f}%)")
print(f"   High Conviction Short (≤0.40): {high_conviction_short} trades ({high_conviction_short/len(y_pred_proba)*100:.1f}%)")
print(f"   No Trade Zone (0.40-0.60): {neutral_zone} trades ({neutral_zone/len(y_pred_proba)*100:.1f}%)")

# Calculate win rate for threshold strategy
if trades_threshold > 0:
    print(f"\n📊 THRESHOLD STRATEGY EFFICIENCY:")
    print(f"   Trade Frequency: {trades_threshold}/{total_days} days ({trades_threshold/total_days*100:.1f}%)")
    market_exposure = days_in_market_threshold / total_days * 100
    print(f"   Market Exposure: {market_exposure:.1f}%")
