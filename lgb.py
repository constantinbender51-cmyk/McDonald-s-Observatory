import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
from datetime import datetime
import warnings
import gdown
import os
import time
warnings.filterwarnings('ignore')

# Custom print function with delay
def delayed_print(message):
    print(message, flush=True)
    time.sleep(0.1)

# Download data from Google Drive
delayed_print("Downloading data from Google Drive...")
file_id = "1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o"
url = f"https://drive.google.com/uc?id={file_id}"
output = "1m.csv"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)
    delayed_print("Download complete!")
else:
    delayed_print("File already exists, skipping download.")

delayed_print("\nLoading data...")
df = pd.read_csv('1m.csv')
delayed_print(f"Original 1-minute data shape: {df.shape}")
delayed_print(f"Columns: {df.columns.tolist()}")

# Ensure we have timestamp
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
elif 'date' in df.columns:
    df['timestamp'] = pd.to_datetime(df['date'])
else:
    df['timestamp'] = pd.to_datetime(df.iloc[:, 0])

# Standardize column names
df.columns = df.columns.str.lower()

# Convert 1-minute data to 1-hour data
delayed_print("\nConverting 1-minute candles to 1-hour candles...")
delayed_print("This will dramatically reduce data size and training time...")

# Set timestamp as index for resampling
df.set_index('timestamp', inplace=True)

# Resample to hourly candles (OHLCV aggregation)
df_hourly = df.resample('1H').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

df_hourly.reset_index(inplace=True)
df = df_hourly

delayed_print(f"Hourly data shape: {df.shape}")
delayed_print(f"Data now spans from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
delayed_print(f"Total hours of data: {len(df)}")
delayed_print(df.head().to_string())

delayed_print("\n" + "="*80)
delayed_print("FEATURE ENGINEERING")
delayed_print("="*80)

# 1. Log Returns
delayed_print("\n1. Calculating log returns...")
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# 2. Moving Averages
delayed_print("2. Calculating moving averages...")
ma_periods = [7, 14, 21, 50, 200]
for period in ma_periods:
    df[f'ma_{period}'] = df['close'].rolling(window=period).mean()

# 3. MA Ratios
delayed_print("3. Calculating MA ratios...")
df['ma_ratio_20_50'] = df['close'].rolling(20).mean() / df['close'].rolling(50).mean()
df['ma_ratio_50_200'] = df['close'].rolling(50).mean() / df['close'].rolling(200).mean()

# 4. MACD
delayed_print("4. Calculating MACD...")
exp1 = df['close'].ewm(span=12, adjust=False).mean()
exp2 = df['close'].ewm(span=26, adjust=False).mean()
macd = exp1 - exp2
signal = macd.ewm(span=9, adjust=False).mean()
df['macd_minus_signal'] = macd - signal

# 5. RSI
delayed_print("5. Calculating RSI...")
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi'] = calculate_rsi(df['close'])

# 6. Stochastic RSI
delayed_print("6. Calculating Stochastic RSI...")
rsi = df['rsi']
stoch_rsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
df['stoch_rsi'] = stoch_rsi

# 7. ATR
delayed_print("7. Calculating ATR...")
high_low = df['high'] - df['low']
high_close = np.abs(df['high'] - df['close'].shift())
low_close = np.abs(df['low'] - df['close'].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr'] = true_range.rolling(14).mean()

# 8. OBV
delayed_print("8. Calculating OBV...")
obv = [0]
for i in range(1, len(df)):
    if df['close'].iloc[i] > df['close'].iloc[i-1]:
        obv.append(obv[-1] + df['volume'].iloc[i])
    elif df['close'].iloc[i] < df['close'].iloc[i-1]:
        obv.append(obv[-1] - df['volume'].iloc[i])
    else:
        obv.append(obv[-1])
df['obv'] = obv

# 9. VWAP
delayed_print("9. Calculating VWAP...")
df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

# 10. Time-based features
delayed_print("10. Adding time-based features...")
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# List of indicators to create lags for
indicator_cols = ['log_return', 'ma_7', 'ma_14', 'ma_21', 'ma_50', 'ma_200',
                  'ma_ratio_20_50', 'ma_ratio_50_200', 'macd_minus_signal',
                  'rsi', 'stoch_rsi', 'atr', 'obv', 'vwap']

# 11. Create lagged features with structured time intervals (OPTIMIZED for memory)
delayed_print("\n11. Creating lagged features with optimized time intervals...")
delayed_print("Using minimal lag set to manage memory on Railway...")

# OPTIMIZED lag positions - only 11 lags total:
# Recent: 5, 15, 30 minutes (3 lags)
# Hourly: 60, 240, 720 minutes = 1h, 4h, 12h (3 lags)
# Daily: 1440, 4320, 7200 minutes = 1d, 3d, 5d (3 lags)
# Weekly: 10080, 20160 minutes = 1w, 2w (2 lags)

lag_positions = [5, 15, 30, 60, 240, 720, 1440, 4320, 7200, 10080, 20160]

delayed_print(f"Number of lag positions: {len(lag_positions)}")
delayed_print(f"Lag structure:")
delayed_print(f"  Recent (minutes): 5, 15, 30")
delayed_print(f"  Hourly: 60 (1h), 240 (4h), 720 (12h)")
delayed_print(f"  Daily: 1440 (1d), 4320 (3d), 7200 (5d)")
delayed_print(f"  Weekly: 10080 (1w), 20160 (2w)")

# Create lagged features with progress tracking
total_lags = len(indicator_cols) * len(lag_positions)
lag_counter = 0

delayed_print(f"\nCreating {total_lags} lagged features...")
for col in indicator_cols:
    if col in df.columns:
        for lag in lag_positions:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            lag_counter += 1
            if lag_counter % 20 == 0:  # Progress update every 20 lags
                delayed_print(f"  Progress: {lag_counter}/{total_lags} lags created...")

delayed_print(f"  Completed: {lag_counter}/{total_lags} lags created!")

delayed_print(f"\nTotal features after lagging: {df.shape[1]}")

# 12. Create target variable (24 hours ahead)
delayed_print("\n12. Creating target variable (predicting 24 hours ahead)...")
df['future_return'] = (df['close'].shift(-24) - df['close']) / df['close'] * 100

# Find optimal threshold for balanced classes
delayed_print("\n13. Finding optimal threshold for balanced class distribution...")
thresholds_to_test = np.arange(0.3, 2.0, 0.1)
best_threshold = None
best_balance = float('inf')

for threshold in thresholds_to_test:
    temp_target = pd.cut(df['future_return'], 
                         bins=[-np.inf, -threshold, threshold, np.inf],
                         labels=[0, 1, 2])  # 0=Down, 1=Sideways, 2=Up
    value_counts = temp_target.value_counts(normalize=True)
    if len(value_counts) == 3:
        balance = value_counts.std()
        if balance < best_balance:
            best_balance = balance
            best_threshold = threshold

delayed_print(f"Optimal threshold: {best_threshold:.2f}%")

# Create target with optimal threshold
df['target'] = pd.cut(df['future_return'], 
                      bins=[-np.inf, -best_threshold, best_threshold, np.inf],
                      labels=[0, 1, 2])  # 0=Down, 1=Sideways, 2=Up

delayed_print("\nClass distribution:")
delayed_print(df['target'].value_counts(normalize=True).to_string())
delayed_print("\nClass counts:")
delayed_print(df['target'].value_counts().to_string())

# Clean data by removing rows with NaNs (avoiding dropna() memory spike)
delayed_print("\n14. Cleaning data...")
delayed_print(f"Rows before cleaning: {len(df)}")

# Drop first 24 rows (longest lag = 24 hours) and last 24 rows (target forward look = 24 hours)
# This removes all rows that would have NaN values without using dropna()
rows_to_drop_start = 24
rows_to_drop_end = 24

delayed_print(f"Removing first {rows_to_drop_start} rows (24-hour lag period)...")
delayed_print(f"Removing last {rows_to_drop_end} rows (24-hour target forward look)...")

df = df.iloc[rows_to_drop_start:-rows_to_drop_end].reset_index(drop=True)

delayed_print(f"Rows after cleaning: {len(df)}")
delayed_print("Memory-efficient cleaning complete!")

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 
                                                           'close', 'volume', 'future_return', 'target']]
X = df[feature_cols]
y = df['target'].astype(int)

delayed_print(f"\nFeature matrix shape: {X.shape}")
delayed_print(f"Target shape: {y.shape}")

# Split data chronologically (80% train, 20% test)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

delayed_print(f"\nTrain set: {X_train.shape}")
delayed_print(f"Test set: {X_test.shape}")

delayed_print("\n" + "="*80)
delayed_print("TRAINING LIGHTGBM MODEL")
delayed_print("="*80)

# Train LightGBM
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

print("\nTraining model...")
model = lgb.train(params,
                  train_data,
                  num_boost_round=200,
                  valid_sets=[train_data, test_data],
                  valid_names=['train', 'valid'])

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# Predictions
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_class)
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_class, 
                          target_names=['Down', 'Sideways', 'Up']))

# Metrics
f1 = f1_score(y_test, y_pred_class, average='macro')
precision = precision_score(y_test, y_pred_class, average='macro')
recall = recall_score(y_test, y_pred_class, average='macro')

print(f"\nMacro F1 Score: {f1:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall: {recall:.4f}")

# Print confusion matrix in a formatted way
print("\nConfusion Matrix (formatted):")
print("                Predicted")
print("              Down  Sideways  Up")
print(f"Actual Down    {cm[0,0]:6d}  {cm[0,1]:8d}  {cm[0,2]:4d}")
print(f"     Sideways  {cm[1,0]:6d}  {cm[1,1]:8d}  {cm[1,2]:4d}")
print(f"     Up        {cm[2,0]:6d}  {cm[2,1]:8d}  {cm[2,2]:4d}")

# Feature importance
print("\n" + "="*80)
print("TOP 20 FEATURE IMPORTANCES")
print("="*80)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string())

print("\n" + "="*80)
print("BACKTESTING")
print("="*80)

# Backtesting
initial_capital = 10000
capital = initial_capital
position = 0  # 0 = no position, 1 = long, -1 = short
trade_log = []

# Get test set with actual prices
test_df = df.iloc[split_idx:].copy()
test_df['predicted'] = y_pred_class

print(f"\nInitial Capital: ${initial_capital:,.2f}")
print(f"Backtesting period: {test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]}")
print(f"Number of predictions: {len(test_df)}")

for idx in range(len(test_df) - 1440):  # -1440 to ensure we can look ahead
    current_prediction = test_df['predicted'].iloc[idx]
    current_price = test_df['close'].iloc[idx]
    future_price = test_df['close'].iloc[idx + 1440]
    
    # Trading logic
    if current_prediction == 2 and position != 1:  # Predict Up, go long
        if position == -1:  # Close short
            pnl = (current_price - entry_price) / entry_price * capital * -1
            capital += pnl
        position = 1
        entry_price = current_price
        
    elif current_prediction == 0 and position != -1:  # Predict Down, go short
        if position == 1:  # Close long
            pnl = (current_price - entry_price) / entry_price * capital
            capital += pnl
        position = -1
        entry_price = current_price
        
    elif current_prediction == 1 and position != 0:  # Predict Sideways, close position
        if position == 1:
            pnl = (current_price - entry_price) / entry_price * capital
        else:
            pnl = (current_price - entry_price) / entry_price * capital * -1
        capital += pnl
        position = 0

# Close any open position at the end
if position != 0:
    final_price = test_df['close'].iloc[-1]
    if position == 1:
        pnl = (final_price - entry_price) / entry_price * capital
    else:
        pnl = (final_price - entry_price) / entry_price * capital * -1
    capital += pnl

final_capital = capital
total_return = (final_capital - initial_capital) / initial_capital * 100

delayed_print(f"\nFinal Capital: ${final_capital:,.2f}")
delayed_print(f"Total Return: {total_return:.2f}%")
delayed_print(f"Profit/Loss: ${final_capital - initial_capital:,.2f}")

# Buy and hold comparison
buy_hold_return = (test_df['close'].iloc[-1] - test_df['close'].iloc[0]) / test_df['close'].iloc[0] * 100
buy_hold_capital = initial_capital * (1 + buy_hold_return / 100)

delayed_print(f"\nBuy & Hold Return: {buy_hold_return:.2f}%")
delayed_print(f"Buy & Hold Final Capital: ${buy_hold_capital:,.2f}")
delayed_print(f"\nStrategy vs Buy & Hold: {total_return - buy_hold_return:.2f}% {'outperformance' if total_return > buy_hold_return else 'underperformance'}")

delayed_print("\n" + "="*80)
delayed_print("BACKTEST COMPLETE")
delayed_print("="*80)
