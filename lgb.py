import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
from datetime import datetime
import warnings
import gdown
import os
warnings.filterwarnings('ignore')

# Download data from Google Drive
print("Downloading data from Google Drive...")
file_id = "1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o"
url = f"https://drive.google.com/uc?id={file_id}"
output = "1m.csv"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)
    print("Download complete!")
else:
    print("File already exists, skipping download.")

print("\nLoading data...")
df = pd.read_csv('1m.csv')
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())

# Ensure we have the right columns (assuming: timestamp, open, high, low, close, volume)
# Adjust column names if needed
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
elif 'date' in df.columns:
    df['timestamp'] = pd.to_datetime(df['date'])
else:
    # Assume first column is timestamp
    df['timestamp'] = pd.to_datetime(df.iloc[:, 0])

# Standardize column names
df.columns = df.columns.str.lower()
required_cols = ['open', 'high', 'low', 'close', 'volume']
print(f"\nChecking for required columns: {required_cols}")

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

# 1. Log Returns
print("\n1. Calculating log returns...")
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# 2. Moving Averages
print("2. Calculating moving averages...")
ma_periods = [7, 14, 21, 50, 200]
for period in ma_periods:
    df[f'ma_{period}'] = df['close'].rolling(window=period).mean()

# 3. MA Ratios
print("3. Calculating MA ratios...")
df['ma_ratio_20_50'] = df['close'].rolling(20).mean() / df['close'].rolling(50).mean()
df['ma_ratio_50_200'] = df['close'].rolling(50).mean() / df['close'].rolling(200).mean()

# 4. MACD
print("4. Calculating MACD...")
exp1 = df['close'].ewm(span=12, adjust=False).mean()
exp2 = df['close'].ewm(span=26, adjust=False).mean()
macd = exp1 - exp2
signal = macd.ewm(span=9, adjust=False).mean()
df['macd_minus_signal'] = macd - signal

# 5. RSI
print("5. Calculating RSI...")
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi'] = calculate_rsi(df['close'])

# 6. Stochastic RSI
print("6. Calculating Stochastic RSI...")
rsi = df['rsi']
stoch_rsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
df['stoch_rsi'] = stoch_rsi

# 7. ATR
print("7. Calculating ATR...")
high_low = df['high'] - df['low']
high_close = np.abs(df['high'] - df['close'].shift())
low_close = np.abs(df['low'] - df['close'].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr'] = true_range.rolling(14).mean()

# 8. OBV
print("8. Calculating OBV...")
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
print("9. Calculating VWAP...")
df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

# 10. Time-based features
print("10. Adding time-based features...")
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# List of indicators to create lags for
indicator_cols = ['log_return', 'ma_7', 'ma_14', 'ma_21', 'ma_50', 'ma_200',
                  'ma_ratio_20_50', 'ma_ratio_50_200', 'macd_minus_signal',
                  'rsi', 'stoch_rsi', 'atr', 'obv', 'vwap']

# 11. Create lagged features with sqrt spacing
print("\n11. Creating lagged features with sqrt spacing...")
print("This may take a few minutes...")

# Generate lag positions: sqrt(1) to sqrt(100000), rounded
lag_positions = np.unique(np.round(np.sqrt(np.arange(1, 100001))).astype(int))
print(f"Number of unique lag positions: {len(lag_positions)}")
print(f"First 10 lags: {lag_positions[:10]}")
print(f"Last 10 lags: {lag_positions[-10:]}")

# Create lagged features
for col in indicator_cols:
    if col in df.columns:
        for lag in lag_positions[:100]:  # Limiting to first 100 lags to manage memory
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

print(f"\nTotal features after lagging: {df.shape[1]}")

# 12. Create target variable (1440 candles ahead = 1 day)
print("\n12. Creating target variable...")
df['future_return'] = (df['close'].shift(-1440) - df['close']) / df['close'] * 100

# Find optimal threshold for balanced classes
print("\n13. Finding optimal threshold for balanced class distribution...")
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

print(f"Optimal threshold: {best_threshold:.2f}%")

# Create target with optimal threshold
df['target'] = pd.cut(df['future_return'], 
                      bins=[-np.inf, -best_threshold, best_threshold, np.inf],
                      labels=[0, 1, 2])  # 0=Down, 1=Sideways, 2=Up

print("\nClass distribution:")
print(df['target'].value_counts(normalize=True))
print("\nClass counts:")
print(df['target'].value_counts())

# Drop rows with NaN values
print("\n14. Cleaning data...")
print(f"Rows before cleaning: {len(df)}")
df = df.dropna()
print(f"Rows after cleaning: {len(df)}")

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 
                                                           'close', 'volume', 'future_return', 'target']]
X = df[feature_cols]
y = df['target'].astype(int)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data chronologically (80% train, 20% test)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

print("\n" + "="*80)
print("TRAINING LIGHTGBM MODEL")
print("="*80)

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

print(f"\nFinal Capital: ${final_capital:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Profit/Loss: ${final_capital - initial_capital:,.2f}")

# Buy and hold comparison
buy_hold_return = (test_df['close'].iloc[-1] - test_df['close'].iloc[0]) / test_df['close'].iloc[0] * 100
buy_hold_capital = initial_capital * (1 + buy_hold_return / 100)

print(f"\nBuy & Hold Return: {buy_hold_return:.2f}%")
print(f"Buy & Hold Final Capital: ${buy_hold_capital:,.2f}")
print(f"\nStrategy vs Buy & Hold: {total_return - buy_hold_return:.2f}% {'outperformance' if total_return > buy_hold_return else 'underperformance'}")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)
