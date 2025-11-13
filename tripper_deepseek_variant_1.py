import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURABLE HYPERPARAMETERS
# =============================================================================

# Ranges for brute force optimization
LOOKBACK_DAYS_RANGE = range(5, 31)  # 5 to 30 inclusive
TARGET_DAYS_AHEAD_RANGE = range(5, 31)  # 5 to 30 inclusive

# =============================================================================

def fetch_bitcoin_data():
    """Fetch daily Bitcoin price data from Binance starting Jan 1, 2018"""
    client = Client()
    
    # Get daily BTCUSDT data from January 1, 2018
    klines = client.get_historical_klines(
        "BTCUSDT",
        Client.KLINE_INTERVAL_1DAY,
        "1 January, 2018"
    )
    
    # Create DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    
    df = pd.DataFrame(klines, columns=columns)
    
    # Convert to proper data types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    
    df.set_index('timestamp', inplace=True)
    
    return df[['open', 'high', 'low', 'close', 'volume']]

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD line and signal line"""
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    return macd, signal_line

def calculate_stochastic_rsi(df, rsi_period=14, stoch_period=14, k=3, d=3):
    """Calculate Stochastic RSI"""
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic RSI
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
    
    return stoch_rsi

def create_features_and_target(df, lookback_days, target_days_ahead):
    """Create features and target variable"""
    # Calculate price change percentages
    df['price_pct_change'] = df['close'].pct_change()
    
    # Calculate volume change percentages
    df['volume_pct_change'] = df['volume'].pct_change()
    
    # Calculate MACD and signal line difference
    macd, signal_line = calculate_macd(df)
    df['macd_signal_diff'] = macd - signal_line
    
    # Calculate Stochastic RSI
    df['stoch_rsi'] = calculate_stochastic_rsi(df)
    
    # Create target: price change direction after target_days_ahead days
    df['future_price'] = df['close'].shift(-target_days_ahead)
    df['price_change_direction'] = (df['future_price'] > df['close']).astype(int)
    
    # Create feature columns for last lookback_days of each indicator
    feature_columns = []
    
    # lookback_days of price change percentages
    for i in range(1, lookback_days + 1):
        df[f'price_pct_change_lag_{i}'] = df['price_pct_change'].shift(i)
        feature_columns.append(f'price_pct_change_lag_{i}')
    
    # lookback_days of volume change percentages
    for i in range(1, lookback_days + 1):
        df[f'volume_pct_change_lag_{i}'] = df['volume_pct_change'].shift(i)
        feature_columns.append(f'volume_pct_change_lag_{i}')
    
    # lookback_days of MACD - Signal differences
    for i in range(1, lookback_days + 1):
        df[f'macd_signal_diff_lag_{i}'] = df['macd_signal_diff'].shift(i)
        feature_columns.append(f'macd_signal_diff_lag_{i}')
    
    # lookback_days of Stochastic RSI values
    for i in range(1, lookback_days + 1):
        df[f'stoch_rsi_lag_{i}'] = df['stoch_rsi'].shift(i)
        feature_columns.append(f'stoch_rsi_lag_{i}')
    
    # Drop rows with NaN values (from lag features and indicator calculations)
    df_clean = df.dropna(subset=feature_columns + ['price_change_direction'])
    
    return df_clean, feature_columns

def evaluate_hyperparameters(lookback_days, target_days_ahead, df):
    """Evaluate model performance for given hyperparameters"""
    try:
        # Create features and target
        df_clean, feature_columns = create_features_and_target(df.copy(), lookback_days, target_days_ahead)
        
        # Check if we have enough data
        if len(df_clean) < 100:
            return None, None, None
        
        # Prepare data for training
        X = df_clean[feature_columns]
        y = df_clean['price_change_direction']
        
        # Split data (chronological split)
        split_index = int(len(X) * 0.8)
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        # Check if we have enough test samples
        if len(X_test) < 50:
            return None, None, None
        
        # Train logistic regression model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Make predictions and calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate baseline accuracy
        baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
        improvement = accuracy - baseline_accuracy
        
        return accuracy, baseline_accuracy, improvement
        
    except Exception as e:
        print(f"Error with lookback={lookback_days}, target={target_days_ahead}: {e}")
        return None, None, None

def main():
    print("Fetching Bitcoin data from Binance...")
    df = fetch_bitcoin_data()
    print(f"Fetched {len(df)} days of data")
    
    print("\nStarting brute force hyperparameter optimization...")
    print(f"Lookback days range: {LOOKBACK_DAYS_RANGE}")
    print(f"Target days ahead range: {TARGET_DAYS_AHEAD_RANGE}")
    print(f"Total combinations to test: {len(LOOKBACK_DAYS_RANGE) * len(TARGET_DAYS_AHEAD_RANGE)}")
    
    results = []
    best_accuracy = 0
    best_params = None
    
    # Test all combinations
    for lookback_days in LOOKBACK_DAYS_RANGE:
        for target_days_ahead in TARGET_DAYS_AHEAD_RANGE:
            print(f"Testing: lookback={lookback_days}, target={target_days_ahead}")
            
            accuracy, baseline, improvement = evaluate_hyperparameters(lookback_days, target_days_ahead, df)
            
            if accuracy is not None:
                results.append({
                    'lookback_days': lookback_days,
                    'target_days_ahead': target_days_ahead,
                    'accuracy': accuracy,
                    'baseline_accuracy': baseline,
                    'improvement': improvement,
                    'total_features': lookback_days * 4  # 4 feature types
                })
                
                # Update best parameters
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = (lookback_days, target_days_ahead)
                
                print(f"  Accuracy: {accuracy:.4f}, Improvement: {improvement:.4f}")
            else:
                print(f"  Skipped - insufficient data")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No valid results found. Please check the data.")
        return
    
    # Sort by accuracy (descending)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("BRUTE FORCE OPTIMIZATION RESULTS")
    print("="*80)
    
    # Display top 10 results
    print("\nTop 10 Parameter Combinations:")
    print(results_df.head(10).round(4).to_string(index=False))
    
    # Best result
    best_result = results_df.iloc[0]
    print(f"\nBEST COMBINATION:")
    print(f"Lookback Days: {best_result['lookback_days']}")
    print(f"Target Days Ahead: {best_result['target_days_ahead']}")
    print(f"Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"Baseline Accuracy: {best_result['baseline_accuracy']:.4f} ({best_result['baseline_accuracy']*100:.2f}%)")
    print(f"Improvement: {best_result['improvement']:.4f} ({best_result['improvement']*100:.2f}%)")
    print(f"Total Features: {best_result['total_features']}")
    
    # Summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"Total combinations tested: {len(LOOKBACK_DAYS_RANGE) * len(TARGET_DAYS_AHEAD_RANGE)}")
    print(f"Valid results: {len(results_df)}")
    print(f"Average accuracy: {results_df['accuracy'].mean():.4f}")
    print(f"Maximum accuracy: {results_df['accuracy'].max():.4f}")
    print(f"Minimum accuracy: {results_df['accuracy'].min():.4f}")
    
    return results_df

if __name__ == "__main__":
    results_df = main()
