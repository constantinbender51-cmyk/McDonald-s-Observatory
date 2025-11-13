import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
X_DAYS = 7            # Number of past days to use as features
ZETA_DAYS = 7         # Target horizon: Predict price direction ZETA days from now
START_DATE = "01 Jan, 2018"
SYMBOL = "BTCUSDT"
INITIAL_CAPITAL = 10000
# ==========================================

def get_binance_data(symbol, start_date):
    """
    Fetches daily klines from Binance starting from start_date until now.
    No API key required for public market data.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Convert start_date to milliseconds timestamp
    dt_obj = datetime.strptime(start_date, "%d %b, %Y")
    start_ts = int(dt_obj.timestamp() * 1000)
    
    klines = []
    print(f"Fetching data for {symbol} from {start_date}...")
    
    while True:
        params = {
            'symbol': symbol,
            'interval': '1d',
            'startTime': start_ts,
            'limit': 1000
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if not data or len(data) == 0:
                break
                
            klines.extend(data)
            
            # Update start_ts to the timestamp of the last kline + 1 day (86400000 ms)
            last_kline_time = data[-1][0]
            start_ts = last_kline_time + 86400000
            
            # Respect API rate limits
            time.sleep(0.1)
            
            # Break if we reached current time
            if last_kline_time >= int(time.time() * 1000):
                break
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    print(f"Total days fetched: {len(klines)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # Type conversion
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('date', inplace=True)
    
    return df[['open', 'high', 'low', 'close', 'volume']]

def calculate_indicators(df):
    """
    Calculates MACD Diff, Stoch RSI, Price Change, Volume Change
    """
    df = df.copy()
    
    # 1. MACD (Standard 12, 26, 9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_diff'] = macd_line - signal_line
    
    # 2. Stochastic RSI (14)
    # First calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Then Stoch RSI
    min_rsi = rsi.rolling(window=14).min()
    max_rsi = rsi.rolling(window=14).max()
    df['stoch_rsi'] = (rsi - min_rsi) / (max_rsi - min_rsi)
    
    # 3. Price Change
    df['price_change'] = df['close'].pct_change()
    
    # 4. Volume Change
    df['volume_change'] = df['volume'].pct_change()
    
    # Fill NaNs created by rolling windows
    df.dropna(inplace=True)
    return df

def prepare_features(df, x_days, zeta_days):
    """
    Creates feature columns for X days of history and the Target.
    Target = 1 if Price(t+zeta) > Price(t), else 0
    """
    data = df.copy()
    feature_cols = []
    
    # Create lag features for X days
    # For t=0 (today), we want values from t-0, t-1, ..., t-(x-1)
    for i in range(x_days):
        for metric in ['macd_diff', 'stoch_rsi', 'price_change', 'volume_change']:
            col_name = f'{metric}_lag_{i}'
            data[col_name] = data[metric].shift(i)
            feature_cols.append(col_name)
            
    # Create Target: Direction ZETA_DAYS ahead
    # Shift(-zeta) gets the future price. 
    # Target is 1 (Long) if Future > Current, else 0 (Short)
    data['future_close'] = data['close'].shift(-zeta_days)
    data['target'] = (data['future_close'] > data['close']).astype(int)
    
    # Drop NaNs created by lagging features and shifting target
    data.dropna(inplace=True)
    
    return data, feature_cols

def calculate_position_size(probability):
    """
    Calculates position size based on conviction.
    Prob > 0.5 => Long, Prob < 0.5 => Short.
    Size is proportional to distance from 0.5.
    Example:
      Prob 0.9 -> (0.9 - 0.5) * 2 = 0.8  (80% Long)
      Prob 0.6 -> (0.6 - 0.5) * 2 = 0.2  (20% Long)
      Prob 0.2 -> (0.2 - 0.5) * 2 = -0.6 (60% Short)
    """
    # Normalize 0.0-1.0 to -1.0 to 1.0 range
    # 0.5 becomes 0.0 (Neutral)
    conviction = (probability - 0.5) * 2
    return conviction

def main():
    # 1. Fetch Data
    df = get_binance_data(SYMBOL, START_DATE)
    
    # 2. Calculate Base Indicators
    df = calculate_indicators(df)
    
    # 3. Prepare Features (X Days) and Target (Zeta Days)
    data, features = prepare_features(df, X_DAYS, ZETA_DAYS)
    
    print(f"Dataset size after processing: {len(data)} rows")
    print(f"Number of features: {len(features)}")
    
    # 4. Train / Test Split
    # We use a time-series split (no random shuffle) to respect causality
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    X_train = train_data[features]
    y_train = train_data['target']
    X_test = test_data[features]
    y_test = test_data['target']
    
    # 5. Train Model
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # 6. Prediction
    # predict_proba returns [prob_class_0, prob_class_1]
    # We want prob_class_1 (Probability of Price Going UP)
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    
    # Accuracy
    acc = accuracy_score(y_test, preds)
    print(f"\nModel Accuracy on Test Data: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # 7. Capital Development Backtest
    # We iterate through the test set day by day.
    # Note: Even though the target predicts 7 days out, the logic specified is:
    # "if for each day the prediction direction is followed..."
    # This implies we take a position today based on the 7-day outlook, and hold it for 1 day (rebalancing daily).
    
    capital = INITIAL_CAPITAL
    capital_history = [capital]
    
    test_data = test_data.copy()
    test_data['model_prob'] = probs
    
    # Using close-to-close return for the next day to simulate daily PnL
    # We shift close price back by 1 to get "tomorrow's close" aligned with "today's prediction"
    # Actually, simpler: Iterate row by row.
    # Row i: We have prediction. We take position.
    # Return is (Close[i+1] - Close[i]) / Close[i]
    
    closes = test_data['close'].values
    dates = test_data.index
    model_probs = test_data['model_prob'].values
    
    for i in range(len(test_data) - 1):
        current_price = closes[i]
        next_price = closes[i+1]
        prob = model_probs[i]
        
        # Calculate conviction/position size (-1.0 to 1.0)
        position_size = calculate_position_size(prob)
        
        # Calculate daily market return
        market_return = (next_price - current_price) / current_price
        
        # Strategy return: Position * Market Return
        # If we are Short (negative position) and market drops (negative return), we make profit.
        daily_pnl = capital * position_size * market_return
        
        capital += daily_pnl
        capital_history.append(capital)
        
    # 8. Visualization
    final_capital = capital_history[-1]
    print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Final Capital:   ${final_capital:.2f}")
    print(f"Return:          {((final_capital - INITIAL_CAPITAL)/INITIAL_CAPITAL)*100:.2f}%")
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates[:len(capital_history)], capital_history, label='Strategy Equity')
    
    # Add a Buy & Hold comparison for context
    buy_hold_return = test_data['close'] / test_data['close'].iloc[0] * INITIAL_CAPITAL
    plt.plot(test_data.index, buy_hold_return, label='Buy & Hold BTC', alpha=0.5, linestyle='--')
    
    plt.title(f"Capital Development (Logistic Regression Backtest)\nX={X_DAYS}, Target={ZETA_DAYS} Days Ahead")
    plt.xlabel("Date")
    plt.ylabel("Capital ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
