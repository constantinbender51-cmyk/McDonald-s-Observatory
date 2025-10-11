import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def fetch_binance_data(symbol='BTCUSDT', start_date='2017-01-01'):
    """Fetch daily OHLCV data from Binance"""
    url = 'https://api.binance.com/api/v3/klines'
    
    start_ms = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ms = int(datetime.now().timestamp() * 1000)
    
    all_data = []
    current_start = start_ms
    
    print("Fetching BTC data from Binance...")
    while current_start < end_ms:
        params = {
            'symbol': symbol,
            'interval': '1d',
            'startTime': current_start,
            'limit': 1000
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        current_start = data[-1][0] + 1
        
        if len(data) < 1000:
            break
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                          'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                                          'taker_buy_quote', 'ignore'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    print(f"Fetched {len(df)} days of data from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def calculate_stoch_rsi(prices, period=14, smooth_k=3, smooth_d=3):
    """Calculate Stochastic RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    stoch_rsi = (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min())
    stoch_rsi = stoch_rsi.rolling(window=smooth_k).mean()
    
    return stoch_rsi

def create_features(df, lookback=10):
    """Create features for the model"""
    df = df.copy()
    
    df['macd_diff'] = calculate_macd(df['close'])
    df['stoch_rsi'] = calculate_stoch_rsi(df['close'])
    df['price_change_pct'] = df['close'].pct_change() * 100
    
    # Calculate historical volatility for risk management
    df['volatility_20d'] = df['close'].pct_change().rolling(window=20).std() * 100
    
    feature_cols = []
    for i in range(1, lookback + 1):
        df[f'macd_diff_lag_{i}'] = df['macd_diff'].shift(i)
        df[f'stoch_rsi_lag_{i}'] = df['stoch_rsi'].shift(i)
        df[f'volume_lag_{i}'] = df['volume'].shift(i)
        df[f'price_change_pct_lag_{i}'] = df['price_change_pct'].shift(i)
        
        feature_cols.extend([f'macd_diff_lag_{i}', f'stoch_rsi_lag_{i}', 
                            f'volume_lag_{i}', f'price_change_pct_lag_{i}'])
    
    df['target_6d_pct'] = ((df['close'].shift(-6) - df['close']) / df['close'] * 100)
    df['target_10d_pct'] = ((df['close'].shift(-10) - df['close']) / df['close'] * 100)
    
    df['target_6d'] = (df['target_6d_pct'] > 0).astype(int)
    df['target_10d'] = (df['target_10d_pct'] > 0).astype(int)
    
    df = df.dropna()
    
    return df, feature_cols

def train_models(df, feature_cols):
    """Train two logistic regression models"""
    X = df[feature_cols]
    y_6d = df['target_6d']
    y_10d = df['target_10d']
    
    X_train, X_test, y_6d_train, y_6d_test, y_10d_train, y_10d_test = train_test_split(
        X, y_6d, y_10d, test_size=0.2, shuffle=False
    )
    
    train_idx = X_train.index
    test_idx = X_test.index
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining models...")
    model_6d = LogisticRegression(max_iter=1000, random_state=42)
    model_10d = LogisticRegression(max_iter=1000, random_state=42)
    
    model_6d.fit(X_train_scaled, y_6d_train)
    model_10d.fit(X_train_scaled, y_10d_train)
    
    print(f"Model 6d accuracy: {model_6d.score(X_test_scaled, y_6d_test):.4f}")
    print(f"Model 10d accuracy: {model_10d.score(X_test_scaled, y_10d_test):.4f}")
    
    return model_6d, model_10d, scaler, train_idx, test_idx

def simulate_trading(df, model_6d, model_10d, scaler, feature_cols, test_idx, leverage=3, stop_loss_pct=2.0, position_size_pct=0.95):
    """Simulate trading strategy on test set with corrected position sizing and risk management"""
    test_df = df.loc[test_idx].copy()
    
    X_test = test_df[feature_cols]
    X_test_scaled = scaler.transform(X_test)
    
    pred_6d = model_6d.predict(X_test_scaled)
    pred_10d = model_10d.predict(X_test_scaled)
    
    test_df['pred_6d'] = pred_6d
    test_df['pred_10d'] = pred_10d
    
    initial_capital = 10000
    cash = initial_capital
    position_size = 0  # Number of BTC held
    position_type = None
    entry_price = 0
    stop_loss_price = 0
    
    equity_curve = []
    trades = []
    
    print("\nSimulating trading strategy...")
    print(f"Leverage: {leverage}x")
    print(f"Stop Loss: {stop_loss_pct}%")
    print(f"Position Size: {position_size_pct * 100}% of available capital")
    
    for idx, row in test_df.iterrows():
        current_price = row['close']
        pred_6 = row['pred_6d']
        pred_10 = row['pred_10d']
        
        # Calculate current equity value
        if position_size == 0:
            current_equity = cash
        else:
            if position_type == 'long':
                # For long: profit = position_size * (current_price - entry_price)
                unrealized_pnl = position_size * (current_price - entry_price)
                current_equity = cash + unrealized_pnl
            else:  # short
                # For short: profit = position_size * (entry_price - current_price)
                unrealized_pnl = position_size * (entry_price - current_price)
                current_equity = cash + unrealized_pnl
        
        equity_curve.append({
            'date': row['timestamp'], 
            'equity': current_equity, 
            'price': current_price, 
            'position': position_type if position_type else 'cash'
        })
        
        # Check stop loss first if we have a position
        if position_size > 0:
            stop_loss_triggered = False
            
            if position_type == 'long' and current_price <= stop_loss_price:
                stop_loss_triggered = True
            elif position_type == 'short' and current_price >= stop_loss_price:
                stop_loss_triggered = True
            
            if stop_loss_triggered:
                if position_type == 'long':
                    pnl = position_size * (current_price - entry_price)
                else:
                    pnl = position_size * (entry_price - current_price)
                
                cash += pnl
                
                trades.append({
                    'entry_date': row['timestamp'],
                    'entry_price': entry_price, 
                    'exit_price': current_price, 
                    'type': position_type, 
                    'pnl': pnl,
                    'exit_reason': 'stop_loss'
                })
                
                position_size = 0
                position_type = None
                continue
        
        # Determine signal alignment
        both_positive = (pred_6 == 1) and (pred_10 == 1)
        both_negative = (pred_6 == 0) and (pred_10 == 0)
        misaligned = (pred_6 != pred_10)
        
        # Trading logic
        if position_size == 0:
            # No position - look to enter
            if both_positive:
                # Enter long position
                capital_to_use = cash * position_size_pct
                position_size = (capital_to_use * leverage) / current_price
                position_type = 'long'
                entry_price = current_price
                stop_loss_price = current_price * (1 - stop_loss_pct / 100)
                cash = cash - capital_to_use
                
            elif both_negative:
                # Enter short position
                capital_to_use = cash * position_size_pct
                position_size = (capital_to_use * leverage) / current_price
                position_type = 'short'
                entry_price = current_price
                stop_loss_price = current_price * (1 + stop_loss_pct / 100)
                cash = cash - capital_to_use
        
        else:
            # We have a position - check if we should exit
            should_exit = False
            exit_reason = ''
            
            if misaligned:
                # Models disagree - exit position
                should_exit = True
                exit_reason = 'signal_misalignment'
                
            elif (both_positive and position_type == 'short'):
                # Signal reversed from bearish to bullish while short
                should_exit = True
                exit_reason = 'signal_reversal'
                
            elif (both_negative and position_type == 'long'):
                # Signal reversed from bullish to bearish while long
                should_exit = True
                exit_reason = 'signal_reversal'
            
            if should_exit:
                # Close position
                if position_type == 'long':
                    pnl = position_size * (current_price - entry_price)
                else:
                    pnl = position_size * (entry_price - current_price)
                
                cash += pnl
                
                trades.append({
                    'entry_date': row['timestamp'],
                    'entry_price': entry_price, 
                    'exit_price': current_price, 
                    'type': position_type, 
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })
                
                position_size = 0
                position_type = None
                
                # Check if we should immediately enter opposite position after reversal
                if exit_reason == 'signal_reversal':
                    if both_positive:
                        capital_to_use = cash * position_size_pct
                        position_size = (capital_to_use * leverage) / current_price
                        position_type = 'long'
                        entry_price = current_price
                        stop_loss_price = current_price * (1 - stop_loss_pct / 100)
                        cash = cash - capital_to_use
                        
                    elif both_negative:
                        capital_to_use = cash * position_size_pct
                        position_size = (capital_to_use * leverage) / current_price
                        position_type = 'short'
                        entry_price = current_price
                        stop_loss_price = current_price * (1 + stop_loss_pct / 100)
                        cash = cash - capital_to_use
    
    # Close any remaining position at end of test period
    if position_size > 0:
        final_price = test_df.iloc[-1]['close']
        if position_type == 'long':
            pnl = position_size * (final_price - entry_price)
        else:
            pnl = position_size * (entry_price - final_price)
        
        cash += pnl
        
        trades.append({
            'entry_date': test_df.iloc[-1]['timestamp'],
            'entry_price': entry_price, 
            'exit_price': final_price, 
            'type': position_type, 
            'pnl': pnl,
            'exit_reason': 'end_of_period'
        })
    
    final_equity = cash
    
    # Calculate performance metrics
    start_price = test_df.iloc[0]['close']
    end_price = test_df.iloc[-1]['close']
    buy_hold_return = ((end_price - start_price) / start_price) * 100
    strategy_return = ((final_equity - initial_capital) / initial_capital) * 100
    
    # Calculate additional metrics
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    
    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
    
    print(f"\nResults:")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Equity: ${final_equity:,.2f}")
    print(f"Strategy Return: {strategy_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Number of Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    
    return equity_curve, trades, final_equity, strategy_return, buy_hold_return

def main():
    df = fetch_binance_data()
    
    df, feature_cols = create_features(df, lookback=10)
    print(f"\nCreated features with {len(feature_cols)} columns")
    
    model_6d, model_10d, scaler, train_idx, test_idx = train_models(df, feature_cols)
    
    # ⬇️ Add this block here (right after training)
    test_df = df.loc[test_idx].copy()
    X_test = test_df[feature_cols]
    X_test_scaled = scaler.transform(X_test)
    test_df['pred_6d'] = model_6d.predict(X_test_scaled)
    test_df['pred_10d'] = model_10d.predict(X_test_scaled)
    
    # Shift predictions forward by one day to avoid lookahead bias
    test_df[['pred_6d', 'pred_10d']] = test_df[['pred_6d', 'pred_10d']].shift(1)
    test_df = test_df.dropna()

    # Then run simulation using the shifted signals
    equity_curve, trades, final_equity, strategy_return, buy_hold_return = simulate_trading(
        df, model_6d, model_10d, scaler, feature_cols, test_idx, 
        leverage=3, stop_loss_pct=2.0, position_size_pct=0.95
    )    
    results_df = pd.DataFrame({
        'Metric': ['Initial Capital', 'Final Equity', 'Strategy Return (%)', 
                   'Buy & Hold Return (%)', 'Number of Trades', 'Leverage', 'Stop Loss (%)'],
        'Value': [10000, final_equity, strategy_return, buy_hold_return, len(trades), 3, 2.0]
    })
    
    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_filename = f'btc_trading_results_{timestamp}.csv'
    equity_filename = f'btc_equity_curve_{timestamp}.csv'
    trades_filename = f'btc_trades_{timestamp}.csv'
    
    results_df.to_csv(results_filename, index=False)
    equity_df.to_csv(equity_filename, index=False)
    if len(trades) > 0:
        trades_df.to_csv(trades_filename, index=False)
    
    print(f"\nResults saved to:")
    print(f"  - {results_filename}")
    print(f"  - {equity_filename}")
    print(f"  - {trades_filename}")
    
    return results_df, equity_df, trades_df

if __name__ == "__main__":
    main()
