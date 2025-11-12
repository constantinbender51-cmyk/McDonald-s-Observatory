import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime
import warnings
import json
import os
from flask import Flask, render_template_string
warnings.filterwarnings('ignore')

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
START_DATE = '2018-01-01'  # Data start date
LOOKBACK_DAYS = 10  # Number of days for input features
PREDICTION_HORIZONS = [6, 10]  # Days ahead to predict
STOP_LOSS_MULTIPLIER = 0.8  # Stop loss as fraction of predicted return (80%)
TRAIN_TEST_SPLIT = 0.8  # Train/test split ratio
INITIAL_CAPITAL = 1000  # Starting capital in USD

# MACD parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Stochastic RSI parameters
STOCH_RSI_PERIOD = 14
STOCH_RSI_SMOOTH_K = 3
STOCH_RSI_SMOOTH_D = 3

# Logistic Regression parameters
LR_MAX_ITER = 1000
LR_RANDOM_STATE = 42

# Binance API parameters
BINANCE_SYMBOL = 'BTCUSDT'
BINANCE_INTERVAL = '1d'

# Web server parameters
PORT = int(os.environ.get('PORT', 5000))
# ============================================================================

def fetch_binance_data(symbol=BINANCE_SYMBOL, interval=BINANCE_INTERVAL, start_date=START_DATE):
    """Fetch OHLCV data from Binance"""
    url = 'https://api.binance.com/api/v3/klines'
    
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000)
    
    all_data = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_ts,
            'limit': 1000
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        current_ts = data[-1][0] + 1
        
        if len(data) < 1000:
            break
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                          'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                          'taker_buy_quote', 'ignore'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].set_index('timestamp')

def calculate_macd(prices, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """Calculate MACD signal line"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return signal_line

def calculate_stoch_rsi(prices, period=STOCH_RSI_PERIOD, smooth_k=STOCH_RSI_SMOOTH_K, smooth_d=STOCH_RSI_SMOOTH_D):
    """Calculate Stochastic RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    stoch_rsi = (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min())
    stoch_rsi_k = stoch_rsi.rolling(window=smooth_k).mean()
    
    return stoch_rsi_k

def create_features(df, lookback=LOOKBACK_DAYS):
    """Create features for the model"""
    features = pd.DataFrame(index=df.index)
    
    # Price percentage changes for last 10 days
    for i in range(1, lookback + 1):
        features[f'close_pct_{i}'] = df['close'].pct_change(i)
    
    # Volume percentage changes for last 10 days
    for i in range(1, lookback + 1):
        features[f'volume_pct_{i}'] = df['volume'].pct_change(i)
    
    # MACD signal line
    features['macd_signal'] = calculate_macd(df['close'])
    
    # Stochastic RSI
    features['stoch_rsi'] = calculate_stoch_rsi(df['close'])
    
    return features

def create_targets(df, horizons=PREDICTION_HORIZONS):
    """Create target variables (future returns)"""
    targets = pd.DataFrame(index=df.index)
    
    for h in horizons:
        targets[f'return_{h}d'] = df['close'].pct_change(h).shift(-h)
        targets[f'direction_{h}d'] = (targets[f'return_{h}d'] > 0).astype(int)
    
    return targets

def prepare_data(df):
    """Prepare features and targets"""
    features = create_features(df)
    targets = create_targets(df)
    
    data = pd.concat([features, targets], axis=1).dropna()
    
    return data

def train_models(data):
    """Train logistic regression models"""
    feature_cols = [col for col in data.columns if col.startswith(('close_pct', 'volume_pct', 'macd', 'stoch'))]
    
    X = data[feature_cols]
    y_6d = data[f'direction_{PREDICTION_HORIZONS[0]}d']
    y_10d = data[f'direction_{PREDICTION_HORIZONS[1]}d']
    
    # Split into train and test
    split_idx = int(len(X) * TRAIN_TEST_SPLIT)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_6d_train, y_6d_test = y_6d.iloc[:split_idx], y_6d.iloc[split_idx:]
    y_10d_train, y_10d_test = y_10d.iloc[:split_idx], y_10d.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    model_6d = LogisticRegression(max_iter=LR_MAX_ITER, random_state=LR_RANDOM_STATE)
    model_10d = LogisticRegression(max_iter=LR_MAX_ITER, random_state=LR_RANDOM_STATE)
    
    model_6d.fit(X_train_scaled, y_6d_train)
    model_10d.fit(X_train_scaled, y_10d_train)
    
    return model_6d, model_10d, scaler, X_test, y_6d_test, y_10d_test, data.iloc[split_idx:], split_idx

def backtest_strategy(df, data, model_6d, model_10d, scaler, test_start_idx, initial_capital=INITIAL_CAPITAL):
    """Backtest the trading strategy with detailed monitoring"""
    feature_cols = [col for col in data.columns if col.startswith(('close_pct', 'volume_pct', 'macd', 'stoch'))]
    
    test_data = data.iloc[test_start_idx:].copy()
    test_prices = df.loc[test_data.index, 'close']
    
    X_test = test_data[feature_cols]
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions
    pred_6d = model_6d.predict(X_test_scaled)
    pred_10d = model_10d.predict(X_test_scaled)
    
    # Get predicted returns for stop loss calculation
    returns_6d = test_data[f'return_{PREDICTION_HORIZONS[0]}d'].values
    
    capital = initial_capital
    position = 0  # 0: no position, 1: long, -1: short
    entry_price = 0
    entry_date = None
    stop_loss = 0
    
    trades = []
    capital_history = []
    benchmark_history = []
    
    initial_benchmark_price = test_prices.iloc[0]
    
    for i in range(len(test_data)):
        current_date = test_data.index[i]
        current_price = test_prices.iloc[i]
        
        # Track capital and benchmark at each step
        current_capital = capital
        if position == 1:
            current_capital = capital * (current_price / entry_price)
        elif position == -1:
            current_capital = capital * (entry_price / current_price)
        
        capital_history.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'capital': current_capital,
            'position': position
        })
        
        benchmark_value = initial_capital * (current_price / initial_benchmark_price)
        benchmark_history.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'value': benchmark_value
        })
        
        # Check stop loss if in position
        if position != 0:
            if position == 1:  # Long position
                if current_price <= stop_loss:
                    # Close position
                    pnl_pct = (current_price - entry_price) / entry_price
                    capital = capital * (1 + pnl_pct)
                    trades.append({
                        'type': 'exit',
                        'position_type': 'long',
                        'date': current_date.strftime('%Y-%m-%d'),
                        'price': current_price,
                        'reason': 'stop_loss',
                        'pnl_pct': pnl_pct * 100,
                        'capital': capital,
                        'entry_date': entry_date,
                        'entry_price': entry_price
                    })
                    position = 0
            elif position == -1:  # Short position
                if current_price >= stop_loss:
                    # Close position
                    pnl_pct = (entry_price - current_price) / entry_price
                    capital = capital * (1 + pnl_pct)
                    trades.append({
                        'type': 'exit',
                        'position_type': 'short',
                        'date': current_date.strftime('%Y-%m-%d'),
                        'price': current_price,
                        'reason': 'stop_loss',
                        'pnl_pct': pnl_pct * 100,
                        'capital': capital,
                        'entry_date': entry_date,
                        'entry_price': entry_price
                    })
                    position = 0
        
        # Generate signals
        signal = 0
        if pred_6d[i] == 1 and pred_10d[i] == 1:
            signal = 1  # Both predict up
        elif pred_6d[i] == 0 and pred_10d[i] == 0:
            signal = -1  # Both predict down
        
        # Execute trades based on signals
        if signal != 0 and signal != position:
            # Close existing position if any
            if position != 0:
                if position == 1:
                    pnl_pct = (current_price - entry_price) / entry_price
                elif position == -1:
                    pnl_pct = (entry_price - current_price) / entry_price
                capital = capital * (1 + pnl_pct)
                trades.append({
                    'type': 'exit',
                    'position_type': 'long' if position == 1 else 'short',
                    'date': current_date.strftime('%Y-%m-%d'),
                    'price': current_price,
                    'reason': 'signal_change',
                    'pnl_pct': pnl_pct * 100,
                    'capital': capital,
                    'entry_date': entry_date,
                    'entry_price': entry_price
                })
            
            # Open new position
            position = signal
            entry_price = current_price
            entry_date = current_date.strftime('%Y-%m-%d')
            
            # Set stop loss based on 6-day prediction magnitude
            predicted_return_6d = returns_6d[i]
            if not np.isnan(predicted_return_6d):
                stop_loss_pct = -STOP_LOSS_MULTIPLIER * abs(predicted_return_6d)
                if position == 1:
                    stop_loss = entry_price * (1 + stop_loss_pct)
                else:
                    stop_loss = entry_price * (1 - stop_loss_pct)
            else:
                # Default 5% stop loss if prediction not available
                if position == 1:
                    stop_loss = entry_price * 0.95
                else:
                    stop_loss = entry_price * 1.05
            
            trades.append({
                'type': 'entry',
                'position_type': 'long' if position == 1 else 'short',
                'date': entry_date,
                'price': entry_price,
                'signal_6d': int(pred_6d[i]),
                'signal_10d': int(pred_10d[i]),
                'stop_loss': stop_loss,
                'capital': capital
            })
    
    # Close final position if any
    if position != 0:
        final_date = test_data.index[-1]
        final_price = test_prices.iloc[-1]
        if position == 1:
            pnl_pct = (final_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - final_price) / entry_price
        capital = capital * (1 + pnl_pct)
        trades.append({
            'type': 'exit',
            'position_type': 'long' if position == 1 else 'short',
            'date': final_date.strftime('%Y-%m-%d'),
            'price': final_price,
            'reason': 'end',
            'pnl_pct': pnl_pct * 100,
            'capital': capital,
            'entry_date': entry_date,
            'entry_price': entry_price
        })
    
    return capital, trades, capital_history, benchmark_history

def save_results(final_capital, benchmark_capital, trades, capital_history, benchmark_history):
    """Save results to JSON file"""
    results = {
        'summary': {
            'final_capital': final_capital,
            'benchmark_capital': benchmark_capital,
            'strategy_return': ((final_capital / INITIAL_CAPITAL) - 1) * 100,
            'benchmark_return': ((benchmark_capital / INITIAL_CAPITAL) - 1) * 100,
            'total_trades': len([t for t in trades if t['type'] == 'entry']),
            'initial_capital': INITIAL_CAPITAL
        },
        'trades': trades,
        'capital_history': capital_history,
        'benchmark_history': benchmark_history,
        'hyperparameters': {
            'start_date': START_DATE,
            'lookback_days': LOOKBACK_DAYS,
            'prediction_horizons': PREDICTION_HORIZONS,
            'stop_loss_multiplier': STOP_LOSS_MULTIPLIER,
            'train_test_split': TRAIN_TEST_SPLIT
        }
    }
    
    with open('backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

# HTML template for web dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Bitcoin Trading Strategy Results</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .summary {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .summary-item {
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }
        .summary-item.benchmark {
            border-left-color: #2196F3;
        }
        .summary-item h3 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }
        .summary-item .value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        canvas {
            max-height: 500px;
        }
        .trades-table {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .entry {
            color: #4CAF50;
            font-weight: bold;
        }
        .exit {
            color: #f44336;
            font-weight: bold;
        }
        .positive {
            color: #4CAF50;
        }
        .negative {
            color: #f44336;
        }
    </style>
</head>
<body>
    <h1>Bitcoin Trading Strategy - Performance Dashboard</h1>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <h3>Strategy Final Capital</h3>
                <div class="value">${{ "%.2f"|format(results.summary.final_capital) }}</div>
            </div>
            <div class="summary-item">
                <h3>Strategy Return</h3>
                <div class="value {{ 'positive' if results.summary.strategy_return > 0 else 'negative' }}">
                    {{ "%.2f"|format(results.summary.strategy_return) }}%
                </div>
            </div>
            <div class="summary-item benchmark">
                <h3>Benchmark Final Capital</h3>
                <div class="value">${{ "%.2f"|format(results.summary.benchmark_capital) }}</div>
            </div>
            <div class="summary-item benchmark">
                <h3>Benchmark Return</h3>
                <div class="value {{ 'positive' if results.summary.benchmark_return > 0 else 'negative' }}">
                    {{ "%.2f"|format(results.summary.benchmark_return) }}%
                </div>
            </div>
            <div class="summary-item">
                <h3>Total Trades</h3>
                <div class="value">{{ results.summary.total_trades }}</div>
            </div>
            <div class="summary-item">
                <h3>Initial Capital</h3>
                <div class="value">${{ results.summary.initial_capital }}</div>
            </div>
        </div>
    </div>
    
    <div class="chart-container">
        <h2>Capital Development Over Time</h2>
        <canvas id="capitalChart"></canvas>
    </div>
    
    <div class="trades-table">
        <h2>Trade History</h2>
        <table>
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Position</th>
                    <th>Date</th>
                    <th>Price</th>
                    <th>Reason</th>
                    <th>P&L %</th>
                    <th>Capital</th>
                </tr>
            </thead>
            <tbody>
                {% for trade in results.trades %}
                <tr>
                    <td class="{{ trade.type }}">{{ trade.type.upper() }}</td>
                    <td>{{ trade.position_type.upper() }}</td>
                    <td>{{ trade.date }}</td>
                    <td>${{ "%.2f"|format(trade.price) }}</td>
                    <td>{{ trade.reason if trade.type == 'exit' else '-' }}</td>
                    <td class="{{ 'positive' if trade.get('pnl_pct', 0) > 0 else 'negative' if trade.get('pnl_pct', 0) < 0 else '' }}">
                        {{ "%.2f"|format(trade.pnl_pct) if trade.get('pnl_pct') is not none else '-' }}%
                    </td>
                    <td>${{ "%.2f"|format(trade.capital) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    <script>
        const results = {{ results|tojson }};
        
        // Prepare data for chart
        const dates = results.capital_history.map(h => h.date);
        const capitalData = results.capital_history.map(h => h.capital);
        const benchmarkData = results.benchmark_history.map(h => h.value);
        
        // Get trade markers
        const entryTrades = results.trades.filter(t => t.type === 'entry');
        const exitTrades = results.trades.filter(t => t.type === 'exit');
        
        const entryPoints = entryTrades.map(t => ({
            x: t.date,
            y: results.capital_history.find(h => h.date === t.date).capital
        }));
        
        const exitPoints = exitTrades.map(t => ({
            x: t.date,
            y: results.capital_history.find(h => h.date === t.date).capital
        }));
        
        // Create chart
        const ctx = document.getElementById('capitalChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [
                    {
                        label: 'Strategy Capital',
                        data: capitalData,
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        tension: 0.1,
                        pointRadius: 0
                    },
                    {
                        label: 'Buy & Hold Benchmark',
                        data: benchmarkData,
                        borderColor: '#2196F3',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        tension: 0.1,
                        pointRadius: 0
                    },
                    {
                        label: 'Entry Points',
                        data: entryPoints,
                        backgroundColor: '#4CAF50',
                        borderColor: '#4CAF50',
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        showLine: false
                    },
                    {
                        label: 'Exit Points',
                        data: exitPoints,
                        backgroundColor: '#f44336',
                        borderColor: '#f44336',
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        showLine: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += '$' + context.parsed.y.toFixed(2);
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(0);
                            }
                        }
                    },
                    x: {
                        ticks: {
                            maxTicksLimit: 20
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
"""

# Flask app
app = Flask(__name__)

# Global variable to store results
backtest_results = None

@app.route('/')
def dashboard():
    if backtest_results is None:
        return "Backtest results not available. Please run the backtest first.", 500
    return render_template_string(HTML_TEMPLATE, results=backtest_results)

def run_backtest():
    """Run the backtest and save results"""
    global backtest_results
    
    print("Fetching Bitcoin data from Binance...")
    df = fetch_binance_data()
    
    print("Preparing features and targets...")
    data = prepare_data(df)
    
    print("Training models...")
    model_6d, model_10d, scaler, X_test, y_6d_test, y_10d_test, test_data, split_idx = train_models(data)
    
    print("Running backtest...")
    final_capital, trades, capital_history, benchmark_history = backtest_strategy(
        df, data, model_6d, model_10d, scaler, split_idx
    )
    
    # Calculate benchmark
    test_prices = df.loc[test_data.index, 'close']
    initial_price = test_prices.iloc[0]
    final_price = test_prices.iloc[-1]
    benchmark_capital = INITIAL_CAPITAL * (final_price / initial_price)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Final Capital (Strategy): ${final_capital:.2f}")
    print(f"Benchmark (Buy & Hold):   ${benchmark_capital:.2f}")
    print(f"Strategy Return:          {((final_capital / INITIAL_CAPITAL) - 1) * 100:.2f}%")
    print(f"Benchmark Return:         {((benchmark_capital / INITIAL_CAPITAL) - 1) * 100:.2f}%")
    print(f"Number of Trades:         {len([t for t in trades if t['type'] == 'entry'])}")
    print("="*50)
    
    # Save results
    backtest_results = save_results(final_capital, benchmark_capital, trades, capital_history, benchmark_history)
    
    print(f"\nResults saved to backtest_results.json")
    print(f"Starting web server on port {PORT}...")

if __name__ == '__main__':
    # Run backtest first
    run_backtest()
    
    # Start web server
    app.run(host='0.0.0.0', port=PORT, debug=False)
