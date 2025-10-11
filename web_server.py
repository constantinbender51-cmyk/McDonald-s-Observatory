from flask import Flask, render_template_string
import pandas as pd
import os
from datetime import datetime
import glob
import sys

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BTC Trading Strategy Results</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 40px;
            font-size: 1.1em;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .metric-card h3 {
            margin: 0 0 10px 0;
            font-size: 0.9em;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-card .value {
            font-size: 2em;
            font-weight: bold;
            margin: 0;
        }
        .positive {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        .negative {
            background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        }
        .neutral {
            background: linear-gradient(135deg, #4776E6 0%, #8E54E9 100%);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border-radius: 10px;
            overflow: hidden;
        }
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 1px;
        }
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .section {
            margin-top: 50px;
        }
        .section h2 {
            color: #2c3e50;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 25px;
        }
        .chart-container {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .timestamp {
            text-align: center;
            color: #95a5a6;
            margin-top: 30px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 BTC Trading Strategy Results</h1>
        <p class="subtitle">Dual Logistic Regression Model with 3x Leverage</p>
        
        <div class="metrics-grid">
            <div class="metric-card neutral">
                <h3>Initial Capital</h3>
                <p class="value">${{ "%.2f"|format(initial_capital) }}</p>
            </div>
            <div class="metric-card {{ 'positive' if final_equity > initial_capital else 'negative' }}">
                <h3>Final Equity</h3>
                <p class="value">${{ "%.2f"|format(final_equity) }}</p>
            </div>
            <div class="metric-card {{ 'positive' if strategy_return > 0 else 'negative' }}">
                <h3>Strategy Return</h3>
                <p class="value">{{ "%.2f"|format(strategy_return) }}%</p>
            </div>
            <div class="metric-card {{ 'positive' if buy_hold_return > 0 else 'negative' }}">
                <h3>Buy & Hold Return</h3>
                <p class="value">{{ "%.2f"|format(buy_hold_return) }}%</p>
            </div>
            <div class="metric-card {{ 'positive' if strategy_return > buy_hold_return else 'negative' }}">
                <h3>Outperformance</h3>
                <p class="value">{{ "%.2f"|format(strategy_return - buy_hold_return) }}%</p>
            </div>
            <div class="metric-card neutral">
                <h3>Total Trades</h3>
                <p class="value">{{ num_trades }}</p>
            </div>
        </div>

        <div class="section">
            <h2>📊 Summary Metrics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {% for _, row in summary_data.iterrows() %}
                    <tr>
                        <td><strong>{{ row['Metric'] }}</strong></td>
                        <td>{{ row['Value'] }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        {% if trades_data is not none and trades_data|length > 0 %}
        <div class="section">
            <h2>💼 Recent Trades (Last 20)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Trade #</th>
                        <th>Type</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {% for idx, row in trades_display.iterrows() %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td><strong>{{ row['type']|upper }}</strong></td>
                        <td>${{ "%.2f"|format(row['entry_price']) }}</td>
                        <td>${{ "%.2f"|format(row['exit_price']) }}</td>
                        <td style="color: {{ 'green' if row['pnl'] > 0 else 'red' }}; font-weight: bold;">
                            ${{ "%.2f"|format(row['pnl']) }}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <div class="section">
            <h2>📈 Equity Curve Sample (Last 30 Days)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>BTC Price</th>
                        <th>Equity</th>
                        <th>Position</th>
                    </tr>
                </thead>
                <tbody>
                    {% for _, row in equity_display.iterrows() %}
                    <tr>
                        <td>{{ row['date'] }}</td>
                        <td>${{ "%.2f"|format(row['price']) }}</td>
                        <td>${{ "%.2f"|format(row['equity']) }}</td>
                        <td>{{ row['position'] if row['position'] else 'None' }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <p class="timestamp">Generated on {{ timestamp }}</p>
    </div>
</body>
</html>
"""

def load_latest_results():
    """Load the most recent results files"""
    results_files = glob.glob('btc_trading_results_*.csv')
    equity_files = glob.glob('btc_equity_curve_*.csv')
    trades_files = glob.glob('btc_trades_*.csv')
    
    if not results_files:
        return None, None, None
    
    latest_results = max(results_files, key=os.path.getctime)
    latest_equity = max(equity_files, key=os.path.getctime)
    latest_trades = max(trades_files, key=os.path.getctime) if trades_files else None
    
    summary_df = pd.read_csv(latest_results)
    equity_df = pd.read_csv(latest_equity)
    trades_df = pd.read_csv(latest_trades) if latest_trades else None
    
    return summary_df, equity_df, trades_df

@app.route('/')
def index():
    """Main route to display results"""
    summary_data, equity_data, trades_data = load_latest_results()
    
    if summary_data is None:
        return "<h1>No results found. Please run the trading strategy first.</h1>"
    
    metrics = dict(zip(summary_data['Metric'], summary_data['Value']))
    
    initial_capital = metrics.get('Initial Capital', 10000)
    final_equity = metrics.get('Final Equity', 0)
    strategy_return = metrics.get('Strategy Return (%)', 0)
    buy_hold_return = metrics.get('Buy & Hold Return (%)', 0)
    num_trades = int(metrics.get('Number of Trades', 0))
    
    equity_display = equity_data.tail(30) if len(equity_data) > 30 else equity_data
    trades_display = trades_data.tail(20) if trades_data is not None and len(trades_data) > 20 else trades_data
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    return render_template_string(
        HTML_TEMPLATE,
        summary_data=summary_data,
        equity_display=equity_display,
        trades_data=trades_data,
        trades_display=trades_display,
        initial_capital=initial_capital,
        final_equity=final_equity,
        strategy_return=strategy_return,
        buy_hold_return=buy_hold_return,
        num_trades=num_trades,
        timestamp=timestamp
    )

if __name__ == '__main__':
    # Check if results already exist
    results_files = glob.glob('btc_trading_results_*.csv')
    
    if not results_files:
        print("No existing results found. Running trading strategy first...")
        # Import and run the trading strategy
        try:
            from main import main as run_trading_strategy
            run_trading_strategy()
            print("Trading strategy completed successfully!")
        except Exception as e:
            print(f"Error running trading strategy: {e}")
            sys.exit(1)
    else:
        print("Found existing results. Starting web server...")
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
