from flask import Flask, render_template_string
import pandas as pd
import os
from datetime import datetime
import glob
import sys
import json

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BTC Trading Strategy Results</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
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
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }
        .timestamp {
            text-align: center;
            color: #95a5a6;
            margin-top: 30px;
            font-size: 0.9em;
        }
        #equityCurve {
            width: 100%;
            height: 600px;
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
            <h2>📈 Equity Curve Performance</h2>
            <div class="chart-container">
                <div id="equityCurve"></div>
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
            <h2>📋 Equity Curve Sample (Last 30 Days)</h2>
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

    <script>
        var chartData = {{ chart_data | safe }};
        
        var traces = [
            {
                x: chartData.dates,
                y: chartData.strategy_equity,
                type: 'scatter',
                mode: 'lines',
                name: 'Strategy Equity',
                line: {
                    color: '#667eea',
                    width: 2.5
                }
            },
            {
                x: chartData.dates,
                y: chartData.buy_hold_equity,
                type: 'scatter',
                mode: 'lines',
                name: 'Buy & Hold',
                line: {
                    color: '#95a5a6',
                    width: 2,
                    dash: 'dash'
                }
            }
        ];

        if (chartData.entry_dates.length > 0) {
            traces.push({
                x: chartData.entry_dates,
                y: chartData.entry_values,
                type: 'scatter',
                mode: 'markers',
                name: 'Entry',
                marker: {
                    color: '#38ef7d',
                    size: 10,
                    symbol: 'triangle-up',
                    line: {
                        color: '#11998e',
                        width: 2
                    }
                }
            });
        }

        if (chartData.exit_dates.length > 0) {
            traces.push({
                x: chartData.exit_dates,
                y: chartData.exit_values,
                type: 'scatter',
                mode: 'markers',
                name: 'Exit',
                marker: {
                    color: '#ff6a00',
                    size: 10,
                    symbol: 'triangle-down',
                    line: {
                        color: '#ee0979',
                        width: 2
                    }
                }
            });
        }

        var layout = {
            title: {
                text: 'Strategy Performance vs Buy & Hold Benchmark',
                font: {
                    size: 20,
                    color: '#2c3e50'
                }
            },
            xaxis: {
                title: 'Date',
                showgrid: true,
                gridcolor: '#ecf0f1'
            },
            yaxis: {
                title: 'Equity ($)',
                showgrid: true,
                gridcolor: '#ecf0f1'
            },
            hovermode: 'x unified',
            plot_bgcolor: 'white',
            paper_bgcolor: '#f8f9fa',
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: '#bdc3c7',
                borderwidth: 1
            },
            margin: {
                l: 60,
                r: 30,
                t: 80,
                b: 60
            }
        };

        var config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };

        Plotly.newPlot('equityCurve', traces, layout, config);
    </script>
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

def prepare_chart_data(equity_df, trades_df, initial_capital):
    """Prepare data for the equity curve chart"""
    equity_df = equity_df.copy()
    
    if 'date' not in equity_df.columns:
        equity_df['date'] = equity_df.index
    
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    
    if 'price' in equity_df.columns and initial_capital:
        first_price = equity_df['price'].iloc[0]
        equity_df['buy_hold_equity'] = initial_capital * (equity_df['price'] / first_price)
    else:
        equity_df['buy_hold_equity'] = initial_capital
    
    entry_dates = []
    entry_values = []
    exit_dates = []
    exit_values = []
    
    if trades_df is not None and len(trades_df) > 0:
        trades_df = trades_df.copy()
        
        if 'entry_date' in trades_df.columns:
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            for _, trade in trades_df.iterrows():
                entry_date = trade['entry_date']
                equity_at_entry = equity_df[equity_df['date'] <= entry_date]['equity'].iloc[-1] if len(equity_df[equity_df['date'] <= entry_date]) > 0 else None
                
                if equity_at_entry is not None:
                    entry_dates.append(entry_date.strftime('%Y-%m-%d'))
                    entry_values.append(equity_at_entry)
        
        if 'exit_date' in trades_df.columns:
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            for _, trade in trades_df.iterrows():
                exit_date = trade['exit_date']
                equity_at_exit = equity_df[equity_df['date'] <= exit_date]['equity'].iloc[-1] if len(equity_df[equity_df['date'] <= exit_date]) > 0 else None
                
                if equity_at_exit is not None:
                    exit_dates.append(exit_date.strftime('%Y-%m-%d'))
                    exit_values.append(equity_at_exit)
    
    chart_data = {
        'dates': equity_df['date'].dt.strftime('%Y-%m-%d').tolist(),
        'strategy_equity': equity_df['equity'].tolist(),
        'buy_hold_equity': equity_df['buy_hold_equity'].tolist(),
        'entry_dates': entry_dates,
        'entry_values': entry_values,
        'exit_dates': exit_dates,
        'exit_values': exit_values
    }
    
    return json.dumps(chart_data)

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
    
    chart_data = prepare_chart_data(equity_data, trades_data, initial_capital)
    
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
        timestamp=timestamp,
        chart_data=chart_data
    )

if __name__ == '__main__':
    results_files = glob.glob('btc_trading_results_*.csv')
    
    if not results_files:
        print("No existing results found. Running trading strategy first...")
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
