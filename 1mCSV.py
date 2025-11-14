import pandas as pd
import requests
from datetime import datetime, timedelta
from flask import Flask, send_file, render_template_string
import io
import os

app = Flask(__name__)

def fetch_binance_data(symbol='BTCUSDT', interval='1m', start_date='2018-01-01', end_date=None):
    """
    Fetch OHLCV data from Binance API
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    
    base_url = 'https://api.binance.com/api/v3/klines'
    all_data = []
    
    current_ts = start_ts
    batch_size = 1000  # Binance allows up to 1000 records per request
    
    print("Fetching Bitcoin data from Binance...")
    
    while current_ts < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_ts,
            'limit': batch_size
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_data.extend(data)
            
            # Update timestamp for next batch (last timestamp + 1 minute)
            current_ts = data[-1][0] + 60000
            
            # Print progress
            current_date = datetime.fromtimestamp(current_ts / 1000).strftime('%Y-%m-%d')
            print(f"Fetched data up to: {current_date}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
    
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    
    df = pd.DataFrame(all_data, columns=columns)
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Select and rename relevant columns
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    
    # Convert numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    
    # Filter data to the specified end date
    df = df[df['datetime'] <= pd.to_datetime(end_date)]
    
    print(f"Total records fetched: {len(df)}")
    return df

def save_data_to_csv(df, filename='bitcoin_ohlcv_1m.csv'):
    """Save DataFrame to CSV file"""
    df.to_csv(filename, index=False)
    return filename

@app.route('/')
def index():
    """Main page with download link"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bitcoin OHLCV Data Download</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 50px auto; 
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #333; }
            .info { 
                background: #e7f3ff; 
                padding: 15px; 
                border-radius: 5px; 
                margin: 20px 0;
            }
            .download-btn {
                display: inline-block;
                background: #007bff;
                color: white;
                padding: 12px 24px;
                text-decoration: none;
                border-radius: 5px;
                font-size: 16px;
                margin: 10px 0;
            }
            .download-btn:hover {
                background: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Bitcoin OHLCV Data</h1>
            <div class="info">
                <h3>Dataset Information:</h3>
                <p><strong>Symbol:</strong> BTC/USDT</p>
                <p><strong>Timeframe:</strong> 1 Minute</p>
                <p><strong>Date Range:</strong> 2018-01-01 to Today</p>
                <p><strong>Data Points:</strong> {{ data_points }} records</p>
                <p><strong>Columns:</strong> datetime, open, high, low, close, volume</p>
            </div>
            <p>Click the button below to download the complete dataset:</p>
            <a href="/download" class="download-btn">📥 Download CSV File</a>
            <p><small>The file contains OHLCV (Open, High, Low, Close, Volume) data for Bitcoin from Binance.</small></p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, data_points=len(global_df))

@app.route('/download')
def download_file():
    """Download the CSV file"""
    # Create a StringIO object to serve the file in memory
    csv_buffer = io.StringIO()
    global_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Create a BytesIO object from the StringIO
    mem = io.BytesIO()
    mem.write(csv_buffer.getvalue().encode('utf-8'))
    mem.seek(0)
    csv_buffer.close()
    
    filename = f"bitcoin_1m_ohlcv_{datetime.now().strftime('%Y%m%d')}.csv"
    
    return send_file(
        mem,
        as_attachment=True,
        download_name=filename,
        mimetype='text/csv'
    )

@app.route('/api/data')
def api_data():
    """Return data as JSON (first 1000 records for preview)"""
    return global_df.head(1000).to_json(orient='records', date_format='iso')

if __name__ == '__main__':
    # Fetch data when the script starts
    print("Starting data fetch from Binance...")
    global_df = fetch_binance_data(
        symbol='BTCUSDT',
        interval='1m',
        start_date='2018-01-01'
    )
    
    if global_df.empty:
        print("No data fetched. Exiting.")
        exit(1)
    
    print(f"Data fetch complete. Total records: {len(global_df)}")
    print("Starting web server on http://localhost:5000")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
