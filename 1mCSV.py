import pandas as pd
import requests
from datetime import datetime, timedelta
from flask import Flask, send_file, render_template_string, jsonify
import io
import threading
import time
import json

app = Flask(__name__)

# Global variables to track fetching progress
fetching_status = {
    'is_fetching': False,
    'current_records': 0,
    'total_records': 0,
    'current_date': '',
    'error': None,
    'dataframe': None
}

def fetch_binance_data(symbol='BTCUSDT', interval='1m', start_date='2018-01-01', end_date=None):
    """
    Fetch OHLCV data from Binance API
    """
    global fetching_status
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    
    base_url = 'https://api.binance.com/api/v3/klines'
    all_data = []
    
    current_ts = start_ts
    batch_size = 1000  # Binance allows up to 1000 records per request
    
    fetching_status['is_fetching'] = True
    fetching_status['current_records'] = 0
    fetching_status['total_records'] = 0
    fetching_status['error'] = None
    fetching_status['current_date'] = start_date
    
    print("Starting Bitcoin data fetch from Binance...")
    
    try:
        while current_ts < end_ts and fetching_status['is_fetching']:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_ts,
                'limit': batch_size
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_data.extend(data)
            
            # Update timestamp for next batch (last timestamp + 1 minute)
            current_ts = data[-1][0] + 60000
            
            # Update progress
            current_date = datetime.fromtimestamp(current_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
            fetching_status['current_records'] = len(all_data)
            fetching_status['current_date'] = current_date
            
            print(f"Fetched {len(all_data)} records up to: {current_date}")
            
            # Small delay to be respectful to the API
            time.sleep(0.1)
        
        if fetching_status['is_fetching']:  # Only process if not cancelled
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
            
            fetching_status['dataframe'] = df
            fetching_status['total_records'] = len(df)
            print(f"Data fetch complete. Total records: {len(df)}")
            
    except Exception as e:
        fetching_status['error'] = str(e)
        print(f"Error fetching data: {e}")
    
    finally:
        fetching_status['is_fetching'] = False

def start_fetching_thread():
    """Start the data fetching in a separate thread"""
    if not fetching_status['is_fetching']:
        thread = threading.Thread(target=fetch_binance_data)
        thread.daemon = True
        thread.start()
        return True
    return False

@app.route('/')
def index():
    """Main page with fetch button and progress display"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bitcoin OHLCV Data Fetcher</title>
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
                text-align: center;
            }
            h1 { color: #333; margin-bottom: 30px; }
            .info { 
                background: #e7f3ff; 
                padding: 15px; 
                border-radius: 5px; 
                margin: 20px 0;
                text-align: left;
            }
            .btn {
                display: inline-block;
                padding: 12px 24px;
                text-decoration: none;
                border-radius: 5px;
                font-size: 16px;
                margin: 10px 5px;
                border: none;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .fetch-btn {
                background: #28a745;
                color: white;
            }
            .fetch-btn:hover {
                background: #218838;
            }
            .fetch-btn:disabled {
                background: #6c757d;
                cursor: not-allowed;
            }
            .download-btn {
                background: #007bff;
                color: white;
            }
            .download-btn:hover {
                background: #0056b3;
            }
            .progress-container {
                margin: 20px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 5px;
                display: none;
            }
            .progress-bar {
                width: 100%;
                height: 20px;
                background: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }
            .progress-fill {
                height: 100%;
                background: #17a2b8;
                width: 0%;
                transition: width 0.3s ease;
            }
            .status-text {
                margin: 10px 0;
                font-weight: bold;
            }
            .error {
                color: #dc3545;
                background: #f8d7da;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .success {
                color: #155724;
                background: #d4edda;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Bitcoin OHLCV Data Fetcher</h1>
            
            <div class="info">
                <h3>Dataset Information:</h3>
                <p><strong>Symbol:</strong> BTC/USDT</p>
                <p><strong>Timeframe:</strong> 1 Minute</p>
                <p><strong>Date Range:</strong> 2018-01-01 to Today</p>
                <p><strong>Expected Data:</strong> ~3+ million records (this may take several minutes)</p>
            </div>

            <button id="fetchBtn" class="btn fetch-btn" onclick="startFetching()">
                🚀 Fetch 1 Minute Binance OHLCV Data
            </button>

            <div id="progressContainer" class="progress-container">
                <h3>Fetching Data...</h3>
                <div class="progress-bar">
                    <div id="progressFill" class="progress-fill"></div>
                </div>
                <div id="statusText" class="status-text">Initializing...</div>
                <div id="currentDate" class="status-text"></div>
                <button id="cancelBtn" class="btn" style="background: #dc3545; color: white;" onclick="cancelFetching()">
                    ❌ Cancel Fetching
                </button>
            </div>

            <div id="errorContainer" class="error" style="display: none;"></div>

            <div id="successContainer" class="success" style="display: none;">
                <h3>✅ Data Fetching Complete!</h3>
                <p id="completionText"></p>
                <a id="downloadLink" href="/download" class="btn download-btn">
                    📥 Download CSV File
                </a>
            </div>

            <div id="preview" style="display: none; margin-top: 30px; text-align: left;">
                <h3>Data Preview (First 5 Records):</h3>
                <pre id="previewData"></pre>
            </div>
        </div>

        <script>
            let progressInterval;
            
            function startFetching() {
                const btn = document.getElementById('fetchBtn');
                btn.disabled = true;
                btn.textContent = 'Starting...';
                
                fetch('/start-fetch', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            showProgress();
                            startProgressUpdate();
                        } else {
                            showError('Already fetching data or error starting process');
                            btn.disabled = false;
                            btn.textContent = '🚀 Fetch 1 Minute Binance OHLCV Data';
                        }
                    })
                    .catch(error => {
                        showError('Error starting fetch: ' + error);
                        btn.disabled = false;
                        btn.textContent = '🚀 Fetch 1 Minute Binance OHLCV Data';
                    });
            }
            
            function cancelFetching() {
                fetch('/cancel-fetch', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            hideProgress();
                            document.getElementById('fetchBtn').disabled = false;
                            document.getElementById('fetchBtn').textContent = '🚀 Fetch 1 Minute Binance OHLCV Data';
                        }
                    });
            }
            
            function showProgress() {
                document.getElementById('progressContainer').style.display = 'block';
                document.getElementById('successContainer').style.display = 'none';
                document.getElementById('errorContainer').style.display = 'none';
                document.getElementById('preview').style.display = 'none';
            }
            
            function hideProgress() {
                document.getElementById('progressContainer').style.display = 'none';
            }
            
            function showSuccess(records) {
                document.getElementById('progressContainer').style.display = 'none';
                document.getElementById('successContainer').style.display = 'block';
                document.getElementById('completionText').textContent = 
                    `Successfully fetched ${records.toLocaleString()} records of Bitcoin OHLCV data.`;
                document.getElementById('fetchBtn').disabled = false;
                document.getElementById('fetchBtn').textContent = '🚀 Fetch Again';
                
                // Show preview
                fetch('/api/data')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('previewData').textContent = JSON.stringify(data, null, 2);
                        document.getElementById('preview').style.display = 'block';
                    });
            }
            
            function showError(message) {
                document.getElementById('errorContainer').style.display = 'block';
                document.getElementById('errorContainer').textContent = message;
            }
            
            function startProgressUpdate() {
                progressInterval = setInterval(updateProgress, 1000);
            }
            
            function updateProgress() {
                fetch('/progress')
                    .then(response => response.json())
                    .then(data => {
                        if (!data.is_fetching && data.total_records > 0) {
                            // Fetching completed
                            clearInterval(progressInterval);
                            showSuccess(data.total_records);
                        } else if (!data.is_fetching && data.error) {
                            // Error occurred
                            clearInterval(progressInterval);
                            showError('Fetching error: ' + data.error);
                            document.getElementById('fetchBtn').disabled = false;
                            document.getElementById('fetchBtn').textContent = '🚀 Fetch 1 Minute Binance OHLCV Data';
                        } else if (data.is_fetching) {
                            // Still fetching
                            const progressFill = document.getElementById('progressFill');
                            const statusText = document.getElementById('statusText');
                            const currentDate = document.getElementById('currentDate');
                            
                            statusText.textContent = `Fetched: ${data.current_records.toLocaleString()} records`;
                            currentDate.textContent = `Current date: ${data.current_date}`;
                            
                            // Simple progress indicator (since we don't know total in advance)
                            const progress = Math.min((data.current_records / 3000000) * 100, 100);
                            progressFill.style.width = progress + '%';
                        }
                    })
                    .catch(error => {
                        console.error('Error updating progress:', error);
                    });
            }
            
            // Check initial state on page load
            window.addEventListener('load', function() {
                fetch('/progress')
                    .then(response => response.json())
                    .then(data => {
                        if (data.is_fetching) {
                            document.getElementById('fetchBtn').disabled = true;
                            document.getElementById('fetchBtn').textContent = 'Fetching in progress...';
                            showProgress();
                            startProgressUpdate();
                        } else if (data.total_records > 0) {
                            showSuccess(data.total_records);
                        }
                    });
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/start-fetch', methods=['POST'])
def start_fetch():
    """Start the data fetching process"""
    if fetching_status['is_fetching']:
        return jsonify({'success': False, 'message': 'Already fetching data'})
    
    success = start_fetching_thread()
    return jsonify({'success': success})

@app.route('/cancel-fetch', methods=['POST'])
def cancel_fetch():
    """Cancel the data fetching process"""
    global fetching_status
    fetching_status['is_fetching'] = False
    return jsonify({'success': True})

@app.route('/progress')
def get_progress():
    """Get the current fetching progress"""
    return jsonify(fetching_status)

@app.route('/api/data')
def api_data():
    """Return data as JSON (first 5 records for preview)"""
    if fetching_status['dataframe'] is not None and not fetching_status['dataframe'].empty:
        return fetching_status['dataframe'].head().to_json(orient='records', date_format='iso')
    return jsonify([])

@app.route('/download')
def download_file():
    """Download the CSV file"""
    if fetching_status['dataframe'] is None or fetching_status['dataframe'].empty:
        return "No data available. Please fetch data first.", 400
    
    # Create a StringIO object to serve the file in memory
    csv_buffer = io.StringIO()
    fetching_status['dataframe'].to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Create a BytesIO object from the StringIO
    mem = io.BytesIO()
    mem.write(csv_buffer.getvalue().encode('utf-8'))
    mem.seek(0)
    csv_buffer.close()
    
    filename = f"bitcoin_1m_ohlcv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return send_file(
        mem,
        as_attachment=True,
        download_name=filename,
        mimetype='text/csv'
    )

if __name__ == '__main__':
    print("Starting web server on http://localhost:5000")
    print("Visit the page and click 'Fetch 1 Minute Binance OHLCV Data' to start downloading historical data")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
