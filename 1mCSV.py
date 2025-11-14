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
    'dataframe': None,
    'completion_time': None,
    'is_test_mode': False
}

def fetch_binance_data(symbol='BTCUSDT', interval='1m', start_date='2018-01-01', test_mode=False):
    """
    Fetch OHLCV data from Binance API
    """
    global fetching_status
    
    if test_mode:
        # For test mode, fetch only last 30 days
        end_date = datetime.now()
        start_date = (end_date - timedelta(days=30)).strftime('%Y-%m-%d')
        end_ts = int(end_date.timestamp() * 1000)
        print(f"Test mode: Fetching data from {start_date} to {end_date.strftime('%Y-%m-%d')}")
    else:
        # Use current time dynamically for end date
        end_date = datetime.now()
        end_ts = int(end_date.timestamp() * 1000)
    
    # Convert start date to timestamp
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    
    base_url = 'https://api.binance.com/api/v3/klines'
    all_data = []
    
    current_ts = start_ts
    batch_size = 1000  # Binance allows up to 1000 records per request
    
    fetching_status['is_fetching'] = True
    fetching_status['current_records'] = 0
    fetching_status['total_records'] = 0
    fetching_status['error'] = None
    fetching_status['current_date'] = start_date
    fetching_status['completion_time'] = None
    fetching_status['is_test_mode'] = test_mode
    
    print(f"Starting Bitcoin data fetch from Binance... {'(TEST MODE - 1 month)' if test_mode else ''}")
    
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
            
            # For test mode, break early if we have enough data (approx 1 month)
            if test_mode and len(all_data) >= 43200:  # 30 days * 24 hours * 60 minutes = 43200
                print("Test mode: Reached approximately 1 month of data, stopping...")
                break
        
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
            
            # Filter data to current time
            df = df[df['datetime'] <= datetime.now()]
            
            fetching_status['dataframe'] = df
            fetching_status['total_records'] = len(df)
            fetching_status['completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Data fetch complete. Total records: {len(df)}")
            
    except Exception as e:
        fetching_status['error'] = str(e)
        print(f"Error fetching data: {e}")
    
    finally:
        fetching_status['is_fetching'] = False

def start_fetching_thread(test_mode=False):
    """Start the data fetching in a separate thread"""
    if not fetching_status['is_fetching']:
        thread = threading.Thread(target=fetch_binance_data, kwargs={'test_mode': test_mode})
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
            .test-btn {
                background: #ffc107;
                color: #212529;
            }
            .test-btn:hover {
                background: #e0a800;
            }
            .btn:disabled {
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
            .last-update {
                font-size: 12px;
                color: #6c757d;
                margin-top: 10px;
            }
            .mode-indicator {
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
                margin-left: 10px;
            }
            .test-mode {
                background: #fff3cd;
                color: #856404;
                border: 1px solid #ffeaa7;
            }
            .full-mode {
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
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
                <p><strong>Full Dataset Range:</strong> 2018-01-01 to Current Time</p>
                <p><strong>Test Dataset Range:</strong> Last 30 days only</p>
                <p><strong>Expected Full Data:</strong> ~3+ million records (may take several minutes)</p>
                <p><strong>Expected Test Data:</strong> ~43,200 records (quick download)</p>
            </div>

            <div style="margin: 20px 0;">
                <button id="fetchBtn" class="btn fetch-btn" onclick="startFetching(false)">
                    🚀 Fetch Full Historical Data
                </button>
                <button id="testBtn" class="btn test-btn" onclick="startFetching(true)">
                    🧪 Test Download (Last 30 Days)
                </button>
            </div>

            <div id="progressContainer" class="progress-container">
                <h3>Fetching Data... <span id="modeIndicator" class="mode-indicator"></span></h3>
                <div class="progress-bar">
                    <div id="progressFill" class="progress-fill"></div>
                </div>
                <div id="statusText" class="status-text">Initializing...</div>
                <div id="currentDate" class="status-text"></div>
                <div id="lastUpdate" class="last-update">Last update: <span id="updateTime">-</span></div>
                <button id="cancelBtn" class="btn" style="background: #dc3545; color: white;" onclick="cancelFetching()">
                    ❌ Cancel Fetching
                </button>
            </div>

            <div id="errorContainer" class="error" style="display: none;"></div>

            <div id="successContainer" class="success" style="display: none;">
                <h3>✅ Data Fetching Complete! <span id="completionMode" class="mode-indicator"></span></h3>
                <p id="completionText"></p>
                <p><strong>Completion Time:</strong> <span id="completionTime"></span></p>
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
            let visibilityHandler;
            let lastProgressState = {};
            
            function startFetching(testMode) {
                const fetchBtn = document.getElementById('fetchBtn');
                const testBtn = document.getElementById('testBtn');
                
                fetchBtn.disabled = true;
                testBtn.disabled = true;
                
                if (testMode) {
                    fetchBtn.textContent = 'Waiting...';
                    testBtn.textContent = 'Starting Test...';
                } else {
                    fetchBtn.textContent = 'Starting...';
                    testBtn.textContent = 'Waiting...';
                }
                
                const endpoint = testMode ? '/start-test-fetch' : '/start-fetch';
                
                fetch(endpoint, { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            showProgress(testMode);
                            startProgressUpdate();
                            setupVisibilityHandler();
                        } else {
                            showError('Already fetching data or error starting process');
                            resetButtons();
                        }
                    })
                    .catch(error => {
                        showError('Error starting fetch: ' + error);
                        resetButtons();
                    });
            }
            
            function resetButtons() {
                document.getElementById('fetchBtn').disabled = false;
                document.getElementById('testBtn').disabled = false;
                document.getElementById('fetchBtn').textContent = '🚀 Fetch Full Historical Data';
                document.getElementById('testBtn').textContent = '🧪 Test Download (Last 30 Days)';
            }
            
            function cancelFetching() {
                fetch('/cancel-fetch', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            hideProgress();
                            resetButtons();
                            cleanupVisibilityHandler();
                        }
                    });
            }
            
            function showProgress(testMode) {
                const modeIndicator = document.getElementById('modeIndicator');
                if (testMode) {
                    modeIndicator.textContent = 'TEST MODE';
                    modeIndicator.className = 'mode-indicator test-mode';
                } else {
                    modeIndicator.textContent = 'FULL MODE';
                    modeIndicator.className = 'mode-indicator full-mode';
                }
                
                document.getElementById('progressContainer').style.display = 'block';
                document.getElementById('successContainer').style.display = 'none';
                document.getElementById('errorContainer').style.display = 'none';
                document.getElementById('preview').style.display = 'none';
            }
            
            function hideProgress() {
                document.getElementById('progressContainer').style.display = 'none';
            }
            
            function showSuccess(records, completionTime, isTestMode) {
                const completionMode = document.getElementById('completionMode');
                if (isTestMode) {
                    completionMode.textContent = 'TEST DATA';
                    completionMode.className = 'mode-indicator test-mode';
                } else {
                    completionMode.textContent = 'FULL DATA';
                    completionMode.className = 'mode-indicator full-mode';
                }
                
                document.getElementById('progressContainer').style.display = 'none';
                document.getElementById('successContainer').style.display = 'block';
                document.getElementById('completionText').textContent = 
                    `Successfully fetched ${records.toLocaleString()} records of Bitcoin OHLCV data.`;
                document.getElementById('completionTime').textContent = completionTime;
                
                resetButtons();
                
                // Show preview
                fetch('/api/data')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('previewData').textContent = JSON.stringify(data, null, 2);
                        document.getElementById('preview').style.display = 'block';
                    });
                    
                cleanupVisibilityHandler();
            }
            
            function showError(message) {
                document.getElementById('errorContainer').style.display = 'block';
                document.getElementById('errorContainer').textContent = message;
                cleanupVisibilityHandler();
            }
            
            function setupVisibilityHandler() {
                // Handle page visibility changes (tab switching, minimize, etc.)
                visibilityHandler = function() {
                    if (!document.hidden) {
                        // Page became visible, force immediate update
                        updateProgress(true);
                    }
                };
                
                document.addEventListener('visibilitychange', visibilityHandler);
            }
            
            function cleanupVisibilityHandler() {
                if (visibilityHandler) {
                    document.removeEventListener('visibilitychange', visibilityHandler);
                    visibilityHandler = null;
                }
            }
            
            function startProgressUpdate() {
                progressInterval = setInterval(() => updateProgress(), 1000);
            }
            
            function updateProgress(force = false) {
                // Use cache-busting to prevent browser caching
                const url = '/progress?' + new Date().getTime();
                
                fetch(url)
                    .then(response => response.json())
                    .then(data => {
                        const updateTime = document.getElementById('updateTime');
                        updateTime.textContent = new Date().toLocaleTimeString();
                        
                        // Check if state changed significantly
                        const stateChanged = 
                            data.is_fetching !== lastProgressState.is_fetching ||
                            data.current_records !== lastProgressState.current_records ||
                            data.total_records !== lastProgressState.total_records;
                        
                        lastProgressState = data;
                        
                        if (!data.is_fetching && data.total_records > 0) {
                            // Fetching completed
                            clearInterval(progressInterval);
                            showSuccess(data.total_records, data.completion_time || 'Unknown', data.is_test_mode);
                        } else if (!data.is_fetching && data.error) {
                            // Error occurred
                            clearInterval(progressInterval);
                            showError('Fetching error: ' + data.error);
                            resetButtons();
                        } else if (data.is_fetching || force || stateChanged) {
                            // Still fetching or force update or state changed
                            const progressFill = document.getElementById('progressFill');
                            const statusText = document.getElementById('statusText');
                            const currentDate = document.getElementById('currentDate');
                            
                            statusText.textContent = `Fetched: ${data.current_records.toLocaleString()} records`;
                            currentDate.textContent = `Current date: ${data.current_date}`;
                            
                            // Progress indicator - different max for test vs full mode
                            const maxRecords = data.is_test_mode ? 43200 : 3500000;
                            const progress = Math.min((data.current_records / maxRecords) * 100, 100);
                            progressFill.style.width = progress + '%';
                        }
                    })
                    .catch(error => {
                        console.error('Error updating progress:', error);
                        // Don't stop the interval on occasional errors
                    });
            }
            
            // Check initial state on page load
            window.addEventListener('load', function() {
                fetch('/progress?' + new Date().getTime())
                    .then(response => response.json())
                    .then(data => {
                        if (data.is_fetching) {
                            document.getElementById('fetchBtn').disabled = true;
                            document.getElementById('testBtn').disabled = true;
                            document.getElementById('fetchBtn').textContent = 'Fetching in progress...';
                            document.getElementById('testBtn').textContent = 'Fetching in progress...';
                            showProgress(data.is_test_mode);
                            startProgressUpdate();
                            setupVisibilityHandler();
                        } else if (data.total_records > 0) {
                            showSuccess(data.total_records, data.completion_time || 'Unknown', data.is_test_mode);
                        }
                    });
            });
            
            // Clean up when page unloads
            window.addEventListener('beforeunload', function() {
                if (progressInterval) {
                    clearInterval(progressInterval);
                }
                cleanupVisibilityHandler();
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/start-fetch', methods=['POST'])
def start_fetch():
    """Start the full data fetching process"""
    if fetching_status['is_fetching']:
        return jsonify({'success': False, 'message': 'Already fetching data'})
    
    success = start_fetching_thread(test_mode=False)
    return jsonify({'success': success})

@app.route('/start-test-fetch', methods=['POST'])
def start_test_fetch():
    """Start the test data fetching process (1 month)"""
    if fetching_status['is_fetching']:
        return jsonify({'success': False, 'message': 'Already fetching data'})
    
    success = start_fetching_thread(test_mode=True)
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
    # Add cache control headers to prevent caching
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
    
    # Different filename for test vs full mode
    if fetching_status['is_test_mode']:
        filename = f"bitcoin_1m_ohlcv_TEST_1month_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        filename = f"bitcoin_1m_ohlcv_FULL_historical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return send_file(
        mem,
        as_attachment=True,
        download_name=filename,
        mimetype='text/csv'
    )

if __name__ == '__main__':
    print("Starting web server on http://localhost:8080")
    print("Visit the page to fetch Bitcoin OHLCV data")
    print("Options:")
    print("  - 🚀 Fetch Full Historical Data: All data from 2018-01-01 to now (~3M+ records)")
    print("  - 🧪 Test Download (Last 30 Days): Quick download of last month only (~43K records)")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
