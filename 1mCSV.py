import pandas as pd
import requests
from datetime import datetime, timedelta
from flask import Flask, send_file, render_template_string, jsonify
import io
import threading
import time
import json
import traceback

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
    'is_test_mode': False,
    'is_complete': False,
    'last_update': None
}

def update_last_update_time():
    """Update the last update timestamp"""
    fetching_status['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def fetch_binance_data(symbol='BTCUSDT', interval='1m', start_date='2018-01-01', test_mode=False):
    """
    Fetch OHLCV data from Binance API with proper handling of incomplete chunks
    """
    global fetching_status
    
    try:
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
        
        # Reset status
        fetching_status.update({
            'is_fetching': True,
            'current_records': 0,
            'total_records': 0,
            'error': None,
            'current_date': start_date,
            'completion_time': None,
            'is_test_mode': test_mode,
            'is_complete': False
        })
        update_last_update_time()
        
        print(f"Starting Bitcoin data fetch from Binance... {'(TEST MODE - 1 month)' if test_mode else ''}")
        print(f"Time range: {start_date} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        request_count = 0
        while current_ts < end_ts and fetching_status['is_fetching']:
            request_count += 1
            
            # Calculate the actual limit for this request
            remaining_minutes = (end_ts - current_ts) / 60000
            actual_limit = min(batch_size, max(1, int(remaining_minutes) + 1))
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_ts,
                'limit': actual_limit
            }
            
            print(f"Request #{request_count}: TS {current_ts} to {end_ts} (limit: {actual_limit})")
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    print("No more data available from API")
                    break
                    
                print(f"Received {len(data)} candles in this batch")
                all_data.extend(data)
                
                # Update progress
                current_date = datetime.fromtimestamp(data[-1][0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                fetching_status['current_records'] = len(all_data)
                fetching_status['current_date'] = current_date
                update_last_update_time()
                
                print(f"Fetched {len(all_data)} records up to: {current_date}")
                
                # Update timestamp for next batch (last timestamp + 1 minute)
                current_ts = data[-1][0] + 60000
                
                # For test mode, break early if we have enough data
                if test_mode and len(all_data) >= 43200:
                    print("Test mode: Reached approximately 1 month of data, stopping...")
                    break
                    
                # Check if we've reached the end
                if data[-1][0] >= end_ts:
                    print(f"Reached end timestamp {end_ts}, stopping...")
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                fetching_status['error'] = f"Network error: {str(e)}"
                update_last_update_time()
                break
                
            # Small delay to be respectful to the API
            time.sleep(0.1)
        
        if fetching_status['is_fetching'] and not fetching_status['error']:  # Only process if not cancelled and no error
            print(f"Processing {len(all_data)} records into DataFrame...")
            
            if not all_data:
                print("No data fetched!")
                fetching_status['error'] = "No data received from Binance API"
                return
            
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
            
            # Remove any potential duplicates
            df = df.drop_duplicates(subset=['datetime'])
            
            fetching_status['dataframe'] = df
            fetching_status['total_records'] = len(df)
            fetching_status['completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            fetching_status['is_complete'] = True
            update_last_update_time()
            
            print(f"✅ Data fetch complete! Total records: {len(df)}")
            print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            
    except Exception as e:
        fetching_status['error'] = f"Unexpected error: {str(e)}"
        print(f"❌ Critical error in fetch_binance_data: {e}")
        traceback.print_exc()
        update_last_update_time()
    
    finally:
        fetching_status['is_fetching'] = False
        update_last_update_time()
        print("Fetching process ended")

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
                display: inline-block;
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
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
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
            .blink {
                animation: blink 1s infinite;
            }
            @keyframes blink {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .auto-update-indicator {
                position: fixed;
                top: 10px;
                right: 10px;
                background: #28a745;
                color: white;
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
                z-index: 1000;
            }
            .status-indicator {
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            .status-idle {
                background: #f8f9fa;
                color: #6c757d;
                border: 1px solid #dee2e6;
            }
            .status-fetching {
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }
            .status-complete {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status-error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .debug-info {
                background: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                font-family: monospace;
                font-size: 12px;
                text-align: left;
                margin: 10px 0;
                display: none;
                max-height: 200px;
                overflow-y: auto;
            }
            .connection-status {
                margin: 10px 0;
                padding: 8px;
                border-radius: 5px;
                font-size: 14px;
            }
            .connection-ok {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .connection-error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
        </style>
    </head>
    <body>
        <div class="auto-update-indicator" title="Page auto-updates every 5 seconds">
            🔄 Auto-Update
        </div>

        <div class="container">
            <h1>📊 Bitcoin OHLCV Data Fetcher</h1>
            
            <div id="connectionStatus" class="connection-status connection-ok">
                ✅ Server Connection: OK
            </div>
            
            <div id="statusIndicator" class="status-indicator status-idle">
                💤 System Ready - Page auto-updates every 5 seconds
            </div>
            
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

            <button onclick="toggleDebug()" class="btn" style="background: #6c757d; color: white; font-size: 12px;">
                🔧 Toggle Debug Info
            </button>

            <div id="debugInfo" class="debug-info">
                <strong>Debug Information:</strong><br>
                <div id="debugContent">No debug information available</div>
            </div>

            <div id="progressContainer" class="progress-container">
                <h3>Fetching Data... <span id="modeIndicator" class="mode-indicator"></span></h3>
                <div class="progress-bar">
                    <div id="progressFill" class="progress-fill"></div>
                </div>
                <div id="statusText" class="status-text">Initializing...</div>
                <div id="currentDate" class="status-text"></div>
                <div id="batchInfo" class="status-text" style="font-size: 14px; color: #6c757d;"></div>
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
                <div style="margin: 20px 0;">
                    <a id="downloadLink" href="/download" class="btn download-btn blink" style="font-size: 18px; padding: 15px 30px;">
                        📥 DOWNLOAD CSV FILE NOW
                    </a>
                </div>
                <p><em>Click the button above to download your data</em></p>
            </div>

            <div id="preview" style="display: none; margin-top: 30px; text-align: left;">
                <h3>Data Preview (First 5 Records):</h3>
                <pre id="previewData"></pre>
            </div>

            <div class="last-update">
                System last updated: <span id="systemUpdateTime">-</span>
            </div>
        </div>

        <script>
            let progressInterval;
            let autoUpdateInterval;
            let visibilityHandler;
            let lastProgressState = {};
            let completionDetected = false;
            let debugEnabled = false;
            let consecutiveErrors = 0;
            const MAX_CONSECUTIVE_ERRORS = 3;
            
            function toggleDebug() {
                debugEnabled = !debugEnabled;
                document.getElementById('debugInfo').style.display = debugEnabled ? 'block' : 'none';
                updateDebugInfo('Debug ' + (debugEnabled ? 'enabled' : 'disabled'));
            }
            
            function updateDebugInfo(message) {
                if (debugEnabled) {
                    const debugContent = document.getElementById('debugContent');
                    const timestamp = new Date().toLocaleTimeString();
                    debugContent.innerHTML = `[${timestamp}] ${message}<br>` + debugContent.innerHTML;
                    
                    // Limit debug content to last 50 lines
                    const lines = debugContent.innerHTML.split('<br>');
                    if (lines.length > 50) {
                        debugContent.innerHTML = lines.slice(0, 50).join('<br>');
                    }
                }
            }
            
            function updateConnectionStatus(isConnected, message = '') {
                const statusElement = document.getElementById('connectionStatus');
                if (isConnected) {
                    statusElement.className = 'connection-status connection-ok';
                    statusElement.innerHTML = '✅ Server Connection: OK';
                    consecutiveErrors = 0;
                } else {
                    statusElement.className = 'connection-status connection-error';
                    statusElement.innerHTML = '❌ Server Connection: ERROR - ' + message;
                    consecutiveErrors++;
                    
                    if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
                        showError('Lost connection to server. Please refresh the page and try again.');
                    }
                }
            }
            
            // Initialize auto-update
            function initializeAutoUpdate() {
                // Start 5-second auto-update interval
                autoUpdateInterval = setInterval(() => {
                    updateDisplay();
                }, 5000);
                
                console.log('🔄 Auto-update initialized: refreshing every 5 seconds');
                updateDebugInfo('Auto-update initialized: refreshing every 5 seconds');
            }
            
            function updateDisplay() {
                const url = '/progress?' + new Date().getTime();
                
                fetch(url, {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    // Add timeout to prevent hanging requests
                    signal: AbortSignal.timeout(10000)
                })
                .then(response => {
                    // Check if response is JSON
                    const contentType = response.headers.get('content-type');
                    if (!contentType || !contentType.includes('application/json')) {
                        throw new Error(`Server returned non-JSON response: ${contentType}`);
                    }
                    return response.json();
                })
                .then(data => {
                    updateConnectionStatus(true);
                    updateSystemStatus(data);
                    updateSystemUpdateTime(data.last_update);
                    
                    if (!isCurrentlyFetching()) {
                        handleStateChange(data);
                    }
                })
                .catch(error => {
                    console.error('Auto-update error:', error);
                    updateConnectionStatus(false, error.message);
                    updateDebugInfo('Auto-update error: ' + error.message);
                    
                    // Don't show error if we're not in an active fetch
                    if (!isCurrentlyFetching()) {
                        updateSystemStatus({ error: 'Connection error' });
                    }
                });
            }
            
            function updateSystemStatus(data) {
                const statusIndicator = document.getElementById('statusIndicator');
                const systemUpdateTime = document.getElementById('systemUpdateTime');
                
                // Update system update time
                systemUpdateTime.textContent = new Date().toLocaleTimeString();
                
                // Update status indicator
                if (data.is_fetching) {
                    statusIndicator.className = 'status-indicator status-fetching';
                    statusIndicator.innerHTML = '🔄 Fetching Data - Auto-updating...';
                } else if (data.is_complete && data.total_records > 0) {
                    statusIndicator.className = 'status-indicator status-complete';
                    statusIndicator.innerHTML = '✅ Data Ready - Auto-updating...';
                } else if (data.error) {
                    statusIndicator.className = 'status-indicator status-error';
                    statusIndicator.innerHTML = '❌ Error - Auto-updating...';
                } else {
                    statusIndicator.className = 'status-indicator status-idle';
                    statusIndicator.innerHTML = '💤 System Ready - Page auto-updates every 5 seconds';
                }
                
                // Update debug info
                if (debugEnabled) {
                    updateDebugInfo(`State: fetching=${data.is_fetching}, complete=${data.is_complete}, records=${data.current_records}, error=${data.error}`);
                }
            }
            
            function updateSystemUpdateTime(lastUpdate) {
                const element = document.getElementById('systemUpdateTime');
                if (lastUpdate) {
                    element.textContent = lastUpdate;
                } else {
                    element.textContent = new Date().toLocaleTimeString();
                }
            }
            
            function isCurrentlyFetching() {
                return document.getElementById('progressContainer').style.display === 'block';
            }
            
            function handleStateChange(data) {
                // Check if state changed significantly
                const stateChanged = 
                    data.is_fetching !== lastProgressState.is_fetching ||
                    data.current_records !== lastProgressState.current_records ||
                    data.total_records !== lastProgressState.total_records ||
                    data.is_complete !== lastProgressState.is_complete;
                
                lastProgressState = data;
                
                // COMPLETION DETECTION - Multiple conditions
                const isComplete = (
                    data.is_complete ||
                    (!data.is_fetching && data.total_records > 0) ||
                    (!data.is_fetching && !data.error && data.current_records > 0)
                );
                
                if (isComplete && data.total_records > 0 && !completionDetected) {
                    // Fetching completed successfully
                    console.log('✅ Auto-update detected completion!');
                    updateDebugInfo('Auto-update detected completion! Records: ' + data.total_records);
                    showSuccess(data.total_records, data.completion_time || new Date().toLocaleString(), data.is_test_mode);
                } else if (!data.is_fetching && data.error && !isCurrentlyFetching()) {
                    // Error occurred
                    console.log('❌ Auto-update detected error:', data.error);
                    updateDebugInfo('Auto-update detected error: ' + data.error);
                    showError('Fetching error: ' + data.error);
                }
            }
            
            function startFetching(testMode) {
                completionDetected = false;
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
                
                updateDebugInfo('Starting fetch: ' + (testMode ? 'TEST mode' : 'FULL mode'));
                
                fetch(endpoint, { 
                    method: 'POST',
                    signal: AbortSignal.timeout(10000)
                })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                        }
                        return response.json();
                    })
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
                updateDebugInfo('Cancelling fetch...');
                fetch('/cancel-fetch', { 
                    method: 'POST',
                    signal: AbortSignal.timeout(10000)
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            hideProgress();
                            resetButtons();
                            cleanupVisibilityHandler();
                        }
                    })
                    .catch(error => {
                        console.error('Cancel fetch error:', error);
                        updateDebugInfo('Cancel fetch error: ' + error);
                        // Still reset buttons even if cancel fails
                        hideProgress();
                        resetButtons();
                        cleanupVisibilityHandler();
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
                if (completionDetected) return; // Prevent multiple triggers
                completionDetected = true;
                
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
                fetch('/api/data', { signal: AbortSignal.timeout(10000) })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('previewData').textContent = JSON.stringify(data, null, 2);
                        document.getElementById('preview').style.display = 'block';
                    })
                    .catch(error => {
                        console.error('Error fetching preview:', error);
                    });
                    
                cleanupVisibilityHandler();
                
                console.log('✅ Success screen shown with download link');
                updateDebugInfo('Success screen shown with download link. Records: ' + records);
            }
            
            function showError(message) {
                document.getElementById('errorContainer').style.display = 'block';
                document.getElementById('errorContainer').textContent = message;
                resetButtons();
                cleanupVisibilityHandler();
                updateDebugInfo('Error shown: ' + message);
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
                
                fetch(url, {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                    'Cache-Control': 'no-cache'
                    },
                    signal: AbortSignal.timeout(10000)
                })
                .then(response => {
                    // Check if response is JSON
                    const contentType = response.headers.get('content-type');
                    if (!contentType || !contentType.includes('application/json')) {
                        throw new Error(`Server returned HTML instead of JSON. Status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    updateConnectionStatus(true);
                    const updateTime = document.getElementById('updateTime');
                    updateTime.textContent = new Date().toLocaleTimeString();
                    
                    // Update system status
                    updateSystemStatus(data);
                    
                    // Check if state changed significantly
                    const stateChanged = 
                        data.is_fetching !== lastProgressState.is_fetching ||
                        data.current_records !== lastProgressState.current_records ||
                        data.total_records !== lastProgressState.total_records ||
                        data.is_complete !== lastProgressState.is_complete;
                    
                    lastProgressState = data;
                    
                    // COMPLETION DETECTION - Multiple conditions
                    const isComplete = (
                        data.is_complete ||
                        (!data.is_fetching && data.total_records > 0) ||
                        (!data.is_fetching && !data.error && data.current_records > 0)
                    );
                    
                    if (isComplete && data.total_records > 0) {
                        // Fetching completed successfully
                        console.log('✅ Completion detected! Stopping interval and showing success');
                        updateDebugInfo('Completion detected! Stopping progress interval');
                        clearInterval(progressInterval);
                        showSuccess(data.total_records, data.completion_time || new Date().toLocaleString(), data.is_test_mode);
                    } else if (!data.is_fetching && data.error) {
                        // Error occurred
                        console.log('❌ Error detected:', data.error);
                        updateDebugInfo('Error detected: ' + data.error);
                        clearInterval(progressInterval);
                        showError('Fetching error: ' + data.error);
                    } else if (data.is_fetching || force || stateChanged) {
                        // Still fetching or force update or state changed
                        const progressFill = document.getElementById('progressFill');
                        const statusText = document.getElementById('statusText');
                        const currentDate = document.getElementById('currentDate');
                        const batchInfo = document.getElementById('batchInfo');
                        
                        statusText.textContent = `Fetched: ${data.current_records.toLocaleString()} records`;
                        currentDate.textContent = `Current date: ${data.current_date}`;
                        batchInfo.textContent = `Processing batches with dynamic limit adjustment...`;
                        
                        // Progress indicator - different max for test vs full mode
                        const maxRecords = data.is_test_mode ? 43200 : 3500000;
                        const progress = data.current_records > 0 ? Math.min((data.current_records / maxRecords) * 100, 100) : 0;
                        progressFill.style.width = progress + '%';
                        
                        if (debugEnabled) {
                            updateDebugInfo(`Progress: ${progress.toFixed(1)}% (${data.current_records}/${maxRecords})`);
                        }
                    }
                })
                .catch(error => {
                    console.error('Error updating progress:', error);
                    updateConnectionStatus(false, error.message);
                    updateDebugInfo('Progress update error: ' + error.message);
                });
            }
            
            // Initialize when page loads
            window.addEventListener('load', function() {
                // Initialize auto-update
                initializeAutoUpdate();
                
                // Do initial state check
                fetch('/progress?' + new Date().getTime(), { 
                    signal: AbortSignal.timeout(10000)
                })
                    .then(response => {
                        const contentType = response.headers.get('content-type');
                        if (!contentType || !contentType.includes('application/json')) {
                            throw new Error(`Server returned HTML instead of JSON. Status: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        updateConnectionStatus(true);
                        updateSystemStatus(data);
                        updateSystemUpdateTime(data.last_update);
                        
                        if (data.is_fetching) {
                            document.getElementById('fetchBtn').disabled = true;
                            document.getElementById('testBtn').disabled = true;
                            document.getElementById('fetchBtn').textContent = 'Fetching in progress...';
                            document.getElementById('testBtn').textContent = 'Fetching in progress...';
                            showProgress(data.is_test_mode);
                            startProgressUpdate();
                            setupVisibilityHandler();
                        } else if (data.total_records > 0 || data.is_complete) {
                            showSuccess(data.total_records, data.completion_time || 'Unknown', data.is_test_mode);
                        }
                    })
                    .catch(error => {
                        console.error('Initial load error:', error);
                        updateConnectionStatus(false, error.message);
                        updateDebugInfo('Initial load error: ' + error.message);
                    });
            });
            
            // Clean up when page unloads
            window.addEventListener('beforeunload', function() {
                if (progressInterval) {
                    clearInterval(progressInterval);
                }
                if (autoUpdateInterval) {
                    clearInterval(autoUpdateInterval);
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
    try:
        if fetching_status['is_fetching']:
            return jsonify({'success': False, 'message': 'Already fetching data'})
        
        success = start_fetching_thread(test_mode=False)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/start-test-fetch', methods=['POST'])
def start_test_fetch():
    """Start the test data fetching process (1 month)"""
    try:
        if fetching_status['is_fetching']:
            return jsonify({'success': False, 'message': 'Already fetching data'})
        
        success = start_fetching_thread(test_mode=True)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/cancel-fetch', methods=['POST'])
def cancel_fetch():
    """Cancel the data fetching process"""
    try:
        fetching_status['is_fetching'] = False
        update_last_update_time()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success':
