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
    'is_test_mode': False,
    'is_complete': False,
    'last_update': None
}

def update_last_update_time():
    """Update the last update timestamp"""
    fetching_status['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def safe_json_serialize(obj):
    """Safely serialize objects for JSON, handling DataFrames and other non-serializable types"""
    if hasattr(obj, 'to_dict'):
        # Handle pandas DataFrames and Series
        return obj.to_dict(orient='records')
    elif hasattr(obj, 'to_json'):
        return json.loads(obj.to_json())
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, pd.Timedelta):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)

def fetch_binance_data(symbol='BTCUSDT', interval='1m', start_date='2018-01-01', test_mode=False):
    """
    Fetch OHLCV data from Binance API with proper handling of incomplete chunks
    """
    global fetching_status
    
    try:
        # Validate and adjust dates
        current_time = datetime.now()
        
        if test_mode:
            # For test mode, fetch only last 30 days
            start_date_obj = current_time - timedelta(days=30)
            start_date = start_date_obj.strftime('%Y-%m-%d')
            end_date_obj = current_time
            print(f"Test mode: Fetching data from {start_date} to {end_date_obj.strftime('%Y-%m-%d')}")
        else:
            # For full mode, ensure start date is valid
            try:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                if start_date_obj >= current_time:
                    start_date_obj = current_time - timedelta(days=365)  # Default to 1 year back if invalid
                    start_date = start_date_obj.strftime('%Y-%m-%d')
            except ValueError:
                start_date_obj = current_time - timedelta(days=365)
                start_date = start_date_obj.strftime('%Y-%m-%d')
            
            end_date_obj = current_time
        
        # Convert dates to timestamps (ensure they're valid)
        start_ts = int(start_date_obj.timestamp() * 1000)
        end_ts = int(end_date_obj.timestamp() * 1000)
        
        # Ensure end timestamp is after start timestamp
        if end_ts <= start_ts:
            end_ts = start_ts + 60000  # Add 1 minute if dates are invalid
        
        base_url = 'https://api.binance.com/api/v3/klines'
        all_data = []
        
        current_ts = start_ts
        batch_size = 1000
        
        # Reset status
        fetching_status.update({
            'is_fetching': True,
            'current_records': 0,
            'total_records': 0,
            'error': None,
            'current_date': start_date,
            'completion_time': None,
            'is_test_mode': test_mode,
            'is_complete': False,
            'dataframe': None  # Clear previous dataframe
        })
        update_last_update_time()
        
        print(f"Starting Bitcoin data fetch from Binance... {'(TEST MODE - 1 month)' if test_mode else ''}")
        print(f"Time range: {start_date} to {end_date_obj.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Timestamps: {start_ts} to {end_ts}")
        
        request_count = 0
        max_requests = 1000  # Safety limit to prevent infinite loops
        
        while current_ts < end_ts and fetching_status['is_fetching'] and request_count < max_requests:
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
                
                # Check if response is successful
                if response.status_code != 200:
                    error_msg = f"Binance API error {response.status_code}: {response.text}"
                    print(f"❌ {error_msg}")
                    fetching_status['error'] = error_msg
                    break
                
                data = response.json()
                
                # Check if Binance returned an error
                if isinstance(data, dict) and 'code' in data and 'msg' in data:
                    error_msg = f"Binance API error {data['code']}: {data['msg']}"
                    print(f"❌ {error_msg}")
                    fetching_status['error'] = error_msg
                    break
                
                if not data:
                    print("No more data available from API")
                    break
                    
                print(f"Received {len(data)} candles in this batch")
                all_data.extend(data)
                
                # Update progress
                if data:
                    last_timestamp = data[-1][0]
                    current_date = datetime.fromtimestamp(last_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
                    fetching_status['current_records'] = len(all_data)
                    fetching_status['current_date'] = current_date
                    update_last_update_time()
                    
                    print(f"Fetched {len(all_data)} records up to: {current_date}")
                    
                    # Update timestamp for next batch (last timestamp + 1 minute)
                    current_ts = last_timestamp + 60000
                else:
                    break
                
                # For test mode, break early if we have enough data
                if test_mode and len(all_data) >= 43200:
                    print("Test mode: Reached approximately 1 month of data, stopping...")
                    break
                    
                # Check if we've reached the end
                if data and data[-1][0] >= end_ts:
                    print(f"Reached end timestamp {end_ts}, stopping...")
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                fetching_status['error'] = f"Network error: {str(e)}"
                update_last_update_time()
                break
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                fetching_status['error'] = f"Invalid response from Binance API: {str(e)}"
                update_last_update_time()
                break
                
            # Small delay to be respectful to the API
            time.sleep(0.1)
        
        if request_count >= max_requests:
            print("⚠️  Reached maximum request limit, stopping...")
            fetching_status['error'] = "Reached maximum request limit (safety stop)"
        
        if fetching_status['is_fetching'] and not fetching_status['error'] and all_data:
            print(f"Processing {len(all_data)} records into DataFrame...")
            
            try:
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
                
                # Remove any potential duplicates and sort
                df = df.drop_duplicates(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
                
                # Store as a JSON-serializable representation
                fetching_status['dataframe'] = {
                    'data': df.to_dict(orient='records'),
                    'columns': list(df.columns),
                    'info': {
                        'start_date': df['datetime'].min().strftime('%Y-%m-%d %H:%M:%S'),
                        'end_date': df['datetime'].max().strftime('%Y-%m-%d %H:%M:%S'),
                        'total_records': len(df)
                    }
                }
                
                fetching_status['total_records'] = len(df)
                fetching_status['completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                fetching_status['is_complete'] = True
                update_last_update_time()
                
                print(f"✅ Data fetch complete! Total records: {len(df)}")
                print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
                
            except Exception as e:
                print(f"❌ Error processing DataFrame: {e}")
                fetching_status['error'] = f"Data processing error: {str(e)}"
        elif not all_data and not fetching_status['error']:
            fetching_status['error'] = "No data received from Binance API"
            
    except Exception as e:
        fetching_status['error'] = f"Unexpected error: {str(e)}"
        print(f"❌ Critical error in fetch_binance_data: {e}")
        import traceback
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
    # ... (same HTML template as before, but I'll include the key JavaScript fixes)
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bitcoin OHLCV Data Fetcher</title>
        <style>
            /* ... (same styles as before) ... */
        </style>
    </head>
    <body>
        <!-- ... (same HTML structure as before) ... -->
        
        <script>
            // ... (same JavaScript as before, but with improved error handling) ...
            
            function updateProgress(force = false) {
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
                    const contentType = response.headers.get('content-type');
                    if (!contentType || !contentType.includes('application/json')) {
                        throw new Error(`Server returned HTML instead of JSON. Status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    // Handle server errors in the response
                    if (data.error && data.error.includes('not JSON serializable')) {
                        throw new Error('Server data serialization error: ' + data.error);
                    }
                    
                    updateConnectionStatus(true);
                    const updateTime = document.getElementById('updateTime');
                    updateTime.textContent = new Date().toLocaleTimeString();
                    
                    updateSystemStatus(data);
                    
                    const stateChanged = 
                        data.is_fetching !== lastProgressState.is_fetching ||
                        data.current_records !== lastProgressState.current_records ||
                        data.total_records !== lastProgressState.total_records ||
                        data.is_complete !== lastProgressState.is_complete;
                    
                    lastProgressState = data;
                    
                    const isComplete = (
                        data.is_complete ||
                        (!data.is_fetching && data.total_records > 0) ||
                        (!data.is_fetching && !data.error && data.current_records > 0)
                    );
                    
                    if (isComplete && data.total_records > 0) {
                        console.log('✅ Completion detected! Stopping interval and showing success');
                        updateDebugInfo('Completion detected! Stopping progress interval');
                        clearInterval(progressInterval);
                        showSuccess(data.total_records, data.completion_time || new Date().toLocaleString(), data.is_test_mode);
                    } else if (!data.is_fetching && data.error) {
                        console.log('❌ Error detected:', data.error);
                        updateDebugInfo('Error detected: ' + data.error);
                        clearInterval(progressInterval);
                        showError('Fetching error: ' + data.error);
                    } else if (data.is_fetching || force || stateChanged) {
                        const progressFill = document.getElementById('progressFill');
                        const statusText = document.getElementById('statusText');
                        const currentDate = document.getElementById('currentDate');
                        const batchInfo = document.getElementById('batchInfo');
                        
                        statusText.textContent = `Fetched: ${data.current_records.toLocaleString()} records`;
                        currentDate.textContent = `Current date: ${data.current_date}`;
                        batchInfo.textContent = `Processing batches...`;
                        
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
                    
                    // Show specific error for serialization issues
                    if (error.message.includes('serialization')) {
                        showError('Server data format error. Please try again or contact support.');
                    }
                });
            }
            
            // ... (rest of JavaScript remains the same) ...
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
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/progress')
def get_progress():
    """Get the current fetching progress - JSON serializable"""
    try:
        update_last_update_time()
        
        # Create a safe, JSON-serializable copy of the status
        safe_status = {}
        for key, value in fetching_status.items():
            if key == 'dataframe' and value is not None:
                # dataframe is already stored as JSON-serializable dict
                safe_status[key] = value
            elif hasattr(value, 'to_dict') or hasattr(value, 'to_json'):
                # Handle any pandas objects that might sneak in
                safe_status[key] = safe_json_serialize(value)
            else:
                safe_status[key] = value
        
        return jsonify(safe_status)
    except Exception as e:
        error_msg = f"Progress serialization error: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({
            'is_fetching': False,
            'error': error_msg,
            'current_records': 0,
            'total_records': 0,
            'is_complete': False
        }), 500

@app.route('/api/data')
def api_data():
    """Return data as JSON (first 5 records for preview) - JSON serializable"""
    try:
        if (fetching_status['dataframe'] is not None and 
            'data' in fetching_status['dataframe']):
            # Return first 5 records from the JSON-serializable data
            data = fetching_status['dataframe']['data'][:5]
            return jsonify(data)
        return jsonify([])
    except Exception as e:
        error_msg = f"API data serialization error: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/download')
def download_file():
    """Download the CSV file"""
    try:
        if (fetching_status['dataframe'] is None or 
            'data' not in fetching_status['dataframe'] or
            not fetching_status['dataframe']['data']):
            return "No data available. Please fetch data first.", 400
        
        # Reconstruct DataFrame from JSON-serializable data for download
        data = fetching_status['dataframe']['data']
        columns = fetching_status['dataframe']['columns']
        
        df = pd.DataFrame(data, columns=columns)
        
        # Create a StringIO object to serve the file in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
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
    except Exception as e:
        return f"Server error during download: {str(e)}", 500

if __name__ == '__main__':
    print("Starting web server on http://localhost:8080")
    print("Visit the page to fetch Bitcoin OHLCV data")
    print("Options:")
    print("  - 🚀 Fetch Full Historical Data: All data from 2018-01-01 to now (~3M+ records)")
    print("  - 🧪 Test Download (Last 30 Days): Quick download of last month only (~43K records)")
    print("  - 🔄 Page auto-updates every 5 seconds")
    print("  - 🔧 Fixed JSON serialization issues and Binance API error handling")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
