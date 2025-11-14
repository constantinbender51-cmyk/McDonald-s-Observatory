from flask import Flask, render_template_string, Response, send_file
import requests
import pandas as pd
from datetime import datetime
import time
import threading
import io

app = Flask(__name__)

# Global variables
data = []
progress = 0
is_complete = False
fetch_started = False

# Configuration
START_DATE = "2018-01-01 00:00:00"
END_DATE = datetime.now()  # Fetch until now
CANDLES_TO_FETCH = 4000  # For testing - will fetch all available if set to None
SYMBOL = "BTCUSDT"
INTERVAL = "1m"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Bitcoin Data Fetcher</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            background-color: white;
        }
        #progress {
            font-size: 24px;
            margin-bottom: 20px;
        }
        #download-btn {
            display: none;
            padding: 15px 30px;
            font-size: 18px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        #download-btn:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div id="progress">Starting...</div>
    <button id="download-btn" onclick="window.location.href='/download'">Download 1m.csv</button>
    
    <script>
        const eventSource = new EventSource('/progress');
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            document.getElementById('progress').textContent = 'Progress: ' + data.progress + '% complete';
            
            if (data.complete) {
                document.getElementById('download-btn').style.display = 'block';
                eventSource.close();
            }
        };
    </script>
</body>
</html>
"""

def fetch_binance_data():
    global data, progress, is_complete, fetch_started
    
    fetch_started = True
    print("Starting data fetch...")
    
    # Convert start date to milliseconds timestamp
    start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    end_ts = int(END_DATE.timestamp() * 1000)
    
    # Calculate total expected candles for progress tracking
    total_minutes = (end_ts - start_ts) // 60000
    target_candles = CANDLES_TO_FETCH if CANDLES_TO_FETCH else total_minutes
    
    endpoint = "https://api.binance.com/api/v3/klines"
    limit = 1000  # Binance max limit per request
    
    current_ts = start_ts
    fetched = 0
    
    while True:
        # Stop if we've reached the target for testing, or if fetching all and reached end time
        if CANDLES_TO_FETCH and fetched >= CANDLES_TO_FETCH:
            break
        if not CANDLES_TO_FETCH and current_ts >= end_ts:
            break
            
        try:
            params = {
                'symbol': SYMBOL,
                'interval': INTERVAL,
                'startTime': current_ts,
                'limit': limit
            }
            
            # Add endTime if fetching all data
            if not CANDLES_TO_FETCH:
                params['endTime'] = end_ts
            
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            candles = response.json()
            
            if not candles:
                break
            
            for candle in candles:
                data.append({
                    'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            fetched += len(candles)
            progress = int((fetched / target_candles) * 100)
            progress = min(progress, 100)  # Cap at 100%
            
            print(f"Fetched {fetched:,} candles ({progress}%)")
            
            # Update timestamp for next request
            current_ts = candles[-1][0] + 60000  # Add 1 minute in milliseconds
            
            # Respect rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(1)
    
    is_complete = True
    print(f"Data fetch complete! Total candles: {len(data):,}")

@app.route('/')
def index():
    # Start fetching data in background if not already started
    if not fetch_started:
        thread = threading.Thread(target=fetch_binance_data, daemon=True)
        thread.start()
    return render_template_string(HTML_TEMPLATE)

@app.route('/progress')
def progress_stream():
    def generate():
        while not is_complete:
            yield f"data: {{'progress': {progress}, 'complete': false}}\n\n"
            time.sleep(0.5)
        yield f"data: {{'progress': 100, 'complete': true}}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/download')
def download():
    if not is_complete or not data:
        return "Data not ready yet", 400
    
    # Create DataFrame and CSV
    df = pd.DataFrame(data)
    
    # Create in-memory file
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    # Convert to bytes
    mem = io.BytesIO()
    mem.write(output.getvalue().encode())
    mem.seek(0)
    
    return send_file(
        mem,
        mimetype='text/csv',
        as_attachment=True,
        download_name='1m.csv'
    )

if __name__ == '__main__':
    print("Starting web server on http://localhost:8080")
    if CANDLES_TO_FETCH:
        print(f"TEST MODE: Will fetch {CANDLES_TO_FETCH:,} candles starting from {START_DATE}")
        print(f"To fetch ALL candles, set CANDLES_TO_FETCH = None")
    else:
        print(f"Will fetch ALL candles from {START_DATE} to now")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
