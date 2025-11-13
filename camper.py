import numpy as np
import pandas as pd
import requests
from datetime import datetime
import time
from sklearn.preprocessing import StandardScaler

# Fetch BTC/USDT hourly data from Binance
def fetch_binance_data():
    print("Fetching hourly BTC/USDT data from Binance...")
    
    start_date = int(datetime(2018, 1, 1).timestamp() * 1000)
    end_date = int(datetime.now().timestamp() * 1000)
    
    all_data = []
    current_start = start_date
    
    while current_start < end_date:
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1h',
            'startTime': current_start,
            'limit': 1000
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if not data or len(data) == 0:
                break
            
            all_data.extend(data)
            current_start = data[-1][0] + 3600000  # +1 hour in milliseconds
            
            print(f"Fetched {len(all_data)} hourly candles...")
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    print(f"Total candles fetched: {len(df)}")
    return df

# Calculate MACD
def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

# Calculate RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate Stochastic RSI
def calculate_stoch_rsi(prices, period=14):
    rsi = calculate_rsi(prices, period)
    stoch_rsi = (rsi - rsi.rolling(window=period).min()) / \
                (rsi.rolling(window=period).max() - rsi.rolling(window=period).min())
    return stoch_rsi.fillna(0.5)

# Prepare features
def prepare_features(df):
    print("Calculating technical indicators...")
    
    # Calculate indicators
    df['price_change'] = df['close'].pct_change() * 100
    df['volume_change'] = df['volume'].pct_change() * 100
    df['macd_diff'] = calculate_macd(df['close'])
    df['stoch_rsi'] = calculate_stoch_rsi(df['close'])
    
    # Fill NaN values
    df = df.fillna(0)
    
    features = []
    targets = []
    
    # Create 24-hour windows
    for i in range(24, len(df) - 24):
        feature = np.concatenate([
            df['price_change'].iloc[i-24:i].values,
            df['volume_change'].iloc[i-24:i].values,
            df['macd_diff'].iloc[i-24:i].values,
            df['stoch_rsi'].iloc[i-24:i].values
        ])
        
        # Target: next day (24 hours) price direction
        future_price = df['close'].iloc[i + 24]
        current_price = df['close'].iloc[i]
        target = 1 if future_price > current_price else 0
        
        features.append(feature)
        targets.append(target)
    
    print(f"Created {len(features)} samples with 96 features each")
    
    return np.array(features), np.array(targets), df

# Neural Network Implementation
class NeuralNetwork:
    def __init__(self):
        self.layers = [96, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 1]
        self.weights = []
        self.biases = []
        
        # He initialization
        for i in range(len(self.layers) - 1):
            scale = np.sqrt(2.0 / self.layers[i])
            w = np.random.randn(self.layers[i], self.layers[i+1]) * scale
            b = np.zeros(self.layers[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        # Adam optimizer parameters
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.t = 0
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            if i == len(self.weights) - 1:  # Output layer
                a = self.sigmoid(z)
            else:  # Hidden layers
                a = self.relu(z)
            
            activations.append(a)
        
        return activations, z_values
    
    def backward(self, X, y, activations, z_values):
        m = X.shape[0]
        grad_weights = [np.zeros_like(w) for w in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        delta = activations[-1] - y.reshape(-1, 1)
        
        # Backpropagate
        for i in range(len(self.weights) - 1, -1, -1):
            grad_weights[i] = np.dot(activations[i].T, delta) / m
            grad_biases[i] = np.mean(delta, axis=0)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(z_values[i-1])
        
        return grad_weights, grad_biases
    
    def adam_update(self, grad_weights, grad_biases, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1
        
        for i in range(len(self.weights)):
            # Update weights
            self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * grad_weights[i]
            self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * (grad_weights[i] ** 2)
            
            m_hat = self.m_weights[i] / (1 - beta1 ** self.t)
            v_hat = self.v_weights[i] / (1 - beta2 ** self.t)
            
            self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Update biases
            self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * grad_biases[i]
            self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * (grad_biases[i] ** 2)
            
            m_hat = self.m_biases[i] / (1 - beta1 ** self.t)
            v_hat = self.v_biases[i] / (1 - beta2 ** self.t)
            
            self.biases[i] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        print(f"Training for {epochs} epochs with batch size {batch_size}...")
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            total_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                activations, z_values = self.forward(X_batch)
                
                # Calculate loss (binary cross-entropy)
                predictions = activations[-1]
                loss = -np.mean(y_batch.reshape(-1, 1) * np.log(predictions + 1e-15) + 
                               (1 - y_batch.reshape(-1, 1)) * np.log(1 - predictions + 1e-15))
                total_loss += loss
                
                # Backward pass
                grad_weights, grad_biases = self.backward(X_batch, y_batch, activations, z_values)
                
                # Update weights
                self.adam_update(grad_weights, grad_biases)
            
            if epoch % 10 == 0:
                avg_loss = total_loss / (len(X_train) / batch_size)
                print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
        
        print("Training complete!")
    
    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1].flatten()

# Simulate trading strategy
def simulate_trading(predictions, prices):
    capital = 1000
    equity = [capital]
    
    for i in range(len(predictions) - 1):
        signal = 1 if predictions[i] > 0.5 else -1
        price_change = (prices[i + 1] - prices[i]) / prices[i]
        
        if signal == 1:  # Long
            capital *= (1 + price_change)
        else:  # Short
            capital *= (1 - price_change)
        
        equity.append(capital)
    
    # Calculate max drawdown
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_drawdown = np.max(drawdown) * 100
    
    return capital, max_drawdown

# Main execution
def main():
    print("=" * 60)
    print("Bitcoin Price Direction Prediction Neural Network")
    print("=" * 60)
    
    # Fetch data
    df = fetch_binance_data()
    
    # Prepare features
    X, y, df_processed = prepare_features(df)
    
    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split (80/20)
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Initialize and train network
    nn = NeuralNetwork()
    nn.train(X_train, y_train, epochs=50, batch_size=32)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions = nn.predict(X_test)
    
    # Calculate accuracy
    pred_labels = (predictions > 0.5).astype(int)
    accuracy = np.mean(pred_labels == y_test) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Simulate trading
    print("\nSimulating trading strategy...")
    test_prices = df_processed['close'].iloc[-(len(predictions) + 25):].values
    final_equity, max_drawdown = simulate_trading(predictions, test_prices[24:])
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Starting Capital: $1,000.00")
    print(f"Final Equity: ${final_equity:.2f}")
    print(f"Total Return: {((final_equity / 1000) - 1) * 100:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
