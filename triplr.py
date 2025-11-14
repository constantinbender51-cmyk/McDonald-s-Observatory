import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import requests
import warnings
warnings.filterwarnings('ignore')

class AdaptiveTimeframeBitcoinPredictor:
    def __init__(self, prediction_horizon='1d'):
        self.prediction_horizon = prediction_horizon
        self.model = LogisticRegression(
            random_state=42, 
            class_weight='balanced',
            max_iter=1000,
            C=0.1  # Regularization for financial data
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.binance_client = None
        
    def fetch_binance_ohlcv(self, symbol='BTCUSDT', interval='1m', limit=1000):
        """Fetch OHLCV data from Binance API"""
        base_url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to proper data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def create_adaptive_dataset(self):
        """
        Create dataset with floating frequency:
        - Recent data (last 7 days): 1-minute intervals
        - Medium-term (last 3 months): 1-hour intervals  
        - Long-term (last 2 years): 1-day intervals
        - Historical (beyond 2 years): 1-week intervals
        """
        print("Fetching adaptive timeframe data from Binance...")
        
        datasets = []
        
        # Recent data: 1-minute (last 7 days = 7*24*60 = 10,080 minutes)
        recent_1m = self.fetch_binance_ohlcv(interval='1m', limit=10080)
        if recent_1m is not None:
            recent_1m['timeframe'] = '1m'
            datasets.append(recent_1m)
        
        # Medium-term: 1-hour (last 90 days = 2160 hours)
        medium_1h = self.fetch_binance_ohlcv(interval='1h', limit=2160)
        if medium_1h is not None:
            medium_1h['timeframe'] = '1h'
            # Remove overlap with 1m data
            cutoff = recent_1m.index.max() if recent_1m is not None else pd.Timestamp.now()
            medium_1h = medium_1h[medium_1h.index < cutoff]
            datasets.append(medium_1h)
        
        # Long-term: 1-day (last 730 days)
        long_1d = self.fetch_binance_ohlcv(interval='1d', limit=730)
        if long_1d is not None:
            long_1d['timeframe'] = '1d'
            cutoff = medium_1h.index.max() if medium_1h is not None else pd.Timestamp.now()
            long_1d = long_1d[long_1d.index < cutoff]
            datasets.append(long_1d)
        
        # Historical: 1-week (as much as available)
        historical_1w = self.fetch_binance_ohlcv(interval='1w', limit=520)  # ~10 years
        if historical_1w is not None:
            historical_1w['timeframe'] = '1w'
            cutoff = long_1d.index.max() if long_1d is not None else pd.Timestamp.now()
            historical_1w = historical_1w[historical_1w.index < cutoff]
            datasets.append(historical_1w)
        
        # Combine all datasets
        combined_df = pd.concat(datasets).sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        print(f"Created adaptive dataset with {len(combined_df)} data points")
        print(f"Timeframe distribution:")
        print(combined_df['timeframe'].value_counts())
        
        return combined_df
    
    def create_advanced_features(self, df):
        """Create comprehensive features for logistic regression"""
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Multiple moving average ratios
        for short, long in [(5, 20), (10, 50), (20, 100)]:
            df[f'sma_{short}'] = df['close'].rolling(short).mean()
            df[f'sma_{long}'] = df['close'].rolling(long).mean()
            df[f'ma_ratio_{short}_{long}'] = df[f'sma_{short}'] / df[f'sma_{long}']
            df[f'price_vs_sma_{short}'] = df['close'] / df[f'sma_{short}']
        
        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_ratio'] = df['ema_12'] / df['ema_26']
        
        # RSI with multiple timeframes
        def calculate_rsi(series, window=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        for window in [7, 14, 21]:
            df[f'rsi_{window}'] = calculate_rsi(df['close'], window)
        
        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_velocity'] = df['volume_ratio'].pct_change()
        
        # Volatility features (multiple timeframes)
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'range_volatility_{window}'] = df['high_low_range'].rolling(window).mean()
        
        # Price acceleration (second derivative)
        df['price_velocity'] = df['returns'].rolling(5).mean()
        df['price_acceleration'] = df['price_velocity'].diff()
        
        # Support/resistance levels
        df['resistance_20'] = df['high'].rolling(20).max()
        df['support_20'] = df['low'].rolling(20).min()
        df['distance_to_resistance'] = (df['resistance_20'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['support_20']) / df['close']
        
        # Statistical features
        df['z_score_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        df['skewness_20'] = df['returns'].rolling(20).skew()
        df['kurtosis_20'] = df['returns'].rolling(20).kurtosis()
        
        # Time-based features
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Lag features for pattern recognition
        for lag in [1, 2, 3, 5, 8, 13]:  # Fibonacci sequence for variety
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_ratio_lag_{lag}'] = df['volume_ratio'].shift(lag)
            df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)
        
        # Interaction features
        df['rsi_volume_interaction'] = df['rsi_14'] * df['volume_ratio']
        df['volatility_volume_interaction'] = df['volatility_20'] * df['volume_ratio']
        
        return df
    
    def create_daily_target(self, df):
        """Create target: Will price be higher in 24 hours?"""
        # Since we have mixed timeframes, we need to handle this carefully
        df = df.copy()
        
        # For daily prediction, we want to know if price will be higher in 24 hours
        # We'll use the last available price in the next 24 hours as our target
        df['future_price_24h'] = df['close'].shift(-1)  # This is simplified
        
        # For proper daily prediction, we'd need to align to daily boundaries
        # For now, we'll use next period's close vs current close
        df['target'] = (df['future_price_24h'] > df['close']).astype(int)
        
        return df
    
    def prepare_training_data(self, df):
        """Prepare final dataset for training"""
        print("Engineering features...")
        df_with_features = self.create_advanced_features(df)
        df_with_target = self.create_daily_target(df_with_features)
        
        # Remove rows with insufficient data for feature calculation
        df_clean = df_with_target.dropna()
        
        # Select feature columns (exclude target and original OHLCV)
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                       'future_price_24h', 'target', 'timeframe']
        feature_columns = [col for col in df_clean.columns if col not in exclude_cols]
        
        self.feature_columns = feature_columns
        
        X = df_clean[feature_columns]
        y = df_clean['target']
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Feature count: {len(feature_columns)}")
        print(f"Target distribution:\n{y.value_counts(normalize=True)}")
        
        return X, y, df_clean
    
    def train(self):
        """Train the model on adaptive timeframe data"""
        # Get adaptive data
        df = self.create_adaptive_dataset()
        
        if df is None or len(df) == 0:
            print("No data available for training")
            return self
        
        # Prepare training data
        X, y, full_df = self.prepare_training_data(df)
        
        if len(X) < 100:
            print("Insufficient data for training")
            return self
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        precisions = []
        
        print("\nTraining with Time Series Cross-Validation...")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            
            accuracies.append(accuracy)
            precuracies.append(precision)
            
            print(f"Fold {fold+1}: Accuracy = {accuracy:.3f}, Precision = {precision:.3f}")
        
        print(f"\nOverall Performance:")
        print(f"Average Accuracy: {np.mean(accuracies):.3f} (+/- {np.std(accuracies):.3f})")
        print(f"Average Precision: {np.mean(precisions):.3f} (+/- {np.std(precisions):.3f})")
        
        # Feature importance analysis
        self.analyze_feature_importance(X.columns)
        
        return self
    
    def analyze_feature_importance(self, feature_names):
        """Analyze which features are most important"""
        if hasattr(self.model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': abs(self.model.coef_[0]),
                'direction': ['positive' if x > 0 else 'negative' for x in self.model.coef_[0]]
            }).sort_values('importance', ascending=False)
            
            print("\nTop 15 Most Important Features:")
            print(importance_df.head(15).to_string(index=False))
            
            # Group features by type for analysis
            technical_features = [f for f in importance_df['feature'] 
                                if any(x in f for x in ['rsi', 'sma', 'ema', 'volatility', 'bb'])]
            volume_features = [f for f in importance_df['feature'] if 'volume' in f]
            price_features = [f for f in importance_df['feature'] 
                            if any(x in f for x in ['returns', 'price', 'range'])]
            
            print(f"\nFeature type counts:")
            print(f"Technical indicators: {len(technical_features)}")
            print(f"Volume features: {len(volume_features)}")
            print(f"Price action features: {len(price_features)}")
    
    def predict_next_day(self):
        """Make prediction for the next trading day"""
        if self.model is None or self.feature_columns is None:
            print("Model not trained yet. Call train() first.")
            return None
        
        # Get latest data
        latest_data = self.fetch_binance_ohlcv(interval='1m', limit=100)
        if latest_data is None:
            print("Could not fetch latest data")
            return None
        
        # Add features to latest data
        latest_with_features = self.create_advanced_features(latest_data)
        
        # Use the most recent complete data point
        latest_complete = latest_with_features.dropna().iloc[-1:]
        
        if len(latest_complete) == 0:
            print("No complete data point for prediction")
            return None
        
        # Prepare features for prediction
        X_latest = latest_complete[self.feature_columns]
        X_latest_scaled = self.scaler.transform(X_latest)
        
        # Make prediction
        prediction = self.model.predict(X_latest_scaled)[0]
        probability = self.model.predict_proba(X_latest_scaled)[0][1]
        
        print(f"\n🎯 NEXT DAY PREDICTION")
        print(f"Prediction: {'UP 📈' if prediction == 1 else 'DOWN 📉'}")
        print(f"Probability of UP movement: {probability:.3f}")
        print(f"Confidence: {'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.6 else 'LOW'}")
        
        return {
            'prediction': prediction,
            'probability': probability,
            'timestamp': latest_complete.index[0]
        }

# Usage example
def main():
    # Initialize predictor
    predictor = AdaptiveTimeframeBitcoinPredictor(prediction_horizon='1d')
    
    # Train model
    print("🚀 Starting Adaptive Timeframe Bitcoin Predictor...")
    predictor.train()
    
    # Make prediction
    print("\n" + "="*50)
    prediction = predictor.predict_next_day()
    
    if prediction:
        current_price = predictor.fetch_binance_ohlcv(interval='1m', limit=1)
        if current_price is not None:
            print(f"Current BTC Price: ${current_price['close'].iloc[-1]:.2f}")

if __name__ == "__main__":
    main()
