import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate the dataset
n_samples = 1000

# Create underlying patterns (without noise)
underlying_col1 = np.linspace(0, 999, n_samples)  # 0 to 999
underlying_col2 = np.linspace(999, 0, n_samples)  # 999 to 0

# Add extreme noise to first two columns
noise_std = 500
noisy_col1 = underlying_col1 + np.random.normal(0, noise_std, n_samples)
noisy_col2 = underlying_col2 + np.random.normal(0, noise_std, n_samples)

# Create third column: 2/3 * underlying_col1 + 1/3 * underlying_col2 (without noise)
col3 = (2/3) * underlying_col1 + (1/3) * underlying_col2

# Function to create time-lagged features
def create_time_lagged_features(data1, data2, target, window_size):
    X = []
    y = []
    for i in range(window_size, len(data1)):
        # Features: previous 'window_size' values from both columns
        features = []
        for j in range(1, window_size + 1):
            features.append(data1[i - j])
            features.append(data2[i - j])
        X.append(features)
        y.append(target[i])
    return np.array(X), np.array(y)

# Create features with different window sizes and evaluate
window_sizes = [1, 3, 5, 10, 20]

print("Prediction accuracy with different numbers of time-lagged features:")
print("=" * 60)

for window_size in window_sizes:
    # Create time-lagged features
    X, y = create_time_lagged_features(noisy_col1, noisy_col2, col3, window_size)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate R-squared score
    r2 = r2_score(y, y_pred)
    
    print(f"Window size: {window_size:2d} | Features: {window_size * 2:2d} | R-squared: {r2:.4f}")

# Show the theoretical coefficients we expect
print("\n" + "=" * 60)
print("Theoretical relationship: 2/3 * col1 + 1/3 * col2")
print("With more time steps, model should better approximate these weights")
