import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

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

# Create third column: 1 * underlying_col1 + 2 * underlying_col2 (without noise)
col3 = 1 * underlying_col1 + 2 * underlying_col2

# Function to create time-lagged features
def create_time_lagged_features(data1, data2, target, window_size):
    X = []
    y = []
    for i in range(window_size, len(data1)):
        features = []
        for j in range(1, window_size + 1):
            features.append(data1[i - j])
            features.append(data2[i - j])
        X.append(features)
        y.append(target[i])
    return np.array(X), np.array(y)

# Test different window sizes
window_sizes = [1, 5, 10, 20, 30, 40, 50, 75, 100]

print("NEW RELATIONSHIP: target = 1*col1 + 2*col2")
print("=" * 70)
print(f"{'Window':>8} {'Features':>10} {'Train R²':>10} {'Test R²':>10} {'Gap':>8}")

for window_size in window_sizes:
    X, y = create_time_lagged_features(noisy_col1, noisy_col2, col3, window_size)
    
    if len(X) < 200:
        continue
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"{window_size:>8} {window_size * 2:>10} {r2_train:>10.4f} {r2_test:>10.4f} {r2_train - r2_test:>8.4f}")

# Analyze what the model learned with large window
print("\n" + "=" * 70)
print("Analysis of learned coefficients (window=100):")
window_size = 100
X, y = create_time_lagged_features(noisy_col1, noisy_col2, col3, window_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

col1_coeffs = model.coef_[0::2]  # Column1 coefficients across time
col2_coeffs = model.coef_[1::2]  # Column2 coefficients across time

print(f"Sum of col1 coefficients: {col1_coeffs.sum():.4f} (expected: 1.0)")
print(f"Sum of col2 coefficients: {col2_coeffs.sum():.4f} (expected: 2.0)")
print(f"First 5 col1 coefficients: {col1_coeffs[:5]}")
print(f"First 5 col2 coefficients: {col2_coeffs[:5]}")

# Check theoretical maximum
perfect_X, perfect_y = create_time_lagged_features(underlying_col1, underlying_col2, col3, 1)
perfect_X_train, perfect_X_test, perfect_y_train, perfect_y_test = train_test_split(
    perfect_X, perfect_y, test_size=0.3, random_state=42)

perfect_model = LinearRegression()
perfect_model.fit(perfect_X_train, perfect_y_train)
perfect_pred = perfect_model.predict(perfect_X_test)
perfect_r2 = r2_score(perfect_y_test, perfect_pred)

print(f"\nTheoretical maximum (noise-free): R² = {perfect_r2:.6f}")
print(f"Expected coefficients: [{1.0:.1f}, {2.0:.1f}]")
print(f"Actual coefficients learned from clean data: [{perfect_model.coef_[0]:.4f}, {perfect_model.coef_[1]:.4f}]")
