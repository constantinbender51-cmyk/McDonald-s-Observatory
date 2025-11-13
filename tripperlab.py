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

# Create third column: 2/3 * underlying_col1 + 1/3 * underlying_col2 (without noise)
col3 = (2/3) * underlying_col1 + (1/3) * underlying_col2

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

# Test much larger window sizes
window_sizes = [1, 5, 10, 20, 30, 40, 50, 75, 100]

print("Testing larger window sizes:")
print("=" * 70)
print(f"{'Window':>8} {'Features':>10} {'Train R²':>10} {'Test R²':>10} {'Gap':>8}")

for window_size in window_sizes:
    X, y = create_time_lagged_features(noisy_col1, noisy_col2, col3, window_size)
    
    # We need enough samples for train/test split with large windows
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

# Let's also check what happens at the theoretical limit
print("\n" + "=" * 70)
print("Theoretical maximum performance:")
# If we could perfectly denoise, what's the best possible R²?
perfect_features = np.column_stack((underlying_col1, underlying_col2))
perfect_X, perfect_y = create_time_lagged_features(underlying_col1, underlying_col2, col3, 1)
perfect_X_train, perfect_X_test, perfect_y_train, perfect_y_test = train_test_split(
    perfect_X, perfect_y, test_size=0.3, random_state=42)

perfect_model = LinearRegression()
perfect_model.fit(perfect_X_train, perfect_y_train)
perfect_pred = perfect_model.predict(perfect_X_test)
perfect_r2 = r2_score(perfect_y_test, perfect_pred)

print(f"With perfect noise-free data: R² = {perfect_r2:.6f}")
print(f"Theoretical coefficients should be: [{2/3:.4f}, {1/3:.4f}]")
print(f"Actual coefficients learned: [{perfect_model.coef_[0]:.4f}, {perfect_model.coef_[1]:.4f}]")
