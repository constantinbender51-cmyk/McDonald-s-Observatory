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

# NEW RELATIONSHIP: 1 * col1 + 2 * col2
col3 = 1 * underlying_col1 + 2 * underlying_col2

print(f"Target statistics with new relationship:")
print(f"Target mean: {col3.mean():.2f}, std: {col3.std():.2f}")
print(f"Target range: {col3.min():.2f} to {col3.max():.2f}")

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
window_sizes = [1, 5, 10, 20, 50, 100]

print("\nNEW RELATIONSHIP: target = 1*col1 + 2*col2")
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

# Let's verify the relationship is actually different
print(f"\nVerification:")
print(f"Sample underlying values: col1={underlying_col1[500]:.1f}, col2={underlying_col2[500]:.1f}")
print(f"Old relationship (2/3,1/3): {(2/3)*underlying_col1[500] + (1/3)*underlying_col2[500]:.1f}")
print(f"New relationship (1,2): {1*underlying_col1[500] + 2*underlying_col2[500]:.1f}")
