import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score

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

# Evaluate with proper train/test split
window_sizes = [1, 3, 5, 10, 20]

print("Proper evaluation with train/test split:")
print("=" * 60)

for window_size in window_sizes:
    X, y = create_time_lagged_features(noisy_col1, noisy_col2, col3, window_size)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Test on unseen data
    y_pred = model.predict(X_test)
    r2_test = r2_score(y_test, y_pred)
    
    # Also check training performance for comparison
    y_pred_train = model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    print(f"Window: {window_size:2d} | Features: {window_size * 2:2d} | "
          f"Train R²: {r2_train:.4f} | Test R²: {r2_test:.4f} | "
          f"Gap: {r2_train - r2_test:+.4f}")

# Cross-validation for more robust evaluation
print("\n" + "=" * 60)
print("Cross-validation results (more reliable):")
for window_size in [1, 5, 10]:
    X, y = create_time_lagged_features(noisy_col1, noisy_col2, col3, window_size)
    model = LinearRegression()
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Window: {window_size:2d} | CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
