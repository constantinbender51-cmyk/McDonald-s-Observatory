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
# Using noise with standard deviation of 500 to make it very hard to predict
noise_std = 500
noisy_col1 = underlying_col1 + np.random.normal(0, noise_std, n_samples)
noisy_col2 = underlying_col2 + np.random.normal(0, noise_std, n_samples)

# Create third column: 2/3 * underlying_col1 + 1/3 * underlying_col2 (without noise)
col3 = (2/3) * underlying_col1 + (1/3) * underlying_col2

# Create the feature matrix (noisy columns 1 and 2) and target (clean column 3)
X = np.column_stack((noisy_col1, noisy_col2))
y = col3

# Split the data (using all for training in this case, but could split for validation)
# For simplicity, we'll use the entire dataset
X_train, y_train = X, y

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_train)

# Calculate and print R-squared score
r2 = r2_score(y_train, y_pred)
print(f"R-squared score: {r2:.4f}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.4f}")

# The model should theoretically learn coefficients close to [0.6667, 0.3333]
# but the extreme noise will make this very difficult
