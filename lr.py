import numpy as np

# ---------- helpers ----------
def sigmoid(z):
    z = np.clip(z, -500, 500)          # avoid overflow
    return 1 / (1 + np.exp(-z))

def add_bias(X):
    return np.c_[np.ones(X.shape[0]), X]

def cross_entropy(y, p_hat):
    # y ∈ {0,1}, p_hat = predicted probability
    p_hat = np.clip(p_hat, 1e-7, 1-1e-7)
    return -np.mean(y*np.log(p_hat) + (1-y)*np.log(1-p_hat))

# ---------- core ----------
class LogisticRegression:
    def __init__(self, lr=0.1, n_iter=10_000, tol=1e-6):
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.losses = []

    def fit(self, X, y):
        X = add_bias(X)
        self.theta = np.zeros(X.shape[1])
        for i in range(self.n_iter):
            z = X @ self.theta
            p_hat = sigmoid(z)
            grad = (X.T @ (p_hat - y)) / y.size
            self.theta -= self.lr * grad
            self.losses.append(cross_entropy(y, p_hat))
            if i > 0 and abs(self.losses[-2] - self.losses[-1]) < self.tol:
                break
        return self

    def predict_proba(self, X):
        return sigmoid(add_bias(X) @ self.theta)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
