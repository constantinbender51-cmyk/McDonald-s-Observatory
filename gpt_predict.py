"""
file: crypto_direction_prediction.py
Purpose: Predict next-day Bitcoin price direction using 1-minute OHLCV data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def load_data(path: str) -> pd.DataFrame:
    """Load 1-minute OHLCV data."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    return df


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 1-min OHLCV to daily candles."""
    daily = pd.DataFrame()
    daily["open"] = df["open"].resample("1D").first()
    daily["high"] = df["high"].resample("1D").max()
    daily["low"] = df["low"].resample("1D").min()
    daily("close") = df["close"].resample("1D").last()
    daily["volume"] = df["volume"].resample("1D").sum()
    daily = daily.dropna()
    return daily


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators. Keep behavior simple for training stability."""
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(7).std()

    df["ema_10"] = df["close"].ewm(span=10).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()

    df["rsi"] = compute_rsi(df["close"], 14)

    return df.dropna()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target: 1 if next-day close > today close, else 0."""
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna()


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> None:
    """Train multiple models with time-series validation and print accuracy."""
    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(n_estimators=300),
        "GradientBoosting": GradientBoostingClassifier()
    }

    for name, model in models.items():
        acc_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)                       # why: training step
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            acc_scores.append(acc)

        print(f"{name} accuracy: {np.mean(acc_scores):.4f}")


def main(path: str) -> None:
    df = load_data(path)
    daily = resample_to_daily(df)
    daily = add_features(daily)
    daily = build_target(daily)

    feature_cols = [c for c in daily.columns if c not in ("target")]
    X = daily[feature_cols]
    y = daily["target"]

    train_and_evaluate(X, y)


if __name__ == "__main__":
    main("your_1min_btc_data.csv")
