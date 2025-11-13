#!/usr/bin/env python3
"""
Binance-weekly-direction-logreg.py
Fetch daily OHLCV from Binance since 2018-01-01, engineer 28-day look-back
features (price & volume %-changes, MACD, Signal, Stoch-RSI), train a
logistic-regression classifier to predict the *sign* of the 7-day-forward
price change, evaluate on a 20 % hold-out test set (time-series split).

Requires:  python-binance, pandas, numpy, ta, scikit-learn
Install:   pip install python-binance pandas numpy ta scikit-learn
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from binance.client import Client
from ta.momentum import StochRSIIndicator
from ta.trend import MACD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --------------------------------------------------
# 1. Parameters
# --------------------------------------------------
START_DATE = "1 Jan 2018"
SYMBOL     = "BTCUSDT"
LOOKBACK   = 28          # days of history for features
HORIZON    = 7           # days ahead we want to predict
TEST_SIZE  = 0.20        # 20 % test split
RANDOM_STATE = 42

# --------------------------------------------------
# 2. Helper: download daily klines
# --------------------------------------------------
def fetch_daily_klines(client, symbol, start_str):
    """Return a DataFrame with daily OHLCV from Binance."""
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_1DAY,
        start_str=start_str,
        klines_type=client.HISTORICAL_KLINES_TYPE_SPOT
    )
    df = pd.DataFrame(
        klines,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_av", "trades", "taker_base_vol",
            "taker_quote_vol", "ignore"
        ]
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)
    return df

# --------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------
def add_technical_indicators(df):
    """Append MACD and Stoch-RSI columns."""
    close = df["close"]
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]      = macd.macd()
    df["macd_sig"]  = macd.macd_signal()

    stoch_rsi = StochRSIIndicator(close=close, window=14)
    df["stoch_rsi"] = stoch_rsi.stochrsi()
    return df

def build_features(df):
    """Create 28-day look-back feature matrix."""
    df = add_technical_indicators(df)

    # %-changes
    df["ret"]  = df["close"].pct_change()
    df["vol_ch"] = df["volume"].pct_change()

    # Target: sign of 7-day forward return
    df["fwd_ret"]  = df["close"].shift(-HORIZON) / df["close"] - 1
    df["target"]   = np.where(df["fwd_ret"] > 0, 1, 0)

    # Collect lags
    feat_cols = []
    for lag in range(1, LOOKBACK):
        df[f"ret_lag{lag}"] = df["ret"].shift(lag)
        feat_cols.append(f"ret_lag{lag}")

    for lag in range(1, LOOKBACK):
        df[f"vol_ch_lag{lag}"] = df["vol_ch"].shift(lag)
        feat_cols.append(f"vol_ch_lag{lag}")

    for lag in range(1, LOOKBACK):
        df[f"macd_lag{lag}"] = df["macd"].shift(lag)
        feat_cols.append(f"macd_lag{lag}")

    for lag in range(1, LOOKBACK):
        df[f"macd_sig_lag{lag}"] = df["macd_sig"].shift(lag)
        feat_cols.append(f"macd_sig_lag{lag}")

    for lag in range(1, LOOKBACK):
        df[f"stoch_rsi_lag{lag}"] = df["stoch_rsi"].shift(lag)
        feat_cols.append(f"stoch_rsi_lag{lag}")

    # Drop rows with NaN
    df.dropna(inplace=True)
    return df, feat_cols

# --------------------------------------------------
# 4. Main
# --------------------------------------------------
def main():
    # Read API key/secret from env vars (optional)
    api_key    = os.getenv("BINANCE_KEY")
    api_secret = os.getenv("BINANCE_SECRET")
    client = Client(api_key=api_key, api_secret=api_secret)

    print("Downloading daily klines …")
    df = fetch_daily_klines(client, SYMBOL, START_DATE)
    print(f"Raw data rows: {len(df)}")

    print("Building features …")
    df, feat_cols = build_features(df)
    print(f"Usable rows after feature engineering: {len(df)}")

    # Train/test split (time-series)
    split_idx = int(len(df) * (1 - TEST_SIZE))
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    X_train = train_df[feat_cols]
    y_train = train_df["target"]
    X_test  = test_df[feat_cols]
    y_test  = test_df["target"]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Model pipeline: standardize + logistic regression
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    )

    print("Training logistic regression …")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)

    print(f"Test-set accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
