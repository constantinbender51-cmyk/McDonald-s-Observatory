#!/usr/bin/env python3
"""
backtest_lr.py
Backtests the linear regression trading strategy with 70/30 train/test split
Starting from January 1, 2018, using daily close prices for stop-loss checks
"""

import logging
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Assuming binance_ohlc module exists with get_ohlc function
try:
    import binance_ohlc
except ImportError:
    print("ERROR: binance_ohlc module not found. Please ensure it's available.")
    sys.exit(1)

# Configuration
SYMBOL = "BTCUSDT"
INTERVAL = "1d"
START_DATE = "2018-01-01"
TRAIN_RATIO = 0.70
LOOKBACK = 10
LEVERAGE = 3.0
STOP_FACTOR = 0.80
TRANSACTION_COST = 0.0007  # 0.07% per trade (fees + slippage)
INITIAL_CAPITAL = 10000  # USD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("backtest")


def stoch_rsi(close: np.ndarray, rsi_period: int = 14, stoch_period: int = 14) -> np.ndarray:
    """Calculate Stochastic RSI indicator."""
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_gain = pd.Series(gain).rolling(rsi_period).mean()
    roll_loss = pd.Series(loss).rolling(rsi_period).mean()
    rs = roll_gain / roll_loss
    rsi = 100 - (100 / (1 + rs))
    stoch = (rsi - rsi.rolling(stoch_period).min()) / \
            (rsi.rolling(stoch_period).max() - rsi.rolling(stoch_period).min())
    return stoch.values


def ema(arr: np.ndarray, n: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    return pd.Series(arr).ewm(span=n, adjust=False).mean().values


class ModelBundle:
    """Linear regression model for price prediction."""
    
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.theta = None
        self.mu = None
        self.sigma = None

    def fit(self, df: pd.DataFrame):
        """Train on 80% of provided data, store mu/sigma/theta."""
        d = df.copy()
        d["y"] = (d["close"].shift(-self.horizon) / d["close"] - 1) * 100
        d = d.dropna(subset=["y"])
        
        split = int(len(d) * 0.8)
        feats = self._build_features(d)
        X = feats[:split]
        y = d["y"].values[:split]
        
        self.mu = X.mean(axis=0)
        self.sigma = np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
        Xz = (X - self.mu) / self.sigma
        Xb = np.c_[np.ones(Xz.shape[0]), Xz]
        self.theta = np.linalg.lstsq(Xb, y, rcond=None)[0]

    def predict_last(self, df: pd.DataFrame) -> float:
        """Return prediction for the last row."""
        feats = self._build_features(df).iloc[[-1]]
        Xz = (feats - self.mu) / self.sigma
        Xb = np.c_[np.ones(Xz.shape[0]), Xz]
        return float(Xb @ self.theta)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return feature matrix (no NaNs)."""
        close = df["close"].values
        stoch = stoch_rsi(close)
        pct = np.concatenate([[np.nan], np.diff(close) / close[:-1] * 100])
        vol_pct = df["volume"].pct_change().values * 100

        macd_line = ema(close, 12) - ema(close, 26)
        macd_sig = ema(macd_line, 9)
        macd_diff = macd_line - macd_sig

        df = df.assign(
            stoch_rsi=stoch,
            pct_chg=pct,
            vol_pct_chg=vol_pct,
            macd_signal=macd_diff,
        )

        for i in range(LOOKBACK):
            df[f"stoch_{i}"] = df["stoch_rsi"].shift(LOOKBACK - i)
            df[f"pct_{i}"] = df["pct_chg"].shift(LOOKBACK - i)
            df[f"vol_{i}"] = df["vol_pct_chg"].shift(LOOKBACK - i)
            df[f"macd_{i}"] = df["macd_signal"].shift(LOOKBACK - i)

        feature_cols = [f"{pre}_{i}" for pre in ["stoch", "pct", "vol", "macd"] for i in range(LOOKBACK)]
        return df[feature_cols].dropna()


class Trade:
    """Represents a single trade."""
    
    def __init__(self, entry_date, entry_price, side, size_btc, stop_price):
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.side = side  # 'long' or 'short'
        self.size_btc = size_btc
        self.stop_price = stop_price
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0.0
        self.pnl_pct = 0.0


class Backtest:
    """Backtesting engine."""
    
    def __init__(self, df: pd.DataFrame, train_split_idx: int):
        self.df = df
        self.train_split_idx = train_split_idx
        self.model6 = ModelBundle(6)
        self.model10 = ModelBundle(10)
        
        self.capital = INITIAL_CAPITAL
        self.equity_curve = []
        self.trades: List[Trade] = []
        self.current_position: Trade = None
        self.current_signal = "HOLD"
        
    def run(self):
        """Execute the backtest."""
        log.info("=" * 60)
        log.info("TRAINING MODELS")
        log.info("=" * 60)
        
        # Train on first 70% of data
        train_df = self.df.iloc[:self.train_split_idx].copy()
        log.info(f"Training period: {train_df.index[0]} to {train_df.index[-1]}")
        log.info(f"Training on {len(train_df)} days of data")
        
        self.model6.fit(train_df)
        self.model10.fit(train_df)
        log.info("Models trained successfully")
        
        log.info("=" * 60)
        log.info("STARTING BACKTEST")
        log.info("=" * 60)
        
        # Test on remaining 30%
        test_df = self.df.iloc[self.train_split_idx:].copy()
        log.info(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")
        log.info(f"Testing on {len(test_df)} days")
        log.info(f"Initial capital: ${self.capital:,.2f}")
        log.info("")
        
        # Walk through test period
        for i in range(len(test_df)):
            current_idx = self.train_split_idx + i
            current_date = self.df.index[current_idx]
            current_price = self.df.iloc[current_idx]["close"]
            
            # Check stop-loss first if we have a position
            if self.current_position:
                if self._check_stop_loss(current_date, current_price):
                    continue
            
            # Get predictions using all data up to current day
            window_df = self.df.iloc[:current_idx + 1].copy()
            pred6 = self.model6.predict_last(window_df)
            pred10 = self.model10.predict_last(window_df)
            
            # Determine signal
            if pred6 > 0 and pred10 > 0:
                new_signal = "BUY"
            elif pred6 < 0 and pred10 < 0:
                new_signal = "SELL"
            else:
                new_signal = "HOLD"
            
            # Execute trades on signal change
            if new_signal != self.current_signal:
                self._handle_signal_change(current_date, current_price, new_signal, abs(pred6))
            
            # Record equity
            equity = self._calculate_equity(current_price)
            self.equity_curve.append({
                'date': current_date,
                'equity': equity,
                'price': current_price,
                'signal': self.current_signal
            })
        
        # Close any remaining position
        if self.current_position:
            final_date = test_df.index[-1]
            final_price = test_df.iloc[-1]["close"]
            self._close_position(final_date, final_price, "END_OF_BACKTEST")
        
        self._print_results()
    
    def _check_stop_loss(self, date, price) -> bool:
        """Check if stop-loss is hit. Returns True if position was closed."""
        pos = self.current_position
        
        if pos.side == "long" and price <= pos.stop_price:
            log.info(f"{date.date()}  STOP-LOSS HIT (long) at ${price:,.2f} (stop: ${pos.stop_price:,.2f})")
            self._close_position(date, pos.stop_price, "STOP_LOSS")
            return True
        elif pos.side == "short" and price >= pos.stop_price:
            log.info(f"{date.date()}  STOP-LOSS HIT (short) at ${price:,.2f} (stop: ${pos.stop_price:,.2f})")
            self._close_position(date, pos.stop_price, "STOP_LOSS")
            return True
        
        return False
    
    def _handle_signal_change(self, date, price, new_signal, pred6_abs):
        """Handle signal changes and position management."""
        prev_signal = self.current_signal
        
        # Close existing position if moving to HOLD or opposite direction
        if self.current_position:
            log.info(f"{date.date()}  Signal change: {prev_signal} → {new_signal}")
            self._close_position(date, price, "SIGNAL_CHANGE")
        
        # Open new position if not HOLD
        if new_signal != "HOLD":
            side = "long" if new_signal == "BUY" else "short"
            self._open_position(date, price, side, pred6_abs)
        
        self.current_signal = new_signal
    
    def _open_position(self, date, price, side, pred6_abs):
        """Open a new position."""
        notional = self.capital * LEVERAGE
        size_btc = notional / price
        
        # Apply transaction costs
        cost = notional * TRANSACTION_COST
        self.capital -= cost
        
        # Calculate stop-loss
        allowed_move = STOP_FACTOR * pred6_abs / 100.0
        if side == "long":
            stop_price = price * (1 - allowed_move)
        else:
            stop_price = price * (1 + allowed_move)
        
        self.current_position = Trade(date, price, side, size_btc, stop_price)
        
        log.info(f"{date.date()}  OPEN {side.upper()} {size_btc:.4f} BTC @ ${price:,.2f}")
        log.info(f"           Notional: ${notional:,.2f} | Stop: ${stop_price:,.2f} | Cost: ${cost:.2f}")
    
    def _close_position(self, date, price, reason):
        """Close the current position."""
        pos = self.current_position
        
        # Calculate P&L
        if pos.side == "long":
            price_change = price - pos.entry_price
        else:  # short
            price_change = pos.entry_price - price
        
        pnl = (price_change / pos.entry_price) * (pos.size_btc * pos.entry_price) * LEVERAGE
        pnl_pct = (pnl / INITIAL_CAPITAL) * 100
        
        # Apply transaction costs
        notional = pos.size_btc * price
        cost = notional * TRANSACTION_COST
        pnl -= cost
        
        self.capital += pnl
        
        pos.exit_date = date
        pos.exit_price = price
        pos.exit_reason = reason
        pos.pnl = pnl
        pos.pnl_pct = pnl_pct
        
        self.trades.append(pos)
        
        log.info(f"{date.date()}  CLOSE {pos.side.upper()} @ ${price:,.2f} | Reason: {reason}")
        log.info(f"           P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%) | Capital: ${self.capital:,.2f}")
        log.info("")
        
        self.current_position = None
    
    def _calculate_equity(self, current_price) -> float:
        """Calculate current portfolio equity."""
        if not self.current_position:
            return self.capital
        
        pos = self.current_position
        if pos.side == "long":
            unrealized_pnl = (current_price - pos.entry_price) * pos.size_btc * LEVERAGE
        else:
            unrealized_pnl = (pos.entry_price - current_price) * pos.size_btc * LEVERAGE
        
        return self.capital + unrealized_pnl
    
    def _print_results(self):
        """Print backtest results and performance metrics."""
        log.info("=" * 60)
        log.info("BACKTEST RESULTS")
        log.info("=" * 60)
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        final_equity = equity_df.iloc[-1]["equity"]
        
        # Calculate returns
        total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        
        # Buy and hold comparison
        test_start_price = self.df.iloc[self.train_split_idx]["close"]
        test_end_price = self.df.iloc[-1]["close"]
        bh_return = (test_end_price - test_start_price) / test_start_price * 100
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Calculate drawdown
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"] * 100
        max_drawdown = equity_df["drawdown"].min()
        
        # Sharpe ratio (annualized)
        equity_df["returns"] = equity_df["equity"].pct_change()
        sharpe = equity_df["returns"].mean() / equity_df["returns"].std() * np.sqrt(365) if len(equity_df) > 1 else 0
        
        log.info(f"Initial Capital:        ${INITIAL_CAPITAL:,.2f}")
        log.info(f"Final Equity:           ${final_equity:,.2f}")
        log.info(f"Total Return:           {total_return:+.2f}%")
        log.info(f"Buy & Hold Return:      {bh_return:+.2f}%")
        log.info(f"Outperformance:         {total_return - bh_return:+.2f}%")
        log.info("")
        log.info(f"Total Trades:           {len(self.trades)}")
        log.info(f"Winning Trades:         {len(winning_trades)}")
        log.info(f"Losing Trades:          {len(losing_trades)}")
        log.info(f"Win Rate:               {win_rate:.1f}%")
        log.info(f"Average Win:            ${avg_win:,.2f}")
        log.info(f"Average Loss:           ${avg_loss:,.2f}")
        log.info(f"Profit Factor:          {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A")
        log.info("")
        log.info(f"Max Drawdown:           {max_drawdown:.2f}%")
        log.info(f"Sharpe Ratio:           {sharpe:.2f}")
        log.info("")
        
        # Visualize equity curve using logging
        log.info("=" * 60)
        log.info("EQUITY CURVE")
        log.info("=" * 60)
        self._log_equity_curve(equity_df)
        
        # Trade log
        log.info("=" * 60)
        log.info("TRADE LOG")
        log.info("=" * 60)
        for i, trade in enumerate(self.trades, 1):
            log.info(f"Trade #{i}: {trade.side.upper()}")
            log.info(f"  Entry:  {trade.entry_date.date()} @ ${trade.entry_price:,.2f}")
            log.info(f"  Exit:   {trade.exit_date.date()} @ ${trade.exit_price:,.2f}")
            log.info(f"  Reason: {trade.exit_reason}")
            log.info(f"  P&L:    ${trade.pnl:+,.2f} ({trade.pnl_pct:+.2f}%)")
            log.info("")
    
    def _log_equity_curve(self, equity_df):
        """Create ASCII visualization of equity curve in logs."""
        # Sample equity curve for visualization (max 50 points)
        sample_size = min(50, len(equity_df))
        step = len(equity_df) // sample_size
        sampled = equity_df.iloc[::step].copy()
        
        # Normalize for ASCII chart (20 rows)
        min_eq = sampled["equity"].min()
        max_eq = sampled["equity"].max()
        chart_height = 20
        
        if max_eq == min_eq:
            log.info("Equity remained constant")
            return
        
        # Create chart
        for row in range(chart_height, -1, -1):
            threshold = min_eq + (max_eq - min_eq) * row / chart_height
            line = ""
            for val in sampled["equity"]:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            
            # Add axis labels
            if row == chart_height:
                log.info(f"${max_eq:>10,.0f} |{line}|")
            elif row == 0:
                log.info(f"${min_eq:>10,.0f} |{line}|")
            elif row == chart_height // 2:
                mid = (max_eq + min_eq) / 2
                log.info(f"${mid:>10,.0f} |{line}|")
            else:
                log.info(f"{' '*12}|{line}|")
        
        # Date axis
        start_date = sampled.iloc[0]["date"].strftime("%Y-%m-%d")
        end_date = sampled.iloc[-1]["date"].strftime("%Y-%m-%d")
        log.info(f"{' '*12} {start_date}{' '*(len(line)-len(start_date)-len(end_date))}{end_date}")


def main():
    log.info("Loading data from Binance...")
    
    # Fetch ALL historical data using the training function
    df = binance_ohlc.get_ohlc_for_training(symbol=SYMBOL, interval=INTERVAL)
    
    # Set timestamp as index and filter from start date
    df = df.set_index('timestamp')
    df = df[df.index >= START_DATE]
    
    if len(df) == 0:
        log.error(f"No data available from {START_DATE}")
        sys.exit(1)
    
    log.info(f"Loaded {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")
    
    # Calculate split
    train_split_idx = int(len(df) * TRAIN_RATIO)
    
    # Run backtest
    bt = Backtest(df, train_split_idx)
    bt.run()


if __name__ == "__main__":
    main()
