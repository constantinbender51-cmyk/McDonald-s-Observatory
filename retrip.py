#!/usr/bin/env python3
"""
backtest_lr.py
Backtests the linear regression trading strategy with 70/30 train/test split
Starting from January 1, 2018, using daily close prices for stop-loss checks
Then starts a web server to display results
"""

import logging
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import time
import json
from flask import Flask, render_template_string
import threading

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

class DelayedLogger(logging.Logger):
    """Logger that adds delay after each log to prevent Railway log scrambling."""
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        super()._log(level, msg, args, exc_info, extra, stack_info)
        time.sleep(0.1)  # Delay after each log output

logging.setLoggerClass(DelayedLogger)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("backtest")

# Global variables for web server
backtest_results = None
equity_data = None
trade_data = None
performance_metrics = None

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
        result = Xb @ self.theta
        return float(result.item())  # Fixed: using .item() instead of float()

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
            current_row = self.df.iloc[current_idx]
            current_price = current_row["close"]
            current_high = current_row["high"]
            current_low = current_row["low"]
            
            # Check stop-loss first if we have a position
            if self.current_position:
                if self._check_stop_loss(current_date, current_high, current_low, current_price):
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
        
        self._prepare_web_data()
        self._print_results()
    
    def _check_stop_loss(self, date, high, low, close_price) -> bool:
        """
        Check if stop-loss is hit - ASSUMES EXACT EXECUTION AT STOP PRICE.
        Returns True if position was closed.
        """
        pos = self.current_position
        
        if pos.side == "long":
            # For long positions, check if price dropped to or below stop
            if close_price <= pos.stop_price:
                # EXACT execution at stop price
                exit_price = pos.stop_price
                log.info(f"{date.date()}  STOP-LOSS HIT (long) - Close: ${close_price:,.2f} | Stop: ${pos.stop_price:,.2f}")
                self._close_position(date, exit_price, "STOP_LOSS")
                return True
        
        elif pos.side == "short":
            # For short positions, check if price rose to or above stop
            if close_price >= pos.stop_price:
                # EXACT execution at stop price
                exit_price = pos.stop_price
                log.info(f"{date.date()}  STOP-LOSS HIT (short) - Close: ${close_price:,.2f} | Stop: ${pos.stop_price:,.2f}")
                self._close_position(date, exit_price, "STOP_LOSS")
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
    
    def _prepare_web_data(self):
        """Prepare data for web display."""
        global equity_data, trade_data, performance_metrics
        
        # Convert equity curve for JSON serialization
        equity_df = pd.DataFrame(self.equity_curve)
        equity_data = {
            'dates': [d.strftime('%Y-%m-%d') for d in equity_df['date']],
            'equity': [float(e) for e in equity_df['equity']],
            'price': [float(p) for p in equity_df['price']],
            'signals': list(equity_df['signal'])
        }
        
        # Prepare trade data
        trade_data = []
        for i, trade in enumerate(self.trades, 1):
            trade_data.append({
                'id': i,
                'side': trade.side.upper(),
                'entry_date': trade.entry_date.strftime('%Y-%m-%d'),
                'entry_price': float(trade.entry_price),
                'exit_date': trade.exit_date.strftime('%Y-%m-%d') if trade.exit_date else '',
                'exit_price': float(trade.exit_price) if trade.exit_price else 0,
                'exit_reason': trade.exit_reason,
                'pnl': float(trade.pnl),
                'pnl_pct': float(trade.pnl_pct)
            })
        
        # Calculate performance metrics
        final_equity = equity_df.iloc[-1]["equity"]
        total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        
        test_start_price = self.df.iloc[self.train_split_idx]["close"]
        test_end_price = self.df.iloc[-1]["close"]
        bh_return = (test_end_price - test_start_price) / test_start_price * 100
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"] * 100
        max_drawdown = equity_df["drawdown"].min()
        
        performance_metrics = {
            'initial_capital': float(INITIAL_CAPITAL),
            'final_equity': float(final_equity),
            'total_return': float(total_return),
            'bh_return': float(bh_return),
            'outperformance': float(total_return - bh_return),
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': float(win_rate),
            'max_drawdown': float(max_drawdown)
        }
    
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
        
        log.info("Starting web server on http://0.0.0.0:8000")


def create_app():
    """Create and configure Flask app."""
    app = Flask(__name__)
    
    # HTML template with Chart.js
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Results - Linear Regression Strategy</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .metric-card { background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #007bff; }
            .metric-value { font-size: 24px; font-weight: bold; color: #333; }
            .metric-label { font-size: 14px; color: #666; margin-top: 5px; }
            .positive { color: #28a745; }
            .negative { color: #dc3545; }
            .chart-container { margin: 30px 0; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f8f9fa; }
            .trade-positive { background-color: #d4edda; }
            .trade-negative { background-color: #f8d7da; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Linear Regression Trading Strategy Backtest</h1>
            <p>Symbol: {{ symbol }} | Period: {{ start_date }} to {{ end_date }} | Initial Capital: ${{ "%.2f"|format(metrics.initial_capital) }}</p>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value {{ 'positive' if metrics.final_equity > metrics.initial_capital else 'negative' }}">
                        ${{ "%.2f"|format(metrics.final_equity) }}
                    </div>
                    <div class="metric-label">Final Equity</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {{ 'positive' if metrics.total_return > 0 else 'negative' }}">
                        {{ "%.2f"|format(metrics.total_return) }}%
                    </div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {{ 'positive' if metrics.bh_return > 0 else 'negative' }}">
                        {{ "%.2f"|format(metrics.bh_return) }}%
                    </div>
                    <div class="metric-label">Buy & Hold</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {{ 'positive' if metrics.outperformance > 0 else 'negative' }}">
                        {{ "%.2f"|format(metrics.outperformance) }}%
                    </div>
                    <div class="metric-label">Outperformance</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ metrics.total_trades }}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {{ 'positive' if metrics.win_rate > 50 else 'negative' }}">
                        {{ "%.1f"|format(metrics.win_rate) }}%
                    </div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value negative">
                        {{ "%.2f"|format(metrics.max_drawdown) }}%
                    </div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
            </div>

            <div class="chart-container">
                <h2>Equity Curve</h2>
                <canvas id="equityChart" height="100"></canvas>
            </div>

            <div class="chart-container">
                <h2>Price vs Equity</h2>
                <canvas id="priceChart" height="100"></canvas>
            </div>

            <h2>Trade History</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Side</th>
                        <th>Entry Date</th>
                        <th>Entry Price</th>
                        <th>Exit Date</th>
                        <th>Exit Price</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in trades %}
                    <tr class="{{ 'trade-positive' if trade.pnl > 0 else 'trade-negative' }}">
                        <td>{{ trade.id }}</td>
                        <td>{{ trade.side }}</td>
                        <td>{{ trade.entry_date }}</td>
                        <td>${{ "%.2f"|format(trade.entry_price) }}</td>
                        <td>{{ trade.exit_date }}</td>
                        <td>${{ "%.2f"|format(trade.exit_price) }}</td>
                        <td class="{{ 'positive' if trade.pnl > 0 else 'negative' }}">${{ "%.2f"|format(trade.pnl) }}</td>
                        <td class="{{ 'positive' if trade.pnl_pct > 0 else 'negative' }}">{{ "%.2f"|format(trade.pnl_pct) }}%</td>
                        <td>{{ trade.exit_reason }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <script>
            // Equity Curve Chart
            const equityCtx = document.getElementById('equityChart').getContext('2d');
            const equityChart = new Chart(equityCtx, {
                type: 'line',
                data: {
                    labels: {{ equity_data.dates | tojson }},
                    datasets: [{
                        label: 'Portfolio Equity',
                        data: {{ equity_data.equity | tojson }},
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        fill: true,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            title: { display: true, text: 'Date' }
                        },
                        y: {
                            title: { display: true, text: 'Equity (USD)' },
                            beginAtZero: false
                        }
                    }
                }
            });

            // Price vs Equity Chart
            const priceCtx = document.getElementById('priceChart').getContext('2d');
            const priceChart = new Chart(priceCtx, {
                type: 'line',
                data: {
                    labels: {{ equity_data.dates | tojson }},
                    datasets: [
                        {
                            label: 'Portfolio Equity',
                            data: {{ equity_data.equity | tojson }},
                            borderColor: '#007bff',
                            backgroundColor: 'rgba(0, 123, 255, 0.1)',
                            fill: true,
                            yAxisID: 'y'
                        },
                        {
                            label: 'BTC Price',
                            data: {{ equity_data.price | tojson }},
                            borderColor: '#28a745',
                            backgroundColor: 'rgba(40, 167, 69, 0.1)',
                            fill: true,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            title: { display: true, text: 'Date' }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: { display: true, text: 'Equity (USD)' }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: { display: true, text: 'BTC Price (USD)' },
                            grid: { drawOnChartArea: false }
                        }
                    }
                }
            });
        </script>
    </body>
    </html>
    """
    
    @app.route('/')
    def index():
        if equity_data is None:
            return "Backtest not completed yet. Please wait..."
        
        return render_template_string(html_template,
            symbol=SYMBOL,
            start_date=START_DATE,
            end_date=datetime.now().strftime('%Y-%m-%d'),
            equity_data=equity_data,
            trades=trade_data,
            metrics=performance_metrics
        )
    
    return app


def start_web_server():
    """Start the Flask web server."""
    app = create_app()
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)


def main():
    import os
    
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
    
    # Start web server after backtest completes
    log.info("Backtest completed. Starting web server...")
    start_web_server()


if __name__ == "__main__":
    main()
