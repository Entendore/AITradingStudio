"""Base agent class and common indicator utilities."""

import numpy as np
from abc import ABC, abstractmethod

from config import INITIAL_CASH, TRADE_SIZE, COMMISSION, MAX_POSITION, MAX_TRADE_MARKERS


class Agent(ABC):
    """Abstract base class for all trading agents."""
    
    __slots__ = ('name', 'color', 'cash', 'shares', 'equity_curve', 
                 'total_trades', 'trade_markers')
    
    def __init__(self, name: str, color):
        self.name = name
        self.color = color
        self.reset_portfolio()
    
    def reset_portfolio(self):
        self.cash = INITIAL_CASH
        self.shares = 0
        self.equity_curve = [INITIAL_CASH]
        self.total_trades = 0
        self.trade_markers = []
    
    def equity(self, price: float) -> float:
        return self.cash + self.shares * price
    
    @abstractmethod
    def decide(self, prices, returns, tick: int) -> str:
        """Return 'buy', 'sell', or 'hold'."""
        ...
    
    def execute(self, action: str, price: float) -> bool:
        """Execute a trade. Returns True if trade was made."""
        if self.equity(price) < 500:
            return False
        
        if action == 'buy' and self.shares < MAX_POSITION:
            cost = TRADE_SIZE * price * (1.0 + COMMISSION)
            if self.cash >= cost:
                self.cash -= cost
                self.shares += TRADE_SIZE
                self.total_trades += 1
                self.trade_markers.append((len(self.equity_curve), 0, price))
                if len(self.trade_markers) > MAX_TRADE_MARKERS:
                    self.trade_markers = self.trade_markers[-MAX_TRADE_MARKERS:]
                return True
        
        elif action == 'sell' and self.shares > -MAX_POSITION:
            rev = TRADE_SIZE * price * (1.0 - COMMISSION)
            self.cash += rev
            self.shares -= TRADE_SIZE
            self.total_trades += 1
            self.trade_markers.append((len(self.equity_curve), 1, price))
            if len(self.trade_markers) > MAX_TRADE_MARKERS:
                self.trade_markers = self.trade_markers[-MAX_TRADE_MARKERS:]
            return True
        
        return False
    
    def record(self, price: float):
        self.equity_curve.append(self.equity(price))
    
    def end_generation(self):
        """Hook called at end of each generation."""
        pass


# ═══════════════════════════════════════════════════════════════
# INDICATOR UTILITIES
# ═══════════════════════════════════════════════════════════════

def extract_features(prices, returns, shares: int) -> np.ndarray:
    """Extract 12 features from price data for ML agents."""
    buf = np.zeros(12, dtype=np.float64)
    n = len(returns)
    
    for i in range(8):
        buf[i] = returns[-(i + 1)] if n > i else 0.0
    
    if n >= 20:
        window = returns[-20:]
        buf[8] = np.mean(window)
        buf[9] = np.std(window)
    else:
        buf[8] = 0.0
        buf[9] = 0.005
    
    if len(prices) >= 20:
        ma = np.mean(prices[-20:])
        buf[10] = (prices[-1] - ma) / max(ma, 0.01)
    else:
        buf[10] = 0.0
    
    buf[11] = np.clip(shares / 60.0, -1.0, 1.0)
    
    return buf


def calc_ema(data, period: int):
    """Calculate EMA. Returns None if insufficient data."""
    if len(data) < period:
        return None
    ema = sum(data[:period]) / period
    k = 2.0 / (period + 1)
    for v in data[period:]:
        ema = v * k + ema * (1.0 - k)
    return ema


def calc_rsi(returns, period: int = 14) -> float:
    """Calculate RSI from returns."""
    if len(returns) < period:
        return 50.0
    # FIX: Convert list slice to numpy array before arithmetic
    window = np.array(returns[-period:], dtype=np.float64)
    gains = np.maximum(window, 0.0)
    losses = np.maximum(-window, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss < 1e-10:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def calc_bollinger(prices, period: int = 20, k: float = 2.0):
    """Calculate Bollinger Bands. Returns (lower, mid, upper) or (None, None, None)."""
    if len(prices) < period:
        return None, None, None
    # FIX: Convert list slice to numpy array for consistency
    window = np.array(prices[-period:], dtype=np.float64)
    ma = np.mean(window)
    std = np.std(window)
    return ma - k * std, ma, ma + k * std


def calc_macd_signal(prices, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD and signal line."""
    e_fast = calc_ema(prices, fast)
    e_slow = calc_ema(prices, slow)
    if e_fast is None or e_slow is None:
        return None, None
    macd = e_fast - e_slow
    return macd, calc_ema([macd], signal)