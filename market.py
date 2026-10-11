"""Market simulator with regime switching."""

import numpy as np
import random
from collections import deque

from config import REGIMES, MAX_PRICE_HISTORY


class Market:
    """Simulates stock price with regime-based dynamics."""
    
    __slots__ = ('price', 'prices', 'returns', 'tick', 'regime', 'regime_left', '_regime_keys')
    
    def __init__(self):
        self._regime_keys = list(REGIMES.keys())
        self.reset()
    
    def reset(self):
        self.price = 100.0
        self.prices = deque([100.0], maxlen=MAX_PRICE_HISTORY)
        self.returns = deque([0.0], maxlen=MAX_PRICE_HISTORY)
        self.tick = 0
        self.regime = random.choice(self._regime_keys)
        self.regime_left = random.randint(*REGIMES[self.regime]['dur'])
    
    def step(self) -> tuple[float, float]:
        """Advance one tick. Returns (price, return)."""
        self.tick += 1
        self.regime_left -= 1
        
        # Regime switching
        if self.regime_left <= 0:
            self.regime = random.choice(self._regime_keys)
            self.regime_left = random.randint(*REGIMES[self.regime]['dur'])
        
        r = REGIMES[self.regime]
        ret = r['drift'] + r['vol'] * np.random.randn()
        
        # Occasional jumps
        if random.random() < 0.025:
            ret += random.choice((-1.0, 1.0)) * random.uniform(0.01, 0.04)
        
        self.price = max(self.price * (1.0 + ret), 0.5)
        self.prices.append(self.price)
        self.returns.append(ret)
        
        return self.price, ret