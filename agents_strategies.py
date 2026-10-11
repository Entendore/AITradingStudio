"""Fixed-rule trading strategy agents."""

import numpy as np
import random

from agents_base import Agent, calc_ema, calc_rsi, calc_bollinger
from config import WARMUP


class MomentumAgent(Agent):
    __slots__ = ()
    def __init__(self, color):
        super().__init__('Momentum', color)
    def decide(self, p, r, t):
        if t < WARMUP or len(r) < 10:
            return 'hold'
        m = sum(r[-10:])
        return 'buy' if m > 0.02 else ('sell' if m < -0.02 else 'hold')


class MeanReversionAgent(Agent):
    __slots__ = ()
    def __init__(self, color):
        super().__init__('Mean Revert', color)
    def decide(self, p, r, t):
        if t < WARMUP or len(p) < 30:
            return 'hold'
        ma = np.mean(p[-30:])
        d = (p[-1] - ma) / max(ma, 0.01)
        return 'buy' if d < -0.025 else ('sell' if d > 0.025 else 'hold')


class MACrossoverAgent(Agent):
    __slots__ = ()
    def __init__(self, color):
        super().__init__('MA Cross', color)
    def decide(self, p, r, t):
        if t < WARMUP or len(p) < 22:
            return 'hold'
        sma, lma = np.mean(p[-5:]), np.mean(p[-20:])
        ps, pl = np.mean(p[-6:-1]), np.mean(p[-21:-1])
        if ps <= pl and sma > lma:
            return 'buy'
        if ps >= pl and sma < lma:
            return 'sell'
        return 'hold'


class BollingerAgent(Agent):
    __slots__ = ()
    def __init__(self, color):
        super().__init__('Bollinger', color)
    def decide(self, p, r, t):
        if t < WARMUP:
            return 'hold'
        low, ma, up = calc_bollinger(p, 20, 2.0)
        if low is None:
            return 'hold'
        rng = up - low
        if rng < 0.01:
            return 'hold'
        pos = (p[-1] - low) / rng
        return 'buy' if pos < 0.1 else ('sell' if pos > 0.9 else 'hold')


class RSIAgent(Agent):
    __slots__ = ()
    def __init__(self, color):
        super().__init__('RSI', color)
    def decide(self, p, r, t):
        if t < WARMUP:
            return 'hold'
        rsi = calc_rsi(r, 14)
        return 'buy' if rsi < 30 else ('sell' if rsi > 70 else 'hold')


class BreakoutAgent(Agent):
    __slots__ = ()
    def __init__(self, color):
        super().__init__('Breakout', color)
    def decide(self, p, r, t):
        if t < WARMUP or len(p) < 30:
            return 'hold'
        w = p[-30:]
        hi, lo = max(w), min(w)
        rng = hi - lo
        if rng < 0.01:
            return 'hold'
        if p[-1] > hi - 0.05 * rng:
            return 'buy'
        if p[-1] < lo + 0.05 * rng:
            return 'sell'
        return 'hold'


class MACDAgent(Agent):
    __slots__ = ('_mh', '_pm', '_ps')
    def __init__(self, color):
        super().__init__('MACD', color)
        self._mh = []
        self._pm = None
        self._ps = None
    def reset_portfolio(self):
        super().reset_portfolio()
        self._mh = []
        self._pm = None
        self._ps = None
    def decide(self, p, r, t):
        if len(p) < 35:
            return 'hold'
        e12, e26 = calc_ema(p, 12), calc_ema(p, 26)
        if e12 is None or e26 is None:
            return 'hold'
        macd = e12 - e26
        self._mh.append(macd)
        if len(self._mh) > 50:
            self._mh = self._mh[-50:]
        sig = calc_ema(self._mh, 9)
        if sig is None:
            return 'hold'
        if self._pm is not None and self._ps is not None:
            if self._pm <= self._ps and macd > sig:
                return 'buy'
            if self._pm >= self._ps and macd < sig:
                return 'sell'
        self._pm, self._ps = macd, sig
        return 'hold'


class IchimokuAgent(Agent):
    __slots__ = ('_pt', '_pk')
    def __init__(self, color):
        super().__init__('Ichimoku', color)
        self._pt = None
        self._pk = None
    def reset_portfolio(self):
        super().reset_portfolio()
        self._pt = None
        self._pk = None
    def decide(self, p, r, t):
        if len(p) < 52:
            return 'hold'
        tk = (max(p[-9:]) + min(p[-9:])) / 2
        kj = (max(p[-26:]) + min(p[-26:])) / 2
        sa = (tk + kj) / 2
        sb = (max(p[-52:]) + min(p[-52:])) / 2
        pr = p[-1]
        if self._pt is not None:
            if self._pt <= self._pk and tk > kj and pr > sa and pr > sb:
                return 'buy'
            if self._pt >= self._pk and tk < kj and pr < sa and pr < sb:
                return 'sell'
        self._pt, self._pk = tk, kj
        return 'hold'


class TurtleAgent(Agent):
    __slots__ = ('_st',)
    def __init__(self, color):
        super().__init__('Turtle', color)
        self._st = 0
    def reset_portfolio(self):
        super().reset_portfolio()
        self._st = 0
    def decide(self, p, r, t):
        if len(p) < 20:
            return 'hold'
        h20, l20 = max(p[-20:]), min(p[-20:])
        h10, l10 = max(p[-10:]), min(p[-10:])
        pr = p[-1]
        if self._st == 0:
            if pr >= h20:
                self._st = 1
                return 'buy'
            if pr <= l20:
                self._st = -1
                return 'sell'
        elif self._st == 1:
            if pr <= l10:
                self._st = 0
                return 'sell'
        elif self._st == -1:
            if pr >= h10:
                self._st = 0
                return 'buy'
        return 'hold'


class ZScoreAgent(Agent):
    __slots__ = ()
    def __init__(self, color):
        super().__init__('Z-Score', color)
    def decide(self, p, r, t):
        if len(r) < 20:
            return 'hold'
        w = r[-20:]
        mu, sigma = np.mean(w), np.std(w)
        if sigma < 1e-6:
            return 'hold'
        z = (r[-1] - mu) / sigma
        if z < -1.8:
            return 'buy'
        if z > 1.8:
            return 'sell'
        if z < -1.0:
            return 'buy'
        if z > 1.0:
            return 'sell'
        return 'hold'


class ParabolicSARAgent(Agent):
    __slots__ = ('_is_long', '_ep', '_af', '_sar', '_init')
    def __init__(self, color):
        super().__init__('Parabolic SAR', color)
        self._is_long = True
        self._ep = 100.0
        self._af = 0.02
        self._sar = 100.0
        self._init = False
    def reset_portfolio(self):
        super().reset_portfolio()
        self._is_long = True
        self._ep = 100.0
        self._af = 0.02
        self._sar = 100.0
        self._init = False
    def decide(self, p, r, t):
        if len(p) < 5:
            return 'hold'
        pr = p[-1]
        if not self._init:
            self._is_long = pr >= p[-2]
            self._ep = pr
            self._sar = min(p[-5:]) if self._is_long else max(p[-5:])
            self._init = True
            return 'hold'
        self._sar = self._sar + self._af * (self._ep - self._sar)
        if self._is_long:
            if pr > self._ep:
                self._ep = pr
                self._af = min(self._af + 0.02, 0.20)
            if pr < self._sar:
                self._is_long = False
                self._sar = self._ep
                self._ep = pr
                self._af = 0.02
                return 'sell'
        else:
            if pr < self._ep:
                self._ep = pr
                self._af = min(self._af + 0.02, 0.20)
            if pr > self._sar:
                self._is_long = True
                self._sar = self._ep
                self._ep = pr
                self._af = 0.02
                return 'buy'
        return 'hold'


class StochasticAgent(Agent):
    __slots__ = ('_k_hist', '_pk', '_pd')
    def __init__(self, color):
        super().__init__('Stochastic', color)
        self._k_hist = []
        self._pk = None
        self._pd = None
    def reset_portfolio(self):
        super().reset_portfolio()
        self._k_hist = []
        self._pk = None
        self._pd = None
    def decide(self, p, r, t):
        if len(p) < 15:
            return 'hold'
        hi = max(p[-14:])
        lo = min(p[-14:])
        rng = hi - lo
        k = ((p[-1] - lo) / rng * 100) if rng > 1e-6 else 50.0
        self._k_hist.append(k)
        if len(self._k_hist) > 20:
            self._k_hist = self._k_hist[-20:]
        d = np.mean(self._k_hist[-3:]) if len(self._k_hist) >= 3 else k
        if self._pk is not None and self._pd is not None:
            if self._pk <= self._pd and k > d and k < 30:
                return 'buy'
            if self._pk >= self._pd and k < d and k > 70:
                return 'sell'
        self._pk, self._pd = k, d
        return 'buy' if k < 20 else ('sell' if k > 80 else 'hold')


class TripleEMAAgent(Agent):
    __slots__ = ()
    def __init__(self, color):
        super().__init__('Triple EMA', color)
    def decide(self, p, r, t):
        if len(p) < 35:
            return 'hold'
        e5, e13, e34 = calc_ema(p, 5), calc_ema(p, 13), calc_ema(p, 34)
        if e5 is None or e13 is None or e34 is None:
            return 'hold'
        return 'buy' if e5 > e13 > e34 else ('sell' if e5 < e13 < e34 else 'hold')


class FibonacciAgent(Agent):
    __slots__ = ()
    def __init__(self, color):
        super().__init__('Fibonacci', color)
    def decide(self, p, r, t):
        if len(p) < 50:
            return 'hold'
        w = p[-50:]
        hi, lo = max(w), min(w)
        rng = hi - lo
        if rng < 0.01:
            return 'hold'
        l236 = hi - 0.236 * rng
        l382 = hi - 0.382 * rng
        l500 = hi - 0.500 * rng
        l618 = hi - 0.618 * rng
        pr = p[-1]
        prev_pr = p[-2] if len(p) >= 2 else pr
        if pr > l500:
            if pr <= l382 and prev_pr > l382:
                return 'buy'
            if pr <= l618 and prev_pr > l618:
                return 'buy'
            if pr <= l236 and prev_pr > l236:
                return 'buy'
        elif pr < l500:
            if pr >= l618 and prev_pr < l618:
                return 'sell'
            if pr >= l382 and prev_pr < l382:
                return 'sell'
            if pr >= l236 and prev_pr < l236:
                return 'sell'
        return 'hold'


class VolSqueezeAgent(Agent):
    __slots__ = ('_was_squeezed',)
    def __init__(self, color):
        super().__init__('Vol Squeeze', color)
        self._was_squeezed = False
    def reset_portfolio(self):
        super().reset_portfolio()
        self._was_squeezed = False
    def decide(self, p, r, t):
        if len(p) < 22:
            return 'hold'
        bl, bm, bu = calc_bollinger(p, 20, 2.0)
        if bl is None:
            return 'hold'
        w = p[-20:]
        kma = np.mean(w)
        kelt_rng = max(w) - min(w)
        kelt_lo = kma - 1.5 * kelt_rng / 2
        kelt_hi = kma + 1.5 * kelt_rng / 2
        is_squeezed = (bl > kelt_lo) and (bu < kelt_hi)
        if self._was_squeezed and not is_squeezed and len(p) >= 5:
            x = np.arange(5)
            y = np.array(p[-5:])
            slope = np.polyfit(x, y, 1)[0]
            self._was_squeezed = is_squeezed
            return 'buy' if slope > 0 else 'sell'
        self._was_squeezed = is_squeezed
        return 'hold'


class EnsembleAgent(Agent):
    __slots__ = ()
    def __init__(self, color):
        super().__init__('Ensemble', color)
    def decide(self, p, r, t):
        if t < WARMUP:
            return 'hold'
        s = 0
        if len(r) >= 10:
            m = sum(r[-10:])
            s += 1 if m > 0.015 else (-1 if m < -0.015 else 0)
        if len(p) >= 30:
            ma = np.mean(p[-30:])
            d = (p[-1] - ma) / max(ma, 0.01)
            s += 1 if d < -0.02 else (-1 if d > 0.02 else 0)
        if len(p) >= 22:
            s += 1 if np.mean(p[-5:]) > np.mean(p[-20:]) else -1
        if len(r) >= 14:
            rsi = calc_rsi(r, 14)
            s += 1 if rsi < 35 else (-1 if rsi > 65 else 0)
        low, ma, up = calc_bollinger(p, 20, 2.0)
        if low is not None:
            rng = up - low
            if rng > 0.01:
                pos = (p[-1] - low) / rng
                s += 1 if pos < 0.15 else (-1 if pos > 0.85 else 0)
        if abs(s) >= 2:
            return 'buy' if s > 0 else 'sell'
        return 'hold'


class RandomAgent(Agent):
    __slots__ = ()
    def __init__(self, color):
        super().__init__('Random', color)
    def decide(self, p, r, t):
        if t < WARMUP:
            return 'hold'
        return random.choices(['buy', 'sell', 'hold'], weights=[1, 1, 4])[0]