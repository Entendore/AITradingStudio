"""Machine learning-based trading agents."""

import numpy as np
import random
from collections import defaultdict

from agents_base import Agent, extract_features
from config import INITIAL_CASH, WARMUP


class QLearningAgent(Agent):
    """Tabular Q-Learning agent with discretized state space."""
    
    __slots__ = ('qt', 'lr', 'gamma', 'eps', '_ls', '_la', '_leq')
    
    def __init__(self):
        self.qt = defaultdict(lambda: np.zeros(3, dtype=np.float64))
        self.lr = 0.15
        self.gamma = 0.9
        self.eps = 0.35
        self._ls = None
        self._la = None
        self._leq = INITIAL_CASH
        super().__init__('Q-Learner', None)  # Color set after import
    
    def reset_portfolio(self):
        super().reset_portfolio()
        self._ls = None
        self._la = None
        self._leq = INITIAL_CASH
    
    def _discretize(self, prices, returns):
        n = len(returns)
        r5 = sum(returns[-5:]) if n >= 5 else 0.0
        r20 = sum(returns[-20:]) if n >= 20 else 0.0
        vol = np.std(returns[-20:]) if n >= 20 else 0.005
        pos = 0 if self.shares <= 0 else (1 if self.shares <= 20 else 2)
        return (
            int(np.clip(r5 / 0.012 + 2.5, 0, 4)),
            int(np.clip(r20 / 0.035 + 2.5, 0, 4)),
            int(np.clip(vol / 0.012, 0, 2)),
            pos
        )
    
    def decide(self, prices, returns, tick: int) -> str:
        if tick < WARMUP:
            return 'hold'
        
        s = self._discretize(prices, returns)
        q_vals = self.qt[s]
        
        if random.random() < self.eps:
            a = random.randint(0, 2)
        else:
            a = int(np.argmax(q_vals))
        
        # Q-learning update
        if self._ls is not None:
            eq = self.equity(prices[-1])
            reward = np.clip(eq - self._leq, -80, 80)
            old_q = self.qt[self._ls][self._la]
            self.qt[self._ls][self._la] = old_q + self.lr * (
                reward + self.gamma * np.max(q_vals) - old_q
            )
            self._leq = eq
        
        self._ls, self._la = s, a
        self.eps = max(0.03, self.eps * 0.997)
        return ('buy', 'hold', 'sell')[a]


class NeuralAgent(Agent):
    """Simple 2-layer neural network with online gradient descent."""
    
    __slots__ = ('W1', 'b1', 'W2', 'b2', 'lr', 'eps', '_pf', '_ph', '_pa', '_leq')
    
    def __init__(self):
        n_in, n_h = 12, 10
        self.W1 = np.random.randn(n_h, n_in) * 0.05
        self.b1 = np.zeros(n_h, dtype=np.float64)
        self.W2 = np.random.randn(3, n_h) * 0.05
        self.b2 = np.zeros(3, dtype=np.float64)
        self.lr = 0.003
        self.eps = 0.30
        self._pf = None
        self._ph = None
        self._pa = None
        self._leq = INITIAL_CASH
        super().__init__('Neural Net', None)
    
    def reset_portfolio(self):
        super().reset_portfolio()
        self._pf = None
        self._ph = None
        self._pa = None
        self._leq = INITIAL_CASH
    
    def decide(self, prices, returns, tick: int) -> str:
        if tick < WARMUP:
            return 'hold'
        
        f = extract_features(prices, returns, self.shares)
        h = np.maximum(0, self.W1 @ f + self.b1)  # ReLU
        output = self.W2 @ h + self.b2
        
        # Online learning update
        if self._pf is not None:
            eq = self.equity(prices[-1])
            reward = np.clip(eq - self._leq, -80, 80) * self.lr
            
            d2 = np.zeros(3)
            d2[self._pa] = reward
            self.W2 += np.outer(d2, self._ph)
            self.b2 += d2
            
            d1 = (self.W2.T @ d2) * (self._ph > 0).astype(np.float64)
            self.W1 += np.outer(d1, self._pf)
            self.b1 += d1
            self._leq = eq
        
        if random.random() < self.eps:
            a = random.randint(0, 2)
        else:
            a = int(np.argmax(output))
        
        self._pf = f.copy()
        self._ph = h.copy()
        self._pa = a
        self.eps = max(0.03, self.eps * 0.997)
        return ('buy', 'hold', 'sell')[a]


class MuZeroAgent(Agent):
    """Simplified MuZero with representation, prediction, and dynamics networks."""
    
    __slots__ = (
        'nf', 'nl', 'nh', 'na',
        'Wr1', 'br1', 'Wr2', 'br2',
        'Wv1', 'bv1', 'Wv2', 'bv2',
        'Wp1', 'bp1', 'Wp2', 'bp2',
        'Wd1', 'bd1', 'Wd2', 'bd2',
        'Wdr', 'bdr',
        'lr_r', 'lr_p', 'lr_d', 'eps',
        'mcts_sims', 'mcts_hor',
        'buffer', '_pf', '_pa', '_leq'
    )
    
    def __init__(self):
        nf, nl, nh, na = 12, 14, 14, 3
        self.nf, self.nl, self.nh, self.na = nf, nl, nh, na
        
        # Representation network
        self.Wr1 = np.random.randn(nh, nf) * 0.08
        self.br1 = np.zeros(nh, dtype=np.float64)
        self.Wr2 = np.random.randn(nl, nh) * 0.08
        self.br2 = np.zeros(nl, dtype=np.float64)
        
        # Value network
        self.Wv1 = np.random.randn(nh, nl) * 0.08
        self.bv1 = np.zeros(nh, dtype=np.float64)
        self.Wv2 = np.random.randn(1, nh) * 0.08
        self.bv2 = np.zeros(1, dtype=np.float64)
        
        # Policy network
        self.Wp1 = np.random.randn(nh, nl) * 0.08
        self.bp1 = np.zeros(nh, dtype=np.float64)
        self.Wp2 = np.random.randn(na, nh) * 0.08
        self.bp2 = np.zeros(na, dtype=np.float64)
        
        # Dynamics network
        self.Wd1 = np.random.randn(nh, nl + na) * 0.08
        self.bd1 = np.zeros(nh, dtype=np.float64)
        self.Wd2 = np.random.randn(nl, nh) * 0.08
        self.bd2 = np.zeros(nl, dtype=np.float64)
        self.Wdr = np.random.randn(1, nh) * 0.08
        self.bdr = np.zeros(1, dtype=np.float64)
        
        self.lr_r, self.lr_p, self.lr_d = 0.004, 0.004, 0.003
        self.eps = 0.30
        self.mcts_sims = 4
        self.mcts_hor = 3
        self.buffer = []
        self._pf = None
        self._pa = None
        self._leq = INITIAL_CASH
        super().__init__('MuZero', None)
    
    def reset_portfolio(self):
        super().reset_portfolio()
        self._pf = None
        self._pa = None
        self._leq = INITIAL_CASH
    
    @staticmethod
    def _forward(W1, b1, W2, b2, x):
        h = np.maximum(0, W1 @ x + b1)
        return W2 @ h + b2, h
    
    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()
    
    def encode(self, f):
        y, _ = self._forward(self.Wr1, self.br1, self.Wr2, self.br2, f)
        return np.maximum(0, y)
    
    def predict(self, lat):
        vz, _ = self._forward(self.Wv1, self.bv1, self.Wv2, self.bv2, lat)
        pz, _ = self._forward(self.Wp1, self.bp1, self.Wp2, self.bp2, lat)
        return np.tanh(vz[0]), pz
    
    def dynamics(self, lat, ai):
        aoh = np.zeros(self.na)
        aoh[ai] = 1.0
        x = np.concatenate([lat, aoh])
        hd = np.maximum(0, self.Wd1 @ x + self.bd1)
        return np.maximum(0, self.Wd2 @ hd + self.bd2), np.tanh((self.Wdr @ hd + self.bdr)[0])
    
    def _mcts(self, lat):
        vis = np.zeros(3)
        vals = np.zeros(3)
        
        for _ in range(self.mcts_sims):
            ucb = np.full(3, -1e9)
            nz = vis > 0
            
            if np.any(nz):
                log_n = np.log(max(vis.sum(), 1.0))
                ucb[nz] = vals[nz] / vis[nz] + 1.5 * np.sqrt(log_n / vis[nz])
            ucb[~nz] = 1e9
            
            ar = int(np.argmax(ucb))
            cl, r = self.dynamics(lat, ar)
            tv = r
            d = 0.95
            
            for _ in range(self.mcts_hor - 1):
                v, p = self.predict(cl)
                cl, r = self.dynamics(cl, int(np.argmax(p)))
                tv += d * r
                d *= 0.95
            
            vb, _ = self.predict(cl)
            tv += d * vb
            vis[ar] += 1
            vals[ar] += tv
        
        return vals / np.maximum(vis, 1)
    
    def decide(self, prices, returns, tick: int) -> str:
        if tick < WARMUP:
            self._pf = extract_features(prices, returns, self.shares)
            self._pa = 1
            self._leq = self.equity(prices[-1])
            return 'hold'
        
        f = extract_features(prices, returns, self.shares)
        
        if self._pf is not None:
            rw = np.clip((self.equity(prices[-1]) - self._leq) / INITIAL_CASH, -1, 1)
            self.buffer.append((self._pf.copy(), self._pa, rw, f.copy()))
        
        if random.random() < self.eps:
            a = random.randint(0, 2)
        else:
            a = int(np.argmax(self._mcts(self.encode(f))))
        
        self._pf = f.copy()
        self._pa = a
        self._leq = self.equity(prices[-1])
        return ('buy', 'hold', 'sell')[a]
    
    def end_generation(self):
        if len(self.buffer) < 30:
            return
        
        rews = [t[2] for t in self.buffer]
        n_steps, gamma = 10, 0.95
        rets = []
        for i in range(len(rews)):
            r = sum((gamma ** j) * rews[i + j] for j in range(min(n_steps, len(rews) - i)))
            rets.append(np.clip(r, -2, 2))
        
        clip_val = 0.1
        
        for _ in range(3):
            idx = list(range(len(self.buffer)))
            random.shuffle(idx)
            
            for i in idx:
                feat, aidx, imm_r, nf = self.buffer[i]
                tv = rets[i]
                
                # Forward passes
                lr_raw, hr = self._forward(self.Wr1, self.br1, self.Wr2, self.br2, feat)
                lat = np.maximum(0, lr_raw)
                nlr_raw, _ = self._forward(self.Wr1, self.br1, self.Wr2, self.br2, nf)
                nlat = np.maximum(0, nlr_raw)
                
                # Value prediction
                hv = np.maximum(0, self.Wv1 @ lat + self.bv1)
                vz = (self.Wv2 @ hv + self.bv2)[0]
                pv = np.tanh(vz)
                
                # Policy prediction
                hp = np.maximum(0, self.Wp1 @ lat + self.bp1)
                pz = self.Wp2 @ hp + self.bp2
                psm = self._softmax(pz)
                
                # Dynamics
                aoh = np.zeros(self.na)
                aoh[aidx] = 1.0
                din = np.concatenate([lat, aoh])
                hd = np.maximum(0, self.Wd1 @ din + self.bd1)
                nlr_pred = self.Wd2 @ hd + self.bd2
                nlp = np.maximum(0, nlr_pred)
                rz = (self.Wdr @ hd + self.bdr)[0]
                pr = np.tanh(rz)
                
                # Gradients
                dvz = (pv - tv) * (1 - pv ** 2) * 0.5
                dpz = rets[i] * (psm - aoh)
                dnlp = 0.5 * (nlp - nlat)
                dnlr = dnlp * (nlr_pred > 0).astype(np.float64)
                drz = (pr - imm_r) * (1 - pr ** 2) * 0.5
                dhd = (self.Wd2.T @ dnlr + drz * self.Wdr[0]) * (hd > 0).astype(np.float64)
                
                # Dynamics updates
                self.Wd2 -= self.lr_d * np.clip(np.outer(dnlr, hd), -clip_val, clip_val)
                self.bd2 -= self.lr_d * np.clip(dnlr, -clip_val, clip_val)
                self.Wdr -= self.lr_d * np.clip(np.outer(np.array([drz]), hd), -clip_val, clip_val)
                self.bdr -= self.lr_d * np.clip(np.array([drz]), -clip_val, clip_val)
                self.Wd1 -= self.lr_d * np.clip(np.outer(dhd, din), -clip_val, clip_val)
                self.bd1 -= self.lr_d * np.clip(dhd, -clip_val, clip_val)
                
                # Value/policy gradients
                dld = (self.Wd1.T @ dhd)[:self.nl]
                dhv = (dvz * self.Wv2[0]) * (hv > 0).astype(np.float64)
                self.Wv2 -= self.lr_p * np.clip(np.outer(np.array([dvz]), hv), -clip_val, clip_val)
                self.bv2 -= self.lr_p * np.clip(np.array([dvz]), -clip_val, clip_val)
                self.Wv1 -= self.lr_p * np.clip(np.outer(dhv, lat), -clip_val, clip_val)
                self.bv1 -= self.lr_p * np.clip(dhv, -clip_val, clip_val)
                
                dhp = (self.Wp2.T @ dpz) * (hp > 0).astype(np.float64)
                self.Wp2 -= self.lr_p * np.clip(np.outer(dpz, hp), -clip_val, clip_val)
                self.bp2 -= self.lr_p * np.clip(dpz, -clip_val, clip_val)
                self.Wp1 -= self.lr_p * np.clip(np.outer(dhp, lat), -clip_val, clip_val)
                self.bp1 -= self.lr_p * np.clip(dhp, -clip_val, clip_val)
                
                # Representation gradients
                dlv = self.Wv1.T @ dhv
                dlp = self.Wp1.T @ dhp
                dlat = dld + dlv + dlp
                dlr = dlat * (lr_raw > 0).astype(np.float64)
                self.Wr2 -= self.lr_r * np.clip(np.outer(dlr, hr), -clip_val, clip_val)
                self.br2 -= self.lr_r * np.clip(dlr, -clip_val, clip_val)
                dhr = (self.Wr2.T @ dlr) * (hr > 0).astype(np.float64)
                self.Wr1 -= self.lr_r * np.clip(np.outer(dhr, feat), -clip_val, clip_val)
                self.br1 -= self.lr_r * np.clip(dhr, -clip_val, clip_val)
        
        self.buffer.clear()
        self.eps = max(0.02, self.eps * 0.997)


class ActorCriticAgent(Agent):
    """Actor-Critic with separate policy and value networks."""
    
    __slots__ = (
        'Wa1', 'ba1', 'Wa2', 'ba2',
        'Wc1', 'bc1', 'Wc2', 'bc2',
        'lr', 'gamma', 'eps',
        '_prev', '_pp'
    )
    
    def __init__(self):
        n_in, n_h = 12, 12
        self.Wa1 = np.random.randn(n_h, n_in) * 0.05
        self.ba1 = np.zeros(n_h, dtype=np.float64)
        self.Wa2 = np.random.randn(3, n_h) * 0.05
        self.ba2 = np.zeros(3, dtype=np.float64)
        self.Wc1 = np.random.randn(n_h, n_in) * 0.05
        self.bc1 = np.zeros(n_h, dtype=np.float64)
        self.Wc2 = np.random.randn(1, n_h) * 0.05
        self.bc2 = np.zeros(1, dtype=np.float64)
        self.lr = 0.003
        self.gamma = 0.95
        self.eps = 0.30
        self._prev = None
        self._pp = None
        super().__init__('Actor-Critic', None)
    
    def reset_portfolio(self):
        super().reset_portfolio()
        self._prev = None
        self._pp = None
    
    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()
    
    def decide(self, prices, returns, tick: int) -> str:
        if tick < WARMUP:
            return 'hold'
        
        feat = extract_features(prices, returns, self.shares)
        ha = np.maximum(0, self.Wa1 @ feat + self.ba1)
        logits = self.Wa2 @ ha + self.ba2
        probs = self._softmax(logits)
        
        hc = np.maximum(0, self.Wc1 @ feat + self.bc1)
        value = (self.Wc2 @ hc + self.bc2)[0]
        
        if random.random() < self.eps:
            aidx = random.randint(0, 2)
        else:
            # Ensure probabilities are properly normalized for np.random.choice
            p = np.clip(probs, 1e-8, 1.0)
            p = p / p.sum()  # Re-normalize after clipping
            aidx = int(np.random.choice(3, p=p))
        
        # TD learning update
        if self._prev is not None:
            pf, pha, phc, pa, pv = self._prev
            reward = np.clip(
                self.equity(prices[-1]) - self.equity(prices[-2]) if len(prices) > 1 else 0,
                -80, 80
            )
            td_err = reward + self.gamma * value - pv
            
            # Clip TD error to prevent instability
            td_err = np.clip(td_err, -100, 100)
            
            # Critic update
            dv = td_err
            dhc = (dv * self.Wc2[0]) * (phc > 0).astype(np.float64)
            self.Wc2 += self.lr * np.outer(np.array([dv]), phc)
            self.bc2 += self.lr * np.array([dv])
            self.Wc1 += self.lr * np.outer(dhc, pf)
            self.bc1 += self.lr * dhc
            
            # Actor update
            dlog = np.array([(1.0 if j == pa else 0.0) - self._pp[j] for j in range(3)])
            d_ent = np.log(np.clip(self._pp, 1e-8, 1.0)) + 1.0
            dp = -td_err * dlog + 0.01 * d_ent
            dha = (self.Wa2.T @ dp) * (pha > 0).astype(np.float64)
            self.Wa2 += self.lr * np.outer(dp, pha)
            self.ba2 += self.lr * dp
            self.Wa1 += self.lr * np.outer(dha, pf)
            self.ba1 += self.lr * dha
        
        self._pp = probs.copy()
        self._prev = (feat.copy(), ha.copy(), hc.copy(), aidx, value)
        self.eps = max(0.03, self.eps * 0.997)
        return ('buy', 'hold', 'sell')[aidx]