"""Simulation engine managing market and agents."""

from config import TICKS_PER_GEN, INITIAL_CASH
from market import Market
from agents_learning import QLearningAgent, NeuralAgent, MuZeroAgent, ActorCriticAgent
from agents_strategies import (
    MomentumAgent, MeanReversionAgent, MACrossoverAgent, BollingerAgent,
    RSIAgent, BreakoutAgent, MACDAgent, IchimokuAgent, TurtleAgent,
    ZScoreAgent, ParabolicSARAgent, StochasticAgent, TripleEMAAgent,
    FibonacciAgent, VolSqueezeAgent, EnsembleAgent, RandomAgent
)
from config import (
    GREEN, ORANGE, BLUE, MINT, SALMON, GOLD, LIME, SKY, CORAL,
    LAVEN, PALE_GREEN, DARK_CYAN, ROSE, BROWN, CYAN, PURPLE,
    MAGENTA, TEAL, DIM
)
from PySide6.QtGui import QColor


# Agent color mapping - ALL must be QColor objects
AGENT_COLORS = {
    'Q-Learner': CYAN,
    'Neural Net': PURPLE,
    'MuZero': MAGENTA,
    'Actor-Critic': TEAL,
    'Momentum': GREEN,
    'Mean Revert': ORANGE,
    'MA Cross': BLUE,
    'Bollinger': MINT,
    'RSI': SALMON,
    'Breakout': GOLD,
    'MACD': LIME,
    'Ichimoku': SKY,
    'Turtle': CORAL,
    'Z-Score': LAVEN,
    'Parabolic SAR': PALE_GREEN,
    'Stochastic': DARK_CYAN,
    'Triple EMA': ROSE,
    'Fibonacci': BROWN,
    'Vol Squeeze': QColor(220, 220, 100),  # Fixed: was tuple
    'Ensemble': QColor(200, 200, 200),     # Fixed: was tuple
    'Random': DIM,
}


def create_agents():
    """Factory function to create all 21 agents with proper colors."""
    return [
        QLearningAgent(),
        NeuralAgent(),
        MuZeroAgent(),
        ActorCriticAgent(),
        MomentumAgent(GREEN),
        MeanReversionAgent(ORANGE),
        MACrossoverAgent(BLUE),
        BollingerAgent(MINT),
        RSIAgent(SALMON),
        BreakoutAgent(GOLD),
        MACDAgent(LIME),
        IchimokuAgent(SKY),
        TurtleAgent(CORAL),
        ZScoreAgent(LAVEN),
        ParabolicSARAgent(PALE_GREEN),
        StochasticAgent(DARK_CYAN),
        TripleEMAAgent(ROSE),
        FibonacciAgent(BROWN),
        VolSqueezeAgent(QColor(220, 220, 100)),  # Fixed: was tuple
        EnsembleAgent(QColor(200, 200, 200)),     # Fixed: was tuple
        RandomAgent(DIM),
    ]


def assign_learning_agent_colors(agents):
    """Assign colors to learning agents (they can't in __init__ due to import order)."""
    for agent in agents:
        if agent.name in AGENT_COLORS and agent.color is None:
            agent.color = AGENT_COLORS[agent.name]


class Simulation:
    """Main simulation controller."""
    
    __slots__ = ('market', 'agents', 'gen', 'tick', 'speed', 'paused', 'sel',
                 'gscores', 'gavgs', '_sorted_cache', '_cache_valid')
    
    def __init__(self):
        self.market = Market()
        self.agents = create_agents()
        assign_learning_agent_colors(self.agents)
        
        self.gen = 1
        self.tick = 0
        self.speed = 5
        self.paused = False
        self.sel = 2
        
        self.gscores = {a.name: [] for a in self.agents}
        self.gavgs = {a.name: [] for a in self.agents}
        
        self._sorted_cache = None
        self._cache_valid = False
    
    def step(self):
        """Advance simulation by one tick."""
        price, ret = self.market.step()
        self.tick += 1
        self._cache_valid = False
        
        # Convert deques to lists for slicing support
        prices = list(self.market.prices)
        returns = list(self.market.returns)
        tick = self.tick
        
        for agent in self.agents:
            action = agent.decide(prices, returns, tick)
            agent.execute(action, price)
            agent.record(price)
        
        if self.tick >= TICKS_PER_GEN:
            self._end_generation()
    
    def _end_generation(self):
        """Handle end of generation: record scores, train learning agents, reset."""
        for agent in self.agents:
            eq = agent.equity_curve[-1]
            self.gscores[agent.name].append(eq)
            
            prev_avg = self.gavgs[agent.name][-1] if self.gavgs[agent.name] else eq
            n = len(self.gscores[agent.name])
            self.gavgs[agent.name].append(prev_avg + (eq - prev_avg) / n)
            
            agent.end_generation()
            agent.reset_portfolio()
        
        self.market.reset()
        self.tick = 0
        self.gen += 1
        self._cache_valid = False
    
    def sorted_agents(self):
        """Return agents sorted by current equity (cached)."""
        if not self._cache_valid:
            self._sorted_cache = sorted(
                self.agents, 
                key=lambda a: a.equity_curve[-1], 
                reverse=True
            )
            self._cache_valid = True
        return self._sorted_cache