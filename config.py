"""Configuration constants and color palette."""

# ═══════════════════════════════════════════════════════════════
# WINDOW & SIMULATION
# ═══════════════════════════════════════════════════════════════
WIDTH, HEIGHT = 1500, 920
FPS = 60
TICKS_PER_GEN = 500
INITIAL_CASH = 10000.0
TRADE_SIZE = 5
COMMISSION = 0.001
MAX_POSITION = 80
WARMUP = 25
MAX_PRICE_HISTORY = 600
MAX_EQUITY_DISPLAY = 300
MAX_TRADE_MARKERS = 200
MAX_GEN_SCORES_DISPLAY = 100

# ═══════════════════════════════════════════════════════════════
# MARKET REGIMES
# ═══════════════════════════════════════════════════════════════
REGIMES = {
    'bull':     {'drift':  0.0004, 'vol': 0.013, 'dur': (80, 220)},
    'bear':     {'drift': -0.0004, 'vol': 0.024, 'dur': (60, 180)},
    'sideways': {'drift':  0.0000, 'vol': 0.007, 'dur': (50, 160)},
}
REGIME_COLORS = {'bull': None, 'bear': None, 'sideways': None}

# ═══════════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════════
from PySide6.QtGui import QColor

BG = QColor(12, 12, 22)
PANEL = QColor(20, 22, 35)
PANEL_HOVER = QColor(28, 30, 48)
BORDER = QColor(45, 48, 70)
TEXT = QColor(190, 195, 210)
DIM = QColor(100, 105, 130)
BRIGHT = QColor(235, 238, 250)
GRID = QColor(30, 33, 50)

GREEN = QColor(30, 210, 110)
RED = QColor(230, 60, 60)
BLUE = QColor(60, 130, 235)
YELLOW = QColor(240, 215, 50)
PURPLE = QColor(175, 80, 235)
ORANGE = QColor(235, 150, 40)
CYAN = QColor(40, 215, 225)
PINK = QColor(235, 90, 175)
MAGENTA = QColor(220, 60, 180)
MINT = QColor(100, 220, 150)
SALMON = QColor(220, 110, 110)
GOLD = QColor(220, 195, 80)
LAVEN = QColor(160, 130, 220)
TEAL = QColor(60, 210, 195)
LIME = QColor(180, 230, 50)
SKY = QColor(100, 180, 255)
CORAL = QColor(255, 127, 80)
PALE_GREEN = QColor(130, 255, 160)
DARK_CYAN = QColor(0, 180, 180)
ROSE = QColor(255, 100, 130)
BROWN = QColor(180, 130, 70)

REGIME_COLORS['bull'] = GREEN
REGIME_COLORS['bear'] = RED
REGIME_COLORS['sideways'] = YELLOW

# Pre-computed dimmed colors cache
_DIM_CACHE: dict = {}

def dim_color(col: QColor, factor: float = 0.4) -> QColor:
    """Get or compute a dimmed version of a color."""
    key = (col.rgba(), factor)
    if key not in _DIM_CACHE:
        _DIM_CACHE[key] = QColor(
            int(col.red() * factor),
            int(col.green() * factor),
            int(col.blue() * factor)
        )
    return _DIM_CACHE[key]

# ═══════════════════════════════════════════════════════════════
# LAYOUT RECTANGLES (improved proportions)
# ═══════════════════════════════════════════════════════════════
from PySide6.QtCore import QRectF

# Top row: Price chart (left) + Equity chart (right)
PRICE_RECT = QRectF(10, 46, 700, 305)
EQUITY_RECT = QRectF(720, 46, 770, 305)

# Bottom row: Gen chart + Leaderboard + Info panel
GEN_RECT = QRectF(10, 361, 460, 440)
BOARD_RECT = QRectF(480, 361, 440, 440)
INFO_RECT = QRectF(930, 361, 560, 440)

CHART_PADDING = {'top': 26, 'bottom': 12, 'left': 58, 'right': 12}