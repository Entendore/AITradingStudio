"""Main application entry point and UI rendering."""

import sys
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QFont, QPolygonF, QPen, QLinearGradient

from config import (
    WIDTH, HEIGHT, FPS, TICKS_PER_GEN, INITIAL_CASH,
    MAX_EQUITY_DISPLAY, MAX_GEN_SCORES_DISPLAY,
    BG, PANEL, PANEL_HOVER, BORDER, TEXT, DIM, BRIGHT, GRID,
    GREEN, RED, YELLOW, PINK,
    PRICE_RECT, EQUITY_RECT, GEN_RECT, BOARD_RECT, INFO_RECT,
    CHART_PADDING, REGIME_COLORS, dim_color
)
from simulation import Simulation


# ═══════════════════════════════════════════════════════════════
# PRE-COMPUTED UI ELEMENTS
# ═══════════════════════════════════════════════════════════════
FONTS = {
    'title': QFont("Segoe UI", 20, QFont.Bold),
    'large': QFont("Consolas", 16, QFont.Bold),
    'medium': QFont("Consolas", 12),
    'small': QFont("Consolas", 10),
    'tiny': QFont("Consolas", 9),
    'mini': QFont("Consolas", 8),
}

RANK_COLORS = [YELLOW, QColor(200, 200, 210), QColor(200, 140, 60)]
RANK_MEDALS = ['🥇', '🥈', '🥉']

CHART_AREAS = {
    'price': QRectF(
        PRICE_RECT.x() + CHART_PADDING['left'],
        PRICE_RECT.y() + CHART_PADDING['top'],
        PRICE_RECT.width() - CHART_PADDING['left'] - CHART_PADDING['right'],
        PRICE_RECT.height() - CHART_PADDING['top'] - CHART_PADDING['bottom']
    ),
    'equity': QRectF(
        EQUITY_RECT.x() + CHART_PADDING['left'],
        EQUITY_RECT.y() + CHART_PADDING['top'],
        EQUITY_RECT.width() - CHART_PADDING['left'] - CHART_PADDING['right'],
        EQUITY_RECT.height() - CHART_PADDING['top'] - CHART_PADDING['bottom']
    ),
    'gen': QRectF(
        GEN_RECT.x() + CHART_PADDING['left'],
        GEN_RECT.y() + CHART_PADDING['top'],
        GEN_RECT.width() - CHART_PADDING['left'] - CHART_PADDING['right'],
        GEN_RECT.height() - CHART_PADDING['top'] - CHART_PADDING['bottom']
    ),
}


class SimulationWidget(QWidget):
    """Custom widget handling all rendering and input."""
    
    def __init__(self):
        super().__init__()
        self.sim = Simulation()
        self.setFixedSize(WIDTH, HEIGHT)
        self.setFocusPolicy(Qt.StrongFocus)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_loop)
        self.timer.start(int(1000 / FPS))
    
    def _update_loop(self):
        if not self.sim.paused:
            steps = self.sim.speed
            for _ in range(steps):
                self.sim.step()
        self.update()
    
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            self.sim.paused = not self.sim.paused
        elif key == Qt.Key_Up:
            self.sim.speed = min(self.sim.speed * 2, 128)
        elif key == Qt.Key_Down:
            self.sim.speed = max(self.sim.speed // 2, 1)
        elif key == Qt.Key_Left:
            self.sim.sel = (self.sim.sel - 1) % len(self.sim.agents)
        elif key == Qt.Key_Right:
            self.sim.sel = (self.sim.sel + 1) % len(self.sim.agents)
        elif key == Qt.Key_R:
            self.sim = Simulation()
        elif key == Qt.Key_G:
            self.sim._end_generation()
        elif key == Qt.Key_Escape:
            QApplication.quit()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BG)
        
        self._draw_header(p)
        self._draw_price_chart(p)
        self._draw_equity_chart(p)
        self._draw_gen_chart(p)
        self._draw_leaderboard(p)
        self._draw_info_panel(p)
        self._draw_controls(p)
        
        p.end()
    
    # ─────────────────────────────────────────────────────────
    # HELPER METHODS
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _draw_panel(p: QPainter, rect: QRectF, title: str, accent: QColor):
        p.setPen(Qt.NoPen)
        p.setBrush(PANEL)
        p.drawRoundedRect(rect, 8, 8)
        
        p.setPen(BORDER)
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect, 8, 8)
        
        # Gradient accent line at top
        grad = QLinearGradient(rect.x(), rect.y(), rect.x() + rect.width(), rect.y())
        grad.setColorAt(0, accent)
        grad.setColorAt(1, QColor(accent.red(), accent.green(), accent.blue(), 80))
        p.setPen(QPen(QBrush(grad), 2))
        p.drawLine(
            int(rect.x() + 8), int(rect.y() + 1),
            int(rect.x() + rect.width() - 8), int(rect.y() + 1)
        )
        
        p.setPen(BRIGHT)
        p.setFont(FONTS['medium'])
        p.drawText(int(rect.x() + 12), int(rect.y() + 20), title)
    
    @staticmethod
    def _draw_grid(p: QPainter, ca: QRectF, mn: float, mx: float, n: int = 5):
        rng = max(mx - mn, 1e-6)
        p.setFont(FONTS['mini'])
        
        for i in range(n + 1):
            y = ca.y() + ca.height() - (i / n) * ca.height()
            
            p.setPen(GRID)
            p.drawLine(int(ca.x()), int(y), int(ca.x() + ca.width()), int(y))
            
            p.setPen(DIM)
            val = mn + (i / n) * rng
            label = f"{val:.1f}" if abs(val) < 1000 else f"{val:.0f}"
            p.drawText(int(ca.x() - 55), int(y) + 4, label)
    
    @staticmethod
    def _compute_points(data, ca: QRectF, mn: float, mx: float) -> list:
        rng = max(mx - mn, 1e-6)
        n = len(data)
        w = ca.width()
        h = ca.height()
        x0 = ca.x()
        y0 = ca.y() + h
        
        points = []
        inv_n = 1.0 / max(n - 1, 1)
        inv_rng = h / rng
        
        for i, v in enumerate(data):
            x = x0 + i * inv_n * w
            y = y0 - (v - mn) * inv_rng
            points.append(QPointF(x, y))
        
        return points
    
    @staticmethod
    def _draw_line(p: QPainter, points: list, color: QColor, width: float = 1.0, alpha: float = 1.0):
        if len(points) < 2:
            return
        if alpha < 1.0:
            p.setOpacity(alpha)
        p.setPen(QPen(color, width))
        p.drawPolyline(QPolygonF(points))
        if alpha < 1.0:
            p.setOpacity(1.0)
    
    @staticmethod
    def _draw_fill(p: QPainter, points: list, color: QColor, alpha: float = 0.15):
        if len(points) < 2:
            return
        fill_points = list(points)
        fill_points.append(QPointF(points[-1].x(), points[0].y()))
        fill_points.append(QPointF(points[0].x(), points[0].y()))
        
        p.setPen(Qt.NoPen)
        p.setOpacity(alpha)
        p.setBrush(color)
        p.drawPolygon(QPolygonF(fill_points))
        p.setOpacity(1.0)
    
    # ─────────────────────────────────────────────────────────
    # PANEL DRAWING METHODS
    # ─────────────────────────────────────────────────────────
    def _draw_header(self, p: QPainter):
        # Title with gradient effect
        p.setFont(FONTS['title'])
        p.setPen(BRIGHT)
        p.drawText(12, 38, "⚡ ULTIMATE STOCK SIM")
        
        p.setFont(FONTS['medium'])
        p.setPen(DIM)
        p.drawText(340, 35, "21 Competing Algorithms")
        
        # Status info on the right
        m = self.sim.market
        rc = REGIME_COLORS[m.regime]
        
        # Regime badge
        regime_txt = m.regime.upper()
        rx = WIDTH - 85
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(rc.red(), rc.green(), rc.blue(), 40))
        p.drawRoundedRect(QRectF(rx - 5, 10, 75, 22), 4, 4)
        p.setPen(rc)
        p.setFont(FONTS['small'])
        p.drawText(int(rx), int(26), regime_txt)
        
        # Stats
        price_change = m.price - 100.0
        price_pct = price_change / 100.0 * 100
        
        info_parts = [
            f"Gen {self.sim.gen}",
            f"Tick {self.sim.tick}/{TICKS_PER_GEN}",
            f"{self.sim.speed}x",
            f"${m.price:.2f}",
            f"({price_pct:+.1f}%)"
        ]
        
        p.setFont(FONTS['medium'])
        p.setPen(DIM)
        full_txt = "  │  ".join(info_parts)
        fm_w = p.fontMetrics().horizontalAdvance(full_txt)
        p.drawText(WIDTH - fm_w - 95, 28, full_txt)
    
    def _draw_price_chart(self, p: QPainter):
        self._draw_panel(p, PRICE_RECT, "📊 PRICE (Last 300 Ticks)", BRIGHT)
        
        ca = CHART_AREAS['price']
        pp = list(self.sim.market.prices)[-MAX_EQUITY_DISPLAY:]
        
        if len(pp) < 2:
            return
        
        mn, mx = min(pp), max(pp)
        pd = max((mx - mn) * 0.08, 0.5)
        mn -= pd
        mx += pd
        
        self._draw_grid(p, ca, mn, mx)
        
        pts = self._compute_points(pp, ca, mn, mx)
        
        # Gradient fill based on price direction
        fill_color = GREEN if pp[-1] >= pp[0] else RED
        self._draw_fill(p, pts, fill_color, 0.12)
        self._draw_line(p, pts, BRIGHT, 2.0)
        
        # Current price with glow effect
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(BRIGHT.red(), BRIGHT.green(), BRIGHT.blue(), 60))
        p.drawEllipse(pts[-1], 8, 8)
        p.setBrush(BRIGHT)
        p.drawEllipse(pts[-1], 4, 4)
        
        # Selected agent trade markers
        ag = self.sim.agents[self.sim.sel]
        
        # Agent name badge
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ag.color.red(), ag.color.green(), ag.color.blue(), 50))
        badge_w = p.fontMetrics().horizontalAdvance(f"  {ag.name}  ") + 8
        p.drawRoundedRect(QRectF(ca.x() + 2, ca.y() + 2, badge_w, 16), 3, 3)
        p.setPen(ag.color)
        p.setFont(FONTS['tiny'])
        p.drawText(int(ca.x() + 6), int(ca.y() + 14), ag.name)
        
        # Trade markers
        si = max(0, len(self.sim.market.prices) - MAX_EQUITY_DISPLAY)
        rng = max(mx - mn, 1e-6)
        
        for ti, tt, tp in ag.trade_markers[-50:]:
            di = ti - si
            if 0 <= di < len(pp):
                x = ca.x() + (di / max(len(pp) - 1, 1)) * ca.width()
                y = ca.y() + ca.height() - ((tp - mn) / rng) * ca.height()
                marker_color = GREEN if tt == 0 else RED
                p.setPen(marker_color)
                p.setFont(FONTS['tiny'])
                p.drawText(int(x) - 4, int(y) - 4, '▲' if tt == 0 else '▼')
    
    def _draw_equity_chart(self, p: QPainter):
        self._draw_panel(p, EQUITY_RECT, "📈 EQUITY CURVES", YELLOW)
        
        ca = CHART_AREAS['equity']
        
        mn, mx = float('inf'), float('-inf')
        for agent in self.sim.agents:
            d = agent.equity_curve[-MAX_EQUITY_DISPLAY:]
            if d:
                local_min, local_max = min(d), max(d)
                if local_min < mn:
                    mn = local_min
                if local_max > mx:
                    mx = local_max
        
        if mn == float('inf'):
            return
        
        pd = max((mx - mn) * 0.05, 1)
        mn -= pd
        mx += pd
        
        self._draw_grid(p, ca, mn, mx)
        
        # Draw initial cash line
        if mn < INITIAL_CASH < mx:
            y = ca.y() + ca.height() - ((INITIAL_CASH - mn) / (mx - mn)) * ca.height()
            p.setPen(QPen(QColor(100, 100, 100, 100), 1, Qt.DashLine))
            p.drawLine(int(ca.x()), int(y), int(ca.x() + ca.width()), int(y))
            p.setPen(DIM)
            p.setFont(FONTS['mini'])
            p.drawText(int(ca.x() + ca.width() - 40), int(y) - 2, "Initial")
        
        sel = self.sim.agents[self.sim.sel]
        
        # Draw non-selected agents first (dimmed)
        for agent in self.sim.agents:
            if agent == sel:
                continue
            d = agent.equity_curve[-MAX_EQUITY_DISPLAY:]
            if len(d) < 2:
                continue
            pts = self._compute_points(d, ca, mn, mx)
            self._draw_line(p, pts, dim_color(agent.color, 0.45), 0.8)
        
        # Draw selected agent on top with fill
        d = sel.equity_curve[-MAX_EQUITY_DISPLAY:]
        if len(d) >= 2:
            pts = self._compute_points(d, ca, mn, mx)
            self._draw_fill(p, pts, sel.color, 0.12)
            self._draw_line(p, pts, sel.color, 2.5)
            
            # End dot
            p.setPen(Qt.NoPen)
            p.setBrush(sel.color)
            p.drawEllipse(pts[-1], 4, 4)
    
    def _draw_gen_chart(self, p: QPainter):
        self._draw_panel(p, GEN_RECT, "📉 GENERATION PERFORMANCE", PINK)
        
        ca = CHART_AREAS['gen']
        
        mn, mx = float('inf'), float('-inf')
        max_gen = 0
        for agent in self.sim.agents:
            d = self.sim.gscores[agent.name][-MAX_GEN_SCORES_DISPLAY:]
            if d:
                local_min, local_max = min(d), max(d)
                if local_min < mn:
                    mn = local_min
                if local_max > mx:
                    mx = local_max
                if len(d) > max_gen:
                    max_gen = len(d)
        
        if mn == float('inf') or max_gen < 2:
            p.setFont(FONTS['small'])
            p.setPen(DIM)
            p.drawText(int(ca.x() + 10), int(ca.y() + ca.height() / 2), "Waiting for data...")
            return
        
        pd = max((mx - mn) * 0.05, 50)
        mn -= pd
        mx += pd
        
        self._draw_grid(p, ca, mn, mx)
        
        # Draw initial cash line
        if mn < INITIAL_CASH < mx:
            y = ca.y() + ca.height() - ((INITIAL_CASH - mn) / (mx - mn)) * ca.height()
            p.setPen(QPen(QColor(100, 100, 100, 80), 1, Qt.DashLine))
            p.drawLine(int(ca.x()), int(y), int(ca.x() + ca.width()), int(y))
        
        sel = self.sim.agents[self.sim.sel]
        
        for agent in self.sim.agents:
            d = self.sim.gscores[agent.name][-MAX_GEN_SCORES_DISPLAY:]
            if len(d) < 2:
                continue
            pts = self._compute_points(d, ca, mn, mx)
            if agent == sel:
                self._draw_line(p, pts, agent.color, 2.0)
                p.setPen(agent.color)
                p.setBrush(agent.color)
                p.drawEllipse(pts[-1], 4, 4)
            else:
                self._draw_line(p, pts, dim_color(agent.color, 0.4), 0.8)
        
        # X-axis labels
        p.setFont(FONTS['mini'])
        p.setPen(DIM)
        step = max(1, max_gen // 6)
        for i in range(0, max_gen, step):
            x = ca.x() + (i / max(max_gen - 1, 1)) * ca.width()
            p.drawText(int(x) - 3, int(ca.y() + ca.height() + 14), str(i + 1))
    
    def _draw_leaderboard(self, p: QPainter):
        self._draw_panel(p, BOARD_RECT, "🏆 LEADERBOARD", YELLOW)
        
        y = BOARD_RECT.y() + 24
        sorted_agents = self.sim.sorted_agents()
        
        # Header
        p.setFont(FONTS['mini'])
        p.setPen(DIM)
        p.drawText(int(BOARD_RECT.x() + 30), int(y), "Agent")
        p.drawText(int(BOARD_RECT.x() + 160), int(y), "P&L")
        p.drawText(int(BOARD_RECT.x() + 290), int(y), "Return")
        p.drawText(int(BOARD_RECT.x() + 370), int(y), "Trades")
        y += 14
        
        # Separator line
        p.setPen(GRID)
        p.drawLine(
            int(BOARD_RECT.x() + 8), int(y),
            int(BOARD_RECT.x() + BOARD_RECT.width() - 8), int(y)
        )
        y += 4
        
        row_height = 17
        br = BOARD_RECT
        
        for i, ag in enumerate(sorted_agents):
            # Highlight selected row
            if ag == self.sim.agents[self.sim.sel]:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(ag.color.red(), ag.color.green(), ag.color.blue(), 25))
                p.drawRoundedRect(QRectF(br.x() + 5, y - 2, br.width() - 10, row_height), 3, 3)
            
            # Rank with medal for top 3
            p.setFont(FONTS['small'])
            if i < 3:
                p.setPen(RANK_COLORS[i])
                p.drawText(int(br.x() + 8), int(y + 12), RANK_MEDALS[i])
            else:
                p.setPen(DIM)
                p.drawText(int(br.x() + 10), int(y + 12), f"{i + 1}")
            
            # Name with color dot
            p.setPen(Qt.NoPen)
            p.setBrush(ag.color)
            p.drawEllipse(QPointF(br.x() + 32, y + 8), 4, 4)
            
            p.setPen(ag.color)
            p.setFont(FONTS['small'])
            p.drawText(int(br.x() + 42), int(y + 12), ag.name)
            
            # P&L
            eq = ag.equity_curve[-1]
            pnl = eq - INITIAL_CASH
            p.setPen(GREEN if pnl >= 0 else RED)
            p.setFont(FONTS['tiny'])
            p.drawText(int(br.x() + 160), int(y + 12), f"${pnl:+,.0f}")
            
            # Return percentage
            ret_pct = pnl / INITIAL_CASH * 100
            p.setPen(GREEN if ret_pct >= 0 else RED)
            p.drawText(int(br.x() + 290), int(y + 12), f"{ret_pct:+.1f}%")
            
            # Trades
            p.setPen(DIM)
            p.drawText(int(br.x() + 370), int(y + 12), f"{ag.total_trades}")
            
            y += row_height
        
        # Running averages section
        y += 10
        p.setPen(QPen(GRID, 1))
        p.drawLine(
            int(BOARD_RECT.x() + 8), int(y),
            int(BOARD_RECT.x() + BOARD_RECT.width() - 8), int(y)
        )
        y += 8
        
        p.setFont(FONTS['mini'])
        p.setPen(DIM)
        p.drawText(int(br.x() + 10), int(y + 10), "Running Avg (Top 4):")
        y += 16
        
        for j in range(min(4, len(sorted_agents))):
            a = sorted_agents[j]
            avgs = self.sim.gavgs[a.name]
            txt = f"• {a.name}: ${avgs[-1]:,.0f}" if avgs else f"• {a.name}: —"
            p.setPen(a.color)
            p.drawText(int(br.x() + 15), int(y + 10), txt)
            y += 14
    
    def _draw_info_panel(self, p: QPainter):
        ag = self.sim.agents[self.sim.sel]
        self._draw_panel(p, INFO_RECT, f"🤖 {ag.name}", ag.color)
        
        y = INFO_RECT.y() + 28
        eq = ag.equity_curve[-1]
        pnl = eq - INITIAL_CASH
        pc = GREEN if pnl >= 0 else RED
        
        # Main stats section
        p.setFont(FONTS['medium'])
        stats = [
            (f"Equity: ${eq:,.2f}", pc),
            (f"P&L: ${pnl:+,.2f} ({pnl / INITIAL_CASH * 100:+.1f}%)", pc),
        ]
        
        for text, color in stats:
            p.setPen(color)
            p.drawText(int(INFO_RECT.x() + 12), int(y + 14), text)
            y += 22
        
        y += 4
        p.setPen(QPen(GRID, 1))
        p.drawLine(
            int(INFO_RECT.x() + 12), int(y),
            int(INFO_RECT.x() + INFO_RECT.width() - 12), int(y)
        )
        y += 12
        
        # Detail stats
        p.setFont(FONTS['small'])
        details = [
            (f"Cash: ${ag.cash:,.2f}", TEXT),
            (f"Shares: {ag.shares:+d}", TEXT),
            (f"Trades: {ag.total_trades}", TEXT),
            (f"Position Value: ${abs(ag.shares) * self.sim.market.price:,.2f}", DIM),
        ]
        
        for text, color in details:
            p.setPen(color)
            p.drawText(int(INFO_RECT.x() + 12), int(y + 12), text)
            y += 18
        
        # Learning agent specific info
        y += 8
        p.setPen(QPen(GRID, 1))
        p.drawLine(
            int(INFO_RECT.x() + 12), int(y),
            int(INFO_RECT.x() + INFO_RECT.width() - 12), int(y)
        )
        y += 12
        
        p.setFont(FONTS['small'])
        p.setPen(YELLOW)
        p.drawText(int(INFO_RECT.x() + 12), int(y + 12), "Agent Parameters:")
        y += 18
        
        # Extract agent-specific parameters
        params = self._get_agent_params(ag)
        p.setFont(FONTS['tiny'])
        for param_name, param_val in params:
            p.setPen(DIM)
            p.drawText(int(INFO_RECT.x() + 16), int(y + 11), f"{param_name}:")
            p.setPen(TEXT)
            p.drawText(int(INFO_RECT.x() + 120), int(y + 11), str(param_val))
            y += 15
        
        # Generation history
        y += 8
        p.setPen(QPen(GRID, 1))
        p.drawLine(
            int(INFO_RECT.x() + 12), int(y),
            int(INFO_RECT.x() + INFO_RECT.width() - 12), int(y)
        )
        y += 12
        
        p.setFont(FONTS['small'])
        p.setPen(YELLOW)
        p.drawText(int(INFO_RECT.x() + 12), int(y + 12), "Generation History:")
        y += 18
        
        scores = self.sim.gscores[ag.name][-8:]
        if scores:
            p.setFont(FONTS['mini'])
            # Mini bar chart
            max_s = max(scores)
            min_s = min(scores)
            rng = max(max_s - min_s, 1)
            
            bar_width = min(55, (INFO_RECT.width() - 40) / len(scores))
            bar_max_height = 50
            
            for i, s in enumerate(scores):
                bar_h = max(2, ((s - min_s) / rng) * bar_max_height)
                bx = INFO_RECT.x() + 15 + i * bar_width
                by = y + bar_max_height - bar_h + 5
                
                color = GREEN if s >= INITIAL_CASH else RED
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(color.red(), color.green(), color.blue(), 150))
                p.drawRoundedRect(QRectF(bx, by, bar_width - 3, bar_h), 2, 2)
                
                p.setPen(DIM)
                p.drawText(int(bx), int(y + bar_max_height + 16), str(len(self.sim.gscores[ag.name]) - len(scores) + i + 1))
    
    def _get_agent_params(self, agent) -> list:
        """Extract agent-specific parameters for display."""
        params = []
        
        if hasattr(agent, 'eps'):
            params.append(("Epsilon", f"{agent.eps:.3f}"))
        if hasattr(agent, 'lr'):
            params.append(("Learning Rate", f"{agent.lr:.4f}"))
        if hasattr(agent, 'gamma'):
            params.append(("Gamma", f"{agent.gamma:.2f}"))
        if hasattr(agent, 'mcts_sims'):
            params.append(("MCTS Sims", agent.mcts_sims))
        if hasattr(agent, 'mcts_hor'):
            params.append(("MCTS Horizon", agent.mcts_hor))
        if hasattr(agent, 'buffer'):
            params.append(("Buffer Size", len(agent.buffer)))
        
        # Strategy-specific
        if hasattr(agent, '_st'):
            params.append(("Turtle State", ['Exit', 'Long', 'Short'][agent._st + 1]))
        if hasattr(agent, '_is_long'):
            params.append(("SAR Direction", "Long" if agent._is_long else "Short"))
        if hasattr(agent, '_was_squeezed'):
            params.append(("Squeezed", "Yes" if agent._was_squeezed else "No"))
        
        if not params:
            params.append(("Type", "Fixed Strategy"))
        
        return params
    
    def _draw_controls(self, p: QPainter):
        y = HEIGHT - 32
        p.fillRect(0, y, WIDTH, 32, QColor(8, 8, 16, 230))
        
        # Separator line
        p.setPen(GRID)
        p.drawLine(0, int(y), WIDTH, int(y))
        
        controls = [
            ("[SPACE]", "Pause"),
            ("[↑/↓]", "Speed"),
            ("[←/→]", "Select"),
            ("[R]", "Reset"),
            ("[G]", "Skip Gen"),
            ("[ESC]", "Quit"),
        ]
        
        p.setFont(FONTS['tiny'])
        x = 15
        for key, desc in controls:
            p.setPen(YELLOW)
            p.drawText(int(x), int(y + 20), key)
            x += p.fontMetrics().horizontalAdvance(key) + 4
            p.setPen(DIM)
            p.drawText(int(x), int(y + 20), desc)
            x += p.fontMetrics().horizontalAdvance(desc) + 20
        
        if self.sim.paused:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(YELLOW.red(), YELLOW.green(), YELLOW.blue(), 40))
            p.drawRoundedRect(QRectF(WIDTH - 120, y + 4, 110, 24), 4, 4)
            p.setPen(YELLOW)
            p.setFont(FONTS['medium'])
            p.drawText(WIDTH - 108, y + 22, "⏸ PAUSED")


def main():
    app = QApplication(sys.argv)
    window = SimulationWidget()
    window.setWindowTitle("Ultimate Stock Trading Sim - 21 Competing Algorithms")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()