"""
Event-driven backtesting engine v3: capital-efficient multi-entry system.

Key improvements over v2:
  - Multi-layer positions: up to N entries per asset at different Z-levels
  - Continuous regime scaling: position size ∝ regime_score
  - Portfolio-level signal ranking: allocate capital to top signals first
  - Correlation-aware entry gating
  - Regime-adaptive exposure targeting
  - Tighter stops for faster capital recycling
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from config import StrategyConfig
from strategy.correlation import (
    get_correlation_at_date, detect_clusters,
    get_cluster_for_ticker, compute_cluster_exposure,
)


@dataclass
class Position:
    direction: int
    entry_price: float
    size: float
    collateral: float
    entry_cost: float
    entry_date: pd.Timestamp
    entry_zscore: float
    stop_price: float
    halflife_at_entry: float
    regime_score_at_entry: float
    layer: int  # Which entry level triggered this


@dataclass
class Trade:
    ticker: str
    direction: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    bars_held: int
    exit_reason: str
    regime_score: float
    layer: int


class BacktestEngine:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.cash = cfg.initial_capital
        self.positions: dict[str, list[Position]] = {}  # ticker → list of layers
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []
        self.peak_equity = cfg.initial_capital
        self._risk_off = False  # Portfolio-level emergency liquidation state

    def _position_mtm(self, pos: Position, price: float) -> float:
        unrealized = pos.direction * pos.size * (price - pos.entry_price)
        return max(pos.collateral + unrealized, 0.0)

    def _total_position_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for ticker, layers in self.positions.items():
            price = prices.get(ticker, 0)
            for pos in layers:
                total += self._position_mtm(pos, price if price > 0 else pos.entry_price)
        return total

    def _equity(self, prices: dict[str, float]) -> float:
        return self.cash + self._total_position_value(prices)

    def _exposure_pct(self, prices: dict[str, float]) -> float:
        eq = self._equity(prices)
        return self._total_position_value(prices) / eq if eq > 0 else 1.0

    def _n_total_positions(self) -> int:
        return sum(len(layers) for layers in self.positions.values())

    def _current_drawdown(self, prices: dict[str, float]) -> float:
        eq = self._equity(prices)
        if eq >= self.peak_equity:
            return 0.0
        return (self.peak_equity - eq) / self.peak_equity

    def _ticker_exposure(self, ticker: str, prices: dict[str, float]) -> float:
        """Current exposure for a specific ticker (all layers)."""
        if ticker not in self.positions:
            return 0.0
        price = prices.get(ticker, 0)
        eq = self._equity(prices)
        if eq <= 0:
            return 0.0
        return sum(self._position_mtm(p, price) for p in self.positions[ticker]) / eq

    def _position_exposure_map(self, prices: dict[str, float]) -> dict[str, float]:
        """Map of ticker → exposure fraction."""
        eq = self._equity(prices)
        if eq <= 0:
            return {}
        result = {}
        for ticker, layers in self.positions.items():
            price = prices.get(ticker, 0)
            result[ticker] = sum(self._position_mtm(p, price) for p in layers) / eq
        return result

    def _compute_position_size(
        self, price: float, atr: float, regime_scale: float,
        level_size_factor: float, prices: dict[str, float],
        trend_strength: float = 0.5,
    ) -> float:
        equity = self._equity(prices)
        if equity <= 0 or atr <= 0:
            return 0.0

        # Base risk per trade
        risk_pct = self.cfg.risk_per_trade

        # Progressive drawdown circuit breaker (V3)
        dd = self._current_drawdown(prices)
        dd_multiplier = 1.0
        for dd_thresh, dd_scale in self.cfg.dd_circuit_breaker_levels:
            if dd > dd_thresh:
                dd_multiplier = dd_scale
        if dd_multiplier <= 0:
            return 0.0  # Hard stop: no new entries
        risk_pct *= dd_multiplier

        # Scale by continuous regime score (V3: no binary cutoff)
        risk_pct *= regime_scale

        # Scale by entry level strength
        risk_pct *= level_size_factor

        # Regime-adaptive: sideways markets → more aggressive
        if trend_strength < 0.3:
            risk_pct *= self.cfg.sideways_exposure_bonus
        elif trend_strength > 0.6:
            risk_pct *= self.cfg.trending_exposure_penalty

        # ATR-based sizing
        dollar_risk = risk_pct * equity
        size = dollar_risk / (atr * self.cfg.atr_multiplier)

        # Cap at max position pct per layer
        size = min(size, self.cfg.max_position_pct * equity / price)

        # Cap at available cash
        max_from_cash = self.cash / (price * (1 + self.cfg.total_cost_per_side))
        size = min(size, max_from_cash * 0.95)  # Keep 5% buffer

        return max(size, 0.0)

    def _enter(
        self, ticker: str, direction: int, price: float, atr: float,
        zscore: float, halflife: float, regime_score: float,
        regime_scale: float, level_idx: int, level_size_factor: float,
        trend_strength: float,
        date: pd.Timestamp, prices: dict[str, float],
    ) -> bool:
        # Check portfolio-level constraints
        if self._exposure_pct(prices) >= self.cfg.max_total_exposure:
            return False
        if self._n_total_positions() >= self.cfg.max_concurrent_positions:
            return False

        # Check per-asset layer limit
        existing = self.positions.get(ticker, [])
        if len(existing) >= self.cfg.max_layers_per_asset:
            return False

        # Check if this specific layer already exists
        existing_layers = {p.layer for p in existing}
        if level_idx in existing_layers:
            return False

        size = self._compute_position_size(
            price, atr, regime_scale, level_size_factor,
            prices, trend_strength,
        )
        if size <= 0:
            return False

        collateral = size * price
        entry_cost = collateral * self.cfg.total_cost_per_side
        if collateral + entry_cost > self.cash:
            return False

        self.cash -= (collateral + entry_cost)

        # ATR-based stop
        stop_distance = atr * self.cfg.stop_atr_multiple
        stop_price = price - stop_distance if direction == 1 else price + stop_distance

        pos = Position(
            direction=direction, entry_price=price, size=size,
            collateral=collateral, entry_cost=entry_cost,
            entry_date=date, entry_zscore=zscore,
            stop_price=stop_price,
            halflife_at_entry=halflife if not np.isnan(halflife) else 15.0,
            regime_score_at_entry=regime_score,
            layer=level_idx,
        )

        if ticker not in self.positions:
            self.positions[ticker] = []
        self.positions[ticker].append(pos)
        return True

    def _exit(self, ticker: str, pos: Position, price: float,
              date: pd.Timestamp, reason: str) -> Trade:
        gross_pnl = pos.direction * pos.size * (price - pos.entry_price)
        exit_cost = pos.size * price * self.cfg.total_cost_per_side
        net_pnl = gross_pnl - exit_cost - pos.entry_cost
        self.cash += pos.collateral + gross_pnl - exit_cost

        trade = Trade(
            ticker=ticker, direction=pos.direction,
            entry_date=pos.entry_date, exit_date=date,
            entry_price=pos.entry_price, exit_price=price,
            size=pos.size, pnl=net_pnl,
            return_pct=net_pnl / pos.collateral if pos.collateral > 0 else 0,
            bars_held=(date - pos.entry_date).days,
            exit_reason=reason,
            regime_score=pos.regime_score_at_entry,
            layer=pos.layer,
        )
        self.trades.append(trade)
        return trade

    def _exit_position_layer(self, ticker: str, pos: Position, price: float,
                             date: pd.Timestamp, reason: str):
        """Exit a specific layer and clean up."""
        self._exit(ticker, pos, price, date, reason)
        self.positions[ticker].remove(pos)
        if not self.positions[ticker]:
            del self.positions[ticker]

    def run(
        self,
        signals_by_asset: dict[str, pd.DataFrame],
        pair_correlations: dict = None,
    ) -> pd.DataFrame:
        """
        Run backtest with:
          - Next-bar execution
          - Multi-layer entries
          - Portfolio-level signal ranking
          - Correlation-aware allocation
        """
        cfg = self.cfg
        all_dates = sorted(set().union(*(df.index for df in signals_by_asset.values())))
        tickers = list(signals_by_asset.keys())

        # Pending orders for next-bar execution
        pending_entries: list[dict] = []
        pending_exits: list[tuple] = []  # (ticker, pos, reason)

        for bar_idx, date in enumerate(all_dates):
            current_prices = {}
            current_opens = {}
            for ticker, sdf in signals_by_asset.items():
                if date in sdf.index:
                    current_prices[ticker] = sdf.loc[date, "close"]
                    current_opens[ticker] = sdf.loc[date, "open"]

            # ── EXECUTE PENDING EXITS (at today's open) ──
            for ticker, pos, reason in pending_exits:
                if ticker in self.positions and pos in self.positions[ticker]:
                    if ticker in current_opens:
                        self._exit_position_layer(ticker, pos, current_opens[ticker], date, reason)
            pending_exits.clear()

            # ── PORTFOLIO-LEVEL EMERGENCY LIQUIDATION (V3) ──
            dd = self._current_drawdown(current_prices)
            if dd >= cfg.dd_liquidation_threshold and self.positions:
                # Risk-off: close ALL positions immediately
                self._risk_off = True
                for ticker in list(self.positions.keys()):
                    price = current_opens.get(ticker, current_prices.get(ticker, 0))
                    for pos in list(self.positions.get(ticker, [])):
                        self._exit_position_layer(ticker, pos, price, date, "emergency_liquidation")
                pending_entries.clear()
            elif self._risk_off and dd <= cfg.dd_reentry_threshold:
                # Recovery: allow re-entry once DD has cooled
                self._risk_off = False

            # ── EXECUTE PENDING ENTRIES (at today's open, ranked by strength) ──
            # Sort by signal strength descending (best signals first)
            if self._risk_off:
                pending_entries.clear()  # Suppress entries during risk-off
            pending_entries.sort(key=lambda x: x["strength"], reverse=True)
            for entry in pending_entries:
                ticker = entry["ticker"]
                if ticker not in current_opens:
                    continue
                # Re-check constraints at execution time
                if self._exposure_pct(current_prices) >= cfg.max_total_exposure:
                    break
                if self._n_total_positions() >= cfg.max_concurrent_positions:
                    break

                # Correlation check at execution time
                if pair_correlations:
                    existing_tickers = list(self.positions.keys())
                    max_corr = get_correlation_at_date(
                        pair_correlations, date, ticker,
                        existing_tickers, cfg.correlation_threshold,
                    )
                    if max_corr > cfg.correlation_threshold:
                        # Check cluster exposure cap
                        clusters = detect_clusters(
                            pair_correlations, date, tickers, cfg.correlation_threshold,
                        )
                        exp_map = self._position_exposure_map(current_prices)
                        cluster = get_cluster_for_ticker(clusters, ticker)
                        cluster_exp = sum(exp_map.get(t, 0) for t in cluster)
                        if cluster_exp >= cfg.max_cluster_exposure:
                            continue  # Skip, cluster is full

                self._enter(
                    ticker, entry["direction"], current_opens[ticker],
                    entry["atr"], entry["zscore"], entry["halflife"],
                    entry["regime_score"], entry["regime_scale"],
                    entry["level_idx"], entry["level_size_factor"],
                    entry["trend_strength"],
                    date, current_prices,
                )
            pending_entries.clear()

            # ── CHECK STOPS (using close as proxy) ──
            for ticker in list(self.positions.keys()):
                if ticker not in current_prices:
                    continue
                price = current_prices[ticker]
                for pos in list(self.positions.get(ticker, [])):
                    # Stop loss
                    if pos.direction == 1 and price <= pos.stop_price:
                        self._exit_position_layer(ticker, pos, price, date, "stop_loss")
                        continue
                    if pos.direction == -1 and price >= pos.stop_price:
                        self._exit_position_layer(ticker, pos, price, date, "stop_loss")
                        continue
                    # Time stop
                    bars_held = (date - pos.entry_date).days
                    if bars_held > pos.halflife_at_entry * cfg.time_stop_halflife_multiple:
                        self._exit_position_layer(ticker, pos, price, date, "time_stop")
                        continue

            # ── GENERATE SIGNALS FOR NEXT BAR ──
            for ticker, sdf in signals_by_asset.items():
                if date not in sdf.index:
                    continue
                row = sdf.loc[date]
                zscore = row.get("zscore", np.nan)
                if pd.isna(zscore):
                    continue

                atr = row.get("atr", np.nan)
                if pd.isna(atr) or atr <= 0:
                    continue

                regime_scale = row.get("regime_scale", 0.0)
                if regime_scale <= 0:
                    continue  # Below hard regime floor

                # ── Signal-based exits → queue for next bar ──
                if ticker in self.positions:
                    for pos in list(self.positions[ticker]):
                        if pos.direction == 1 and row.get("raw_exit_long", False):
                            pending_exits.append((ticker, pos, "signal_exit"))
                        elif pos.direction == -1 and row.get("raw_exit_short", False):
                            pending_exits.append((ticker, pos, "signal_exit"))

                # ── Multi-level entries → queue for next bar ──
                n_levels = len(cfg.entry_levels)
                existing_layers = set()
                if ticker in self.positions:
                    existing_layers = {p.layer for p in self.positions[ticker]}

                for level_idx in range(n_levels):
                    if level_idx in existing_layers:
                        continue

                    z_thresh, size_factor, _ = cfg.entry_levels[level_idx]
                    long_sig = row.get(f"entry_long_L{level_idx}", False)
                    short_sig = row.get(f"entry_short_L{level_idx}", False)

                    if long_sig:
                        pending_entries.append({
                            "ticker": ticker, "direction": 1,
                            "strength": abs(zscore) * row.get("regime_score", 0),
                            "level_idx": level_idx,
                            "level_size_factor": size_factor,
                            "atr": atr, "zscore": zscore,
                            "halflife": row.get("halflife", 15),
                            "regime_score": row.get("regime_score", 0),
                            "regime_scale": regime_scale,
                            "trend_strength": row.get("trend_strength", 0.5),
                        })
                    elif short_sig:
                        pending_entries.append({
                            "ticker": ticker, "direction": -1,
                            "strength": abs(zscore) * row.get("regime_score", 0),
                            "level_idx": level_idx,
                            "level_size_factor": size_factor,
                            "atr": atr, "zscore": zscore,
                            "halflife": row.get("halflife", 15),
                            "regime_score": row.get("regime_score", 0),
                            "regime_scale": regime_scale,
                            "trend_strength": row.get("trend_strength", 0.5),
                        })

            # ── RECORD EQUITY ──
            equity = self._equity(current_prices)
            self.peak_equity = max(self.peak_equity, equity)
            n_pos = self._n_total_positions()
            self.equity_curve.append({
                "date": date, "equity": equity, "cash": self.cash,
                "n_positions": n_pos,
                "exposure": self._exposure_pct(current_prices),
                "drawdown": self._current_drawdown(current_prices),
            })

        # Close remaining positions
        for ticker in list(self.positions.keys()):
            sdf = signals_by_asset[ticker]
            last_price = sdf.iloc[-1]["close"]
            last_date = sdf.index[-1]
            for pos in list(self.positions.get(ticker, [])):
                self._exit_position_layer(ticker, pos, last_price, last_date, "end_of_backtest")

        eq_df = pd.DataFrame(self.equity_curve).set_index("date")
        return eq_df

    def get_trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "ticker": t.ticker,
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "entry_date": t.entry_date, "exit_date": t.exit_date,
            "entry_price": t.entry_price, "exit_price": t.exit_price,
            "size": t.size, "pnl": t.pnl,
            "return_pct": t.return_pct, "bars_held": t.bars_held,
            "exit_reason": t.exit_reason, "regime_score": t.regime_score,
            "layer": t.layer,
        } for t in self.trades])
