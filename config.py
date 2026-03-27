"""
Regime-Aware Mean Reversion Strategy — Configuration v3
Capital-efficient multi-entry system with correlation-aware allocation.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StrategyConfig:
    # ── Assets ──
    assets: tuple = ("BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD")
    start_date: str = "2021-03-27"
    end_date: str = "2026-03-27"

    # ── Z-Score Signal (log-price + EMA) ──
    zscore_window: int = 30
    zscore_std_window: int = 30

    # ── Multi-Level Entry Thresholds (V3) ──
    # Each tuple: (z_threshold, size_factor, requires_rsi)
    # Lower Z = weaker signal → smaller position
    entry_levels: tuple = (
        (1.8, 0.65, True),    # Moderate: RSI required, 65% size
        (2.0, 1.00, False),   # Standard: full size
        (2.5, 1.20, False),   # Strong: overweight
    )
    z_exit: float = 0.3            # Tighter exit for faster turnover
    max_layers_per_asset: int = 3  # Max concurrent positions per asset

    # ── RSI Confirmation (V3) ──
    rsi_window: int = 14
    rsi_oversold: float = 40.0     # Long confirmation: RSI < this
    rsi_overbought: float = 60.0   # Short confirmation: RSI > this

    # ── OU Regime Detection ──
    ou_window: int = 60
    ou_halflife_min: float = 2.0
    ou_halflife_max: float = 50.0
    ou_pvalue_threshold: float = 0.05
    ou_halflife_ema_span: int = 10

    # ── Regime Score Weights ──
    regime_weight_halflife: float = 0.4
    regime_weight_r2: float = 0.3
    regime_weight_pvalue: float = 0.3

    # ── Continuous Regime Scaling (V3: replaces binary filter) ──
    # Position size multiplier = clamp(regime_score / regime_scale_center, min, max)
    regime_scale_min: float = 0.15      # Even weak regimes get small allocation
    regime_scale_center: float = 0.5    # At this score → 1.0x sizing
    regime_scale_max: float = 1.4       # Strong regimes get bonus
    regime_score_threshold: float = 0.1  # Hard floor (below this = no trade)

    # ── Trend Filter ──
    trend_ema_window: int = 200
    use_trend_filter: bool = True
    momentum_window: int = 30

    # ── Regime-Adaptive Exposure (V3) ──
    # In strong trends: reduce MR exposure; in sideways: increase
    trend_strength_window: int = 60    # Window to measure trend strength
    sideways_exposure_bonus: float = 1.3   # Multiply exposure cap in sideways
    trending_exposure_penalty: float = 0.6  # Reduce exposure cap in trends

    # ── Position Sizing (ATR-based) ──
    atr_window: int = 14
    atr_multiplier: float = 1.2       # Lower k → larger positions for crypto
    risk_per_trade: float = 0.04      # 4% equity risk per trade
    max_position_pct: float = 0.18    # Max 18% equity per layer
    max_total_exposure: float = 0.38  # Max 38% total (primary DD control lever)
    target_exposure: float = 0.30     # Target ~30% average deployment
    # Progressive drawdown circuit breaker (V3: very mild for MR compatibility)
    dd_circuit_breaker_levels: tuple = (
        (0.15, 0.50),   # DD > 15%: reduce sizing 50%
        (0.20, 0.0),    # DD > 20%: stop all new entries
    )
    # Portfolio-level emergency liquidation (V3)
    dd_liquidation_threshold: float = 1.0   # Disabled (exposure cap handles DD)
    dd_reentry_threshold: float = 0.08      # N/A when liquidation disabled

    # ── Correlation-Aware Allocation (V3) ──
    correlation_window: int = 60       # Rolling correlation lookback
    correlation_threshold: float = 0.90  # Only truly redundant pairs cluster
    max_cluster_exposure: float = 0.35   # Limit correlated cluster risk
    max_concurrent_positions: int = 15   # Portfolio-wide position limit

    # ── Stops ──
    stop_atr_multiple: float = 2.8     # Balanced: not too tight for crypto vol
    time_stop_halflife_multiple: float = 2.0

    # ── Execution ──
    fee_per_side: float = 0.0005
    slippage: float = 0.0005
    initial_capital: float = 100_000.0
    crypto_trading_days: int = 365

    # ── Walk-Forward ──
    wf_train_days: int = 252
    wf_test_days: int = 126
    wf_step_days: int = 126

    # ── Sensitivity Grid ──
    sensitivity_z_range: tuple = (1.5, 1.8, 2.0, 2.5)
    sensitivity_window_range: tuple = (20, 30, 40, 50, 60)

    # ── Output ──
    output_dir: str = "output"
    cache_dir: str = "data/cache"

    @property
    def total_cost_per_side(self) -> float:
        return self.fee_per_side + self.slippage


def get_v2_config() -> StrategyConfig:
    """Return V2-equivalent config for before/after comparison."""
    return StrategyConfig(
        entry_levels=((2.0, 1.0, False),),
        z_exit=0.5,
        max_layers_per_asset=1,
        regime_scale_min=0.0,
        regime_scale_center=0.5,
        regime_scale_max=1.0,
        regime_score_threshold=0.4,
        risk_per_trade=0.02,
        max_position_pct=0.20,
        max_total_exposure=0.50,
        target_exposure=0.30,
        stop_atr_multiple=3.0,
        atr_multiplier=2.0,
        max_concurrent_positions=5,
        correlation_threshold=1.0,  # Effectively disabled
        max_cluster_exposure=1.0,
        dd_liquidation_threshold=1.0,  # Effectively disabled for V2
    )
