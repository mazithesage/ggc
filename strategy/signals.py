"""
Signal generation v3: multi-level entries, RSI confirmation, signal strength scoring.

Signal flow:
  1. X(t) = log(Close)
  2. Z(t) = (X(t) - EMA(X, n)) / rolling_std(X, n)
  3. ATR for position sizing and stops
  4. RSI for secondary confirmation on weak signals
  5. Regime score from OU calibration (continuous scaling)
  6. Multi-level entry signals: Z thresholds × regime × RSI gating
  7. Signal strength = |Z| × regime_score (for portfolio ranking)
  8. Trend strength for regime-adaptive exposure
"""
import numpy as np
import pandas as pd
from config import StrategyConfig
from strategy.regime import compute_regime_stats


def compute_log_zscore(close: pd.Series, ema_span: int, std_window: int) -> pd.Series:
    """Z-score of log-price vs its EMA, normalized by rolling std."""
    log_price = np.log(close)
    ema = log_price.ewm(span=ema_span, min_periods=ema_span).mean()
    std = log_price.rolling(window=std_window, min_periods=std_window).std(ddof=1)
    return (log_price - ema) / std


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Average True Range via EMA."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=window, min_periods=window).mean()


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using exponential moving average."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=window, min_periods=window).mean()
    avg_loss = loss.ewm(span=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_trend_filter(close: pd.Series, ema_window: int) -> pd.Series:
    """True when price is above long-term EMA (bullish regime)."""
    ema = close.ewm(span=ema_window, min_periods=ema_window).mean()
    return close > ema


def compute_trend_strength(close: pd.Series, window: int) -> pd.Series:
    """
    Measure trend strength: ratio of directional move to total path.
    Near 0 = sideways, near 1 = strong trend.
    Used for regime-adaptive exposure.
    """
    net_move = (close - close.shift(window)).abs()
    total_path = close.diff().abs().rolling(window).sum()
    return (net_move / total_path.replace(0, 1e-10)).clip(0, 1)


def compute_momentum(close: pd.Series, window: int) -> pd.Series:
    """Simple return over lookback window."""
    return close.pct_change(periods=window)


def generate_signals(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    long_only: bool = False,
) -> pd.DataFrame:
    """
    Generate all signals and indicators for a single asset.

    Returns DataFrame with: close, open, zscore, atr, rsi, regime_score,
      trend_up, trend_strength, signal_strength, halflife,
      + per-level entry signals (entry_long_L0..L3, entry_short_L0..L3)
    """
    close = df["Close"].copy()
    log_price = np.log(close)

    signals = pd.DataFrame(index=df.index)
    signals["close"] = close
    signals["open"] = df["Open"].copy() if "Open" in df.columns else close
    signals["high"] = df["High"].copy() if "High" in df.columns else close
    signals["low"] = df["Low"].copy() if "Low" in df.columns else close

    # ── Z-Score ──
    signals["zscore"] = compute_log_zscore(close, cfg.zscore_window, cfg.zscore_std_window)

    # ── ATR ──
    signals["atr"] = compute_atr(signals["high"], signals["low"], close, cfg.atr_window)

    # ── RSI ──
    signals["rsi"] = compute_rsi(close, cfg.rsi_window)

    # ── Regime Score (continuous) ──
    regime = compute_regime_stats(log_price, cfg)
    signals["halflife"] = regime["halflife"]
    signals["halflife_raw"] = regime["halflife_raw"]
    signals["regime_score"] = regime["regime_score"]
    signals["ou_pvalue"] = regime["pvalue"]
    signals["ou_r2"] = regime["r2"]

    # ── Trend Filter & Strength ──
    signals["trend_up"] = compute_trend_filter(close, cfg.trend_ema_window)
    signals["momentum"] = compute_momentum(close, cfg.momentum_window)
    signals["trend_strength"] = compute_trend_strength(close, cfg.trend_strength_window)

    # ── Signal Strength (for portfolio-level ranking) ──
    signals["signal_strength"] = signals["zscore"].abs() * signals["regime_score"]

    # ── Regime Scale Factor (continuous position sizing multiplier) ──
    raw_scale = signals["regime_score"] / cfg.regime_scale_center
    signals["regime_scale"] = raw_scale.clip(cfg.regime_scale_min, cfg.regime_scale_max)
    # Hard floor: zero out below absolute minimum
    signals.loc[signals["regime_score"] < cfg.regime_score_threshold, "regime_scale"] = 0.0

    # ── Multi-Level Entry Signals ──
    z = signals["zscore"]
    rsi = signals["rsi"]

    for level_idx, (z_thresh, size_factor, requires_rsi) in enumerate(cfg.entry_levels):
        # Long entries: Z below -threshold
        long_z_ok = z < -z_thresh

        # Trend gating
        if cfg.use_trend_filter:
            long_trend_ok = signals["trend_up"] | (z < -(z_thresh + 0.5))
        else:
            long_trend_ok = True

        # RSI confirmation (only required for weak signals)
        long_rsi_ok = (rsi < cfg.rsi_oversold) if requires_rsi else True

        signals[f"entry_long_L{level_idx}"] = long_z_ok & long_trend_ok & long_rsi_ok

        # Short entries
        if long_only:
            signals[f"entry_short_L{level_idx}"] = False
        else:
            short_z_ok = z > z_thresh
            short_trend_ok = ~signals["trend_up"] if cfg.use_trend_filter else True
            short_rsi_ok = (rsi > cfg.rsi_overbought) if requires_rsi else True
            signals[f"entry_short_L{level_idx}"] = short_z_ok & short_trend_ok & short_rsi_ok

    # ── Exit Signals ──
    signals["raw_exit_long"] = z > -cfg.z_exit
    signals["raw_exit_short"] = z < cfg.z_exit

    # Store level metadata for the engine
    signals.attrs["entry_levels"] = cfg.entry_levels
    signals.attrs["n_levels"] = len(cfg.entry_levels)

    return signals
