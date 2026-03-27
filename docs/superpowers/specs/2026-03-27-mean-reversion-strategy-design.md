# Mean Reversion Trading Strategy — Design Spec

## Overview
A statistically grounded mean reversion strategy for crypto spot markets (BTC, ETH, SOL) combining Z-score signals with Ornstein-Uhlenbeck regime filtering.

## Data
- **Source:** yfinance (daily OHLCV)
- **Assets:** BTC-USD, ETH-USD, SOL-USD
- **Period:** 5 years (2021-03-27 to 2026-03-27)
- **Granularity:** Daily bars

## Signal: Z-Score
```
Z(t) = (Close(t) - SMA(t, n)) / RollingStd(t, n)
```
- Window n = 40 days
- Long entry: Z < -2.0 | Long exit: Z > -0.5
- Short entry: Z > +2.0 | Short exit: Z < +0.5

## Regime Filter: OU Half-Life
- Discretized OU: `ΔPrice = a + b * Price(t-1) + ε`, θ = -b, t½ = ln(2)/θ
- Rolling calibration: 60 days
- Trade only when half-life ∈ [2, 30] days

## Position Sizing (Fixed Fractional, Anti-Martingale)
- Risk 2% equity per trade
- Size = (0.02 * Equity) / (Entry * StopDistance%)
- Max 20% per asset, 50% total exposure
- Never increase position on losses

## Risk Management
- Stop-loss: 3σ from entry (~5-8%)
- Time stop: 2× half-life without reversion
- Take-profit: Z crosses ±0.5

## Transaction Costs
- Fee: 0.05% per side (0.1% round-trip)
- Slippage: 0.05% per trade
- Total friction: ~0.15% round-trip

## Backtester
- Event-driven, bar-by-bar processing
- No lookahead bias (signals from data at time t only)
- Tracks: cash, positions, equity curve, trade log

## Metrics
Sharpe, Sortino, Max Drawdown, CAGR, Win Rate, Profit Factor, Avg Duration, Trades/Year

## Bonus
- Parameter sensitivity (Z-threshold × window heatmap)
- Equity curve visualization
- Regime overlay plots
- Comparative cross-asset analysis

## Project Structure
```
├── data/loader.py        # yfinance download + caching
├── strategy/signals.py   # Z-score, OU calibration
├── backtest/engine.py    # Event-driven backtester
├── analysis/metrics.py   # Performance computation
├── analysis/plots.py     # Visualization
├── config.py             # All parameters
├── main.py               # Entry point
└── requirements.txt
```
