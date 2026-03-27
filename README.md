# ggc

Regime-aware mean reversion strategy for crypto spot markets. Backtested across BTC, ETH, SOL, BNB, and XRP over a 5-year period (2021–2026).

The system fits an Ornstein-Uhlenbeck process to log-prices on a rolling basis, extracts a composite regime score (half-life, R², p-value), and uses it to continuously scale position sizes. Entries are triggered by Z-score deviations from an EMA-smoothed mean, with multi-level thresholds that allow pyramiding into stronger signals. Correlation clustering prevents redundant exposure across assets that move together.

Two strategy versions are included — V2 (single-entry baseline) and V3 (multi-layer, capital-efficient) — run side-by-side for comparison.

## Project structure

```
config.py               Strategy parameters (frozen dataclass)
main.py                 Entry point — runs V2, V3, walk-forward, comparison

data/
  loader.py             Downloads OHLCV via yfinance, caches to CSV
  cache/                Cached daily bars per asset

strategy/
  signals.py            Z-score computation, RSI, trend filter, entry/exit logic
  regime.py             OU process calibration, regime score composition
  correlation.py        Rolling pairwise correlations, union-find clustering

backtest/
  engine.py             Bar-by-bar event-driven backtester
  walkforward.py        Rolling train/test with parameter grid search

analysis/
  metrics.py            Sharpe, Sortino, Calmar, drawdown, profit factor, etc.
  plots.py              Equity curves, Z-score overlays, exposure timelines

output/                 Generated CSVs, PNGs, reports
```

## How it works

### Signal generation

Log-prices are normalized into a Z-score relative to a rolling EMA:

```
Z(t) = (ln(Close) - EMA(ln(Close), 30)) / StdDev(ln(Close), 30)
```

Three entry levels fire at different Z thresholds (long side shown):

| Level | Z threshold | Size factor | RSI required |
|-------|-------------|-------------|--------------|
| Moderate | -1.8 | 0.65x | Yes (< 40) |
| Standard | -2.0 | 1.00x | No |
| Strong | -2.5 | 1.20x | No |

Up to 3 layers can be open per asset simultaneously. A 200-day EMA trend filter is applied (waived for extreme Z-scores).

Exits trigger when Z crosses back through -0.3, or via ATR-based stop-loss (2.8x), or time stop (2x estimated half-life).

### Regime detection

A rolling AR(1) regression calibrates an OU process on log-prices over a 60-day window:

```
dX(t) = a + b * X(t-1) + e
theta = -ln(1 + b)
half_life = ln(2) / theta
```

The regime score is a weighted composite:

- Half-life quality (40%) — Gaussian centered at ~10 days, bounds [2, 50]
- R² (30%) — goodness of fit
- P-value (30%) — statistical confidence

Position sizing scales continuously with regime score: 0.15x at the floor, 1.4x at the ceiling. Below a score of 0.1, no new entries are taken.

### Risk management

- ATR-based position sizing: `size = (risk_pct * equity) / (ATR * 1.2)`
- Max 18% equity per layer, 38% total portfolio exposure
- Drawdown circuit breaker: 50% sizing reduction at 15% DD, full halt at 20%
- Correlation cluster cap: 35% max exposure to assets with rolling correlation > 0.90
- Max 15 concurrent positions across all assets

### Walk-forward validation

Rolling 252-day train / 126-day test windows, stepping 126 days at a time. Grid search over Z-thresholds and lookback windows on the training set (maximize Sharpe), then deploy out-of-sample on the test set.

## V2 vs V3

| | V2 | V3 |
|---|---|---|
| Entry levels | 1 | 3 (multi-layer) |
| Regime filter | Binary (threshold = 0.4) | Continuous (0.15x–1.4x) |
| Correlation | Disabled | Active clustering |
| RSI confirmation | None | Weak signals only |
| Risk per trade | 2% | 4% |
| Max exposure | 50% | 38% |
| Stop distance | 3.0x ATR | 2.8x ATR |

V3 trades more frequently with tighter exits, aiming for higher capital utilization and better risk-adjusted returns through regime-adaptive sizing.

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+.

## Usage

```
python main.py
```

This runs the full pipeline:
1. Downloads/loads cached OHLCV data
2. Runs V2 baseline backtest
3. Runs V3 multi-entry backtest
4. Prints V2 vs V3 comparison table
5. Runs walk-forward validation on V3
6. Generates all charts and saves trade logs to `output/`

## Configuration

All parameters live in `config.py` as a frozen dataclass. To experiment, modify the defaults in `StrategyConfig` or create a new config:

```python
from config import StrategyConfig

cfg = StrategyConfig(
    risk_per_trade=0.03,
    entry_levels=(
        (1.5, 0.50, True),
        (2.0, 1.00, False),
        (3.0, 1.50, False),
    ),
    max_total_exposure=0.45,
)
```

`get_v2_config()` returns the V2-equivalent settings for comparison runs.

## Dependencies

- yfinance — OHLCV data
- pandas, numpy — data handling
- scipy, statsmodels — OU calibration, statistical tests
- matplotlib, seaborn — visualization

## Notes

- Data is cached in `data/cache/` after the first download. Delete the cache to force a refresh.
- The backtest executes signals at next-bar open to avoid lookahead bias.
- Slippage and fees are modeled at 0.05% per side (configurable).
- Crypto markets trade 365 days/year — annualization uses 365, not 252.
- Short signals are generated but the default run uses `long_only=True`.
