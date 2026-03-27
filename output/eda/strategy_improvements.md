# Strategy Improvement Roadmap: Regime-Aware Mean Reversion v3

## Executive Summary

18 concrete, actionable improvements organized by category and prioritized by expected risk-adjusted return impact versus implementation complexity. Each idea is grounded in a specific code-level observation from the current V3 implementation.

**Prioritization framework:** Ideas ranked by the ratio of expected impact to implementation complexity, with an explicit penalty for overfitting risk.

---

## 1. Signal Improvements

### 1.1 Adaptive Z-Score Normalization Window

**What:** Replace the fixed `zscore_window=30` (`config.py:16-17`) with a window that adapts to the current OU half-life. Use `z_window = clamp(halflife * k, 15, 90)` where `k` is a single tunable scalar (e.g., 2.0).

**Why:** A 30-day window oversmooths fast-reverting regimes (hl=5d) and undersmooths slow ones (hl=40d). Linking the Z-window to half-life aligns the signal horizon with the actual reversion speed.

**Modify:** `generate_signals()` in `strategy/signals.py:94` to accept a rolling half-life series and compute per-bar adaptive windows.

- **Complexity:** Low
- **Impact:** Medium (0.05-0.15 Sharpe)
- **Caveat:** `pd.rolling` does not natively support variable windows. Adds one parameter (`k`) — search in walk-forward grid.

### 1.2 Bollinger Band Width / Volatility Squeeze Filter

**What:** Add BBW as a pre-entry gate. Only allow entries when `bbw < bbw_ema * 0.85`. During volatility expansions, extreme Z-scores often represent trend continuation, not reversion.

**Why:** The current strategy enters on Z-score magnitude alone (`strategy/signals.py:128-150`). A squeeze filter suppresses entries during breakouts, improving win rate on the moderate L0 entries (Z=1.8) where signal is weakest.

- **Complexity:** Low (~10 lines)
- **Impact:** Medium
- **Caveat:** May reduce trade frequency. Monitor that total trades/year stays above 15.

### 1.3 Volume-Weighted Signal Strength

**What:** Weight `signal_strength` (`signals.py:116`, currently `|Z| * regime_score`) by relative volume. High-volume reversals get priority for capital allocation in the portfolio ranking (`engine.py:298`).

**Why:** Volume data is already downloaded via yfinance but currently unused (`data/loader.py`). High-volume reversals have higher predictive power — they indicate genuine liquidity-driven mean reversion.

- **Complexity:** Low
- **Impact:** Medium
- **Caveat:** Crypto volume data from yfinance includes wash trading. Use relative volume (current vs own history) rather than absolute levels.

### 1.4 Dynamic Exit Threshold Based on Entry Z-Score

**What:** Replace fixed `z_exit=0.3` (`config.py:27`) with exits proportional to entry Z-score. Formula: `z_exit = entry_z * 0.15` (exit when Z reverts 85% toward zero).

**Why:** Strong entries (Z=-2.5) that revert to Z=-0.5 have captured significant alpha but wait unnecessarily. Weak entries (Z=-1.8) may not capture enough to cover costs. Per-trade adaptive exits improve capital recycling.

**Modify:** Exit logic in `engine.py:374-379` — currently asset-level, needs to become position-level. `entry_zscore` is already stored in the `Position` dataclass (`engine.py:29`).

- **Complexity:** Medium
- **Impact:** Medium
- **Caveat:** Makes backtesting logic more complex. Verify no lookahead bias in per-position exit check.

### 1.5 RSI Divergence for Entry Timing

**What:** When price makes a new low but RSI makes a higher low (bullish divergence), advance the entry by one level for sizing purposes. Pure timing refinement on top of existing signals.

- **Complexity:** Medium (30-50 lines)
- **Impact:** Low-Medium
- **Caveat:** Noisy on daily bars. Use minimum 5-bar lookback for extrema.

---

## 2. Regime Detection Enhancements

### 2.1 Hidden Markov Model for Regime Classification

**What:** Replace or augment the rolling OU regression (`strategy/regime.py:16-93`) with a 2-3 state HMM fitted to daily returns. States: mean-reverting, trending, high-volatility.

**Why:** HMMs provide forward-looking state probabilities that detect regime shifts 5-15 days earlier than the 60-day rolling OU window.

- **Complexity:** High (requires `hmmlearn`, careful initialization)
- **Impact:** High (0.1-0.3 Sharpe)
- **Caveat:** Notorious instability in parameter estimation. Use n_init=10 random restarts. Overfitting risk is significant.

### 2.2 Learned Regime Score Weights via Walk-Forward

**What:** Replace hardcoded regime weights (`config.py:43-45`: halflife=0.4, R²=0.3, pvalue=0.3) with weights optimized during each walk-forward training window. Add 2-3 weight presets to the grid, expanding from 9 to 27 combinations.

- **Complexity:** Low-Medium
- **Impact:** Medium
- **Caveat:** Monitor OOS degradation vs in-sample.

### 2.3 Hurst Exponent as Complementary Regime Signal

**What:** Add rolling Hurst exponent (H) via rescaled range analysis as a fourth input to the regime score. H < 0.5 = mean reverting.

**Why:** OU measures reversion speed; Hurst measures persistence structure. They are partially independent — combining them reduces false-positive regime classifications.

- **Complexity:** Medium (~40 lines + one more weight in `regime.py:134-140`)
- **Impact:** Medium
- **Caveat:** Hurst estimation is noisy on short windows. Consider DFA variant.

---

## 3. Risk Management Upgrades

### 3.1 Trailing Stop Based on Z-Score Reversion Progress

**What:** Supplement the fixed ATR stop (`engine.py:198-199`, 2.8x) with a trailing stop that activates once Z-score reverts past a midpoint. Track `best_price_since_entry` per position.

**Why:** Directly addresses the "winners turning into losers" problem. Successful reversions currently give back all gains before the stop triggers.

- **Complexity:** Medium (modify bar loop `engine.py:338-354`)
- **Impact:** High (improves Profit Factor and reduces Max DD)
- **Caveat:** Activation threshold is an additional parameter. Calibrate: activate after 50% reversion, trail at 0.5x ATR.

### 3.2 Dynamic Exposure Caps Based on Cross-Asset Regime Agreement

**What:** Modulate `max_total_exposure` (38%, `config.py:70`) based on how many assets simultaneously show strong mean-reverting regimes. More opportunities = higher cap.

- **Complexity:** Low (modify cap check in `_enter()`, `engine.py:168`)
- **Impact:** Medium
- **Caveat:** If all 5 assets show mean-reversion simultaneously, it may be a market-wide structural event — be cautious.

### 3.3 Per-Asset Volatility-Regime Position Sizing

**What:** When current ATR is in the bottom 25th percentile of its 252-day history, size up 20%; top 25th, size down 20%. Compute: `atr_percentile = atr.rolling(252).rank(pct=True)`.

- **Complexity:** Low
- **Impact:** Low-Medium
- **Caveat:** Overlaps with `trending_exposure_penalty` (`config.py:63`). Ensure combined scaling isn't too restrictive.

### 3.4 Portfolio-Level Risk Parity Allocation

**What:** Replace equal `risk_per_trade=0.04` (`config.py:68`) with inverse-volatility weighting targeting equal volatility contribution per asset.

- **Complexity:** Medium
- **Impact:** Medium
- **Caveat:** Risk parity can allocate heavily to low-vol assets that are trending, not reverting. Gate behind regime score.

---

## 4. Alternative Data Integration

### 4.1 Perpetual Futures Funding Rate

**What:** Integrate funding rates as a contrarian entry confirmation. Extreme negative funding (crowded shorts) boosts long entry sizing.

**Why:** Funding rates are a direct measure of market positioning orthogonal to price-based signals. Likely the single highest-impact alternative data source for this strategy.

- **Complexity:** Medium (Binance API or historical CSV, 8h → daily aggregation)
- **Impact:** High
- **Caveat:** Data may not cover full 5-year period for all assets. Spot-only strategy would need a separate data pipeline.

### 4.2 On-Chain Net Flow to Exchanges

**What:** Track net BTC/ETH/SOL flows to centralized exchanges. Large outflows = accumulation → long confirmation.

- **Complexity:** High (requires Glassnode/CryptoQuant, $40-200/month)
- **Impact:** Medium-High
- **Caveat:** Not available for BNB/XRP through standard providers.

### 4.3 Crypto Fear & Greed Index as Regime Overlay

**What:** Extreme fear boosts long regime score by 10-20%; extreme greed boosts short confidence. Free API with historical data, ~30 lines.

- **Complexity:** Low
- **Impact:** Low-Medium
- **Caveat:** Publicly available = alpha partially priced in. Use as a filter, not primary signal.

---

## 5. Execution & Infrastructure

### 5.1 Adaptive Slippage Model

**What:** Replace fixed 0.05% slippage with `slippage = base * (trade_size / median_volume)^0.5`. Load Volume data already available from yfinance.

- **Complexity:** Low-Medium
- **Impact:** Medium (primarily backtest accuracy)
- **Caveat:** Volume data quality is imperfect. Use conservative multiplier.

### 5.2 Limit Order Entry Modeling

**What:** Place limit orders at signal price instead of entering at next bar's open. Fill at `min(open, close_at_signal)` for longs.

- **Complexity:** Medium
- **Impact:** Low-Medium (saves 0.02-0.05% per trade)
- **Caveat:** Introduces non-fill risk. Use conservative fill assumptions.

---

## 6. ML/AI Opportunities

### 6.1 Gradient-Boosted Regime Classifier

**What:** Train XGBoost/LightGBM to predict "mean-reverting next 10 days" using OU stats, Hurst, ATR, RSI, trend strength. Replace handcrafted composite score.

- **Complexity:** High (walk-forward CV with strict temporal separation)
- **Impact:** High (0.15-0.30 Sharpe if properly validated)
- **Caveat:** Highest overfitting risk. Limit to 5-8 features, max_depth=3, require OOS AUC > 0.55.

### 6.2 Walk-Forward Parameter Selection via Bayesian Optimization

**What:** Replace the 9-combo grid search (`walkforward.py:45-46`) with Optuna. Enables searching continuous parameter space including regime weights, RSI thresholds, and stop multiples.

- **Complexity:** Medium (~40 lines of changes)
- **Impact:** Medium-High
- **Caveat:** Limit to 30-50 trials per fold. Use median of last 10 evaluations.

---

## Priority Matrix

| Rank | ID  | Improvement                     | Impact  | Complexity | Overfit Risk |
|------|-----|---------------------------------|---------|------------|--------------|
| 1    | 4.1 | Funding Rate Integration        | High    | Medium     | Low          |
| 2    | 3.1 | Trailing Z-Score Stop           | High    | Medium     | Low          |
| 3    | 1.2 | Bollinger Squeeze Filter        | Medium  | Low        | Low          |
| 4    | 1.1 | Adaptive Z-Window               | Medium  | Low        | Low          |
| 5    | 5.1 | Adaptive Slippage Model         | Medium  | Low        | None         |
| 6    | 1.3 | Volume-Weighted Signal Strength | Medium  | Low        | Low          |
| 7    | 3.2 | Dynamic Exposure Caps           | Medium  | Low        | Low          |
| 8    | 2.2 | Learned Regime Weights          | Medium  | Low-Med    | Medium       |
| 9    | 1.4 | Dynamic Exit Threshold          | Medium  | Medium     | Low          |
| 10   | 6.2 | Bayesian Optimization for WF    | Med-Hi  | Medium     | Medium       |
| 11   | 3.3 | Vol-Regime Position Sizing      | Low-Med | Low        | Low          |
| 12   | 2.3 | Hurst Exponent                  | Medium  | Medium     | Low          |
| 13   | 3.4 | Risk Parity Allocation          | Medium  | Medium     | Low          |
| 14   | 4.3 | Fear & Greed Index              | Low-Med | Low        | Low          |
| 15   | 5.2 | Limit Order Entry               | Low-Med | Medium     | None         |
| 16   | 1.5 | RSI Divergence                  | Low-Med | Medium     | Low          |
| 17   | 2.1 | HMM Regime Detection            | High    | High       | High         |
| 18   | 6.1 | Gradient-Boosted Regime         | High    | High       | High         |

---

## Recommended Implementation Phases

### Phase 1: Quick Wins (1-2 weeks)
Low-complexity, high-impact items that do not add overfitting risk:
- **5.1** Adaptive slippage model (improves backtest realism)
- **1.2** Bollinger squeeze filter (reduces bad entries)
- **1.3** Volume-weighted signal strength (better capital allocation)
- **3.2** Dynamic exposure caps (better capital utilization)

### Phase 2: Signal Refinement (2-4 weeks)
Improve entry/exit quality:
- **1.1** Adaptive Z-window linked to half-life
- **1.4** Dynamic exit thresholds per position
- **3.1** Trailing stop based on Z-score reversion progress
- **2.2** Walk-forward learned regime weights

### Phase 3: Alternative Data (4-6 weeks)
Add orthogonal data sources:
- **4.1** Funding rate integration (highest expected alpha)
- **4.3** Fear & Greed Index (low effort, modest alpha)
- **2.3** Hurst exponent (complementary regime signal)

### Phase 4: ML/Advanced (6-10 weeks)
Only after Phases 1-3 are validated OOS:
- **6.2** Bayesian optimization for walk-forward
- **6.1** Gradient-boosted regime classifier (with extreme caution)
- **3.4** Risk parity allocation

---

## Deployment Decision Rule

An improvement is deployed only if:
- OOS Sharpe improves by > 0.05
- IS-to-OOS Sharpe degradation < 30%
- Max Drawdown does not increase by > 3 percentage points
- Trade count remains in [20, 200] per year range
