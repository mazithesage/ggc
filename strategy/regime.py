"""
Regime detection via Ornstein-Uhlenbeck process calibration.

Fits rolling AR(1) regression on log-prices:
    ΔX(t) = a + b·X(t-1) + ε

Extracts: θ = -b, half-life = ln(2)/θ, p-value of b, R² of fit.
Builds a composite regime score ∈ [0, 1] for position-scaling.
"""
import numpy as np
import pandas as pd
from scipy import stats
from config import StrategyConfig


def _rolling_ou_regression(log_price: pd.Series, window: int) -> pd.DataFrame:
    """
    Rolling OLS: ΔX = a + b·X_lag + ε
    Returns DataFrame with columns: theta, halflife, pvalue, r2
    """
    delta = log_price.diff()
    n = len(log_price)

    theta = np.full(n, np.nan)
    halflife = np.full(n, np.nan)
    pvalue = np.full(n, np.nan)
    r2 = np.full(n, np.nan)

    for i in range(window, n):
        y = delta.iloc[i - window + 1 : i + 1].values
        x_lag = log_price.iloc[i - window : i].values

        mask = ~(np.isnan(y) | np.isnan(x_lag))
        if mask.sum() < max(10, window // 3):
            continue

        y_c = y[mask]
        x_c = x_lag[mask]
        n_obs = len(y_c)

        # OLS with constant: y = a + b*x
        X_mat = np.column_stack([np.ones(n_obs), x_c])
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X_mat, y_c, rcond=None)
        except np.linalg.LinAlgError:
            continue

        b = beta[1]

        # b must be negative for mean reversion
        if b >= 0:
            continue

        # Compute t-statistic and p-value for b
        y_hat = X_mat @ beta
        resid = y_c - y_hat
        ss_res = np.sum(resid ** 2)
        dof = n_obs - 2
        if dof <= 0:
            continue

        mse = ss_res / dof
        try:
            var_beta = mse * np.linalg.inv(X_mat.T @ X_mat)
        except np.linalg.LinAlgError:
            continue

        se_b = np.sqrt(max(var_beta[1, 1], 1e-20))
        t_stat = b / se_b
        p_val = 2 * stats.t.sf(abs(t_stat), dof)  # Two-sided

        # R²
        ss_tot = np.sum((y_c - y_c.mean()) ** 2)
        r2_val = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # θ and half-life (exact discrete→continuous mapping)
        theta_val = -np.log(1 + b) if (1 + b) > 0 else np.nan
        if theta_val is not None and theta_val > 0:
            hl_val = np.log(2) / theta_val
        else:
            continue

        theta[i] = theta_val
        halflife[i] = hl_val
        pvalue[i] = p_val
        r2[i] = r2_val

    return pd.DataFrame({
        "theta": theta,
        "halflife_raw": halflife,
        "pvalue": pvalue,
        "r2": r2,
    }, index=log_price.index)


def compute_regime_stats(log_price: pd.Series, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Compute all regime statistics and the composite regime score.
    Returns DataFrame with: halflife, halflife_smooth, pvalue, r2, regime_score
    """
    ou = _rolling_ou_regression(log_price, cfg.ou_window)

    # EMA-smooth the half-life for stability
    ou["halflife"] = (
        ou["halflife_raw"]
        .ewm(span=cfg.ou_halflife_ema_span, min_periods=1)
        .mean()
    )

    # ── Component scores (each ∈ [0, 1]) ──

    # 1. Half-life proximity: Gaussian centered at geometric mean of range
    hl = ou["halflife"]
    hl_center = np.sqrt(cfg.ou_halflife_min * cfg.ou_halflife_max)
    hl_sigma = (cfg.ou_halflife_max - cfg.ou_halflife_min) / 3
    score_hl = np.exp(-0.5 * ((hl - hl_center) / hl_sigma) ** 2)
    # Zero out if outside hard bounds
    score_hl = score_hl.where(
        (hl >= cfg.ou_halflife_min) & (hl <= cfg.ou_halflife_max), 0.0
    )

    # 2. R² score: direct mapping, clamp
    score_r2 = ou["r2"].clip(0, 1).fillna(0)

    # 3. p-value score: 1 when p < threshold, decays above
    pv = ou["pvalue"].fillna(1.0)
    score_pv = np.where(
        pv <= cfg.ou_pvalue_threshold,
        1.0,
        np.maximum(0, 1 - (pv - cfg.ou_pvalue_threshold) / (0.2 - cfg.ou_pvalue_threshold))
    )
    score_pv = np.clip(score_pv, 0, 1)

    # ── Composite regime score ──
    w_hl = cfg.regime_weight_halflife
    w_r2 = cfg.regime_weight_r2
    w_pv = cfg.regime_weight_pvalue
    w_total = w_hl + w_r2 + w_pv

    regime_score = (w_hl * score_hl + w_r2 * score_r2 + w_pv * score_pv) / w_total

    ou["score_halflife"] = score_hl
    ou["score_r2"] = score_r2
    ou["score_pvalue"] = score_pv
    ou["regime_score"] = regime_score

    return ou
