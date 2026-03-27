"""
Walk-forward validation: rolling train/test windows with parameter recalibration.
Updated for V3 multi-entry engine.
"""
import sys
import numpy as np
import pandas as pd
from config import StrategyConfig
from strategy.signals import generate_signals
from strategy.correlation import compute_rolling_correlations
from backtest.engine import BacktestEngine
from analysis.metrics import compute_metrics


def _run_quick_backtest(
    signals_by_asset: dict[str, pd.DataFrame],
    cfg: StrategyConfig,
) -> float:
    """Run backtest and return Sharpe ratio (fast evaluation for grid search)."""
    engine = BacktestEngine(cfg)
    eq = engine.run(signals_by_asset)
    if len(eq) < 20:
        return -999.0
    returns = eq["equity"].pct_change().dropna()
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(cfg.crypto_trading_days)


def walk_forward_test(
    data: dict[str, pd.DataFrame],
    base_cfg: StrategyConfig,
    long_only: bool = True,
    pair_correlations: dict = None,
) -> dict:
    """Run walk-forward validation with V3 engine."""
    all_dates = sorted(set().union(*(df.index for df in data.values())))
    n_dates = len(all_dates)

    train_n = base_cfg.wf_train_days
    test_n = base_cfg.wf_test_days
    step_n = base_cfg.wf_step_days

    # Parameter grid
    z_entries = [1.8, 2.0, 2.5]
    windows = [20, 30, 40]

    folds = []
    start = 0
    while start + train_n + test_n <= n_dates:
        train_start = all_dates[start]
        train_end = all_dates[start + train_n - 1]
        test_start = all_dates[start + train_n]
        test_end_idx = min(start + train_n + test_n - 1, n_dates - 1)
        test_end = all_dates[test_end_idx]
        folds.append({
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
        })
        start += step_n

    if not folds:
        return {"oos_equity": pd.DataFrame(), "oos_trades": pd.DataFrame(),
                "fold_results": [], "best_params_per_fold": []}

    print(f"  Walk-forward: {len(folds)} folds")

    all_oos_equity = []
    all_oos_trades = []
    fold_results = []
    best_params_list = []

    for fi, fold in enumerate(folds):
        sys.stdout.write(f"\r  Fold {fi+1}/{len(folds)}: "
                         f"train [{fold['train_start'].date()}-{fold['train_end'].date()}] "
                         f"test [{fold['test_start'].date()}-{fold['test_end'].date()}]  ")
        sys.stdout.flush()

        # ── TRAINING: grid search ──
        best_sharpe = -999
        best_z = 2.0
        best_w = 30

        for z_e in z_entries:
            for w in windows:
                train_cfg = StrategyConfig(
                    zscore_window=w, zscore_std_window=w,
                    entry_levels=((z_e, 1.0, False),),
                    z_exit=z_e / 4,
                    max_layers_per_asset=1,
                )
                train_signals = {}
                for ticker, df in data.items():
                    train_df = df.loc[fold["train_start"]:fold["train_end"]]
                    if len(train_df) < w + 60:
                        continue
                    train_signals[ticker] = generate_signals(train_df, train_cfg, long_only=long_only)

                if not train_signals:
                    continue

                sharpe = _run_quick_backtest(train_signals, train_cfg)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_z = z_e
                    best_w = w

        best_params_list.append({
            "fold": fi, "z_entry": best_z, "window": best_w,
            "train_sharpe": best_sharpe,
        })

        # ── TESTING: apply best params with V3 multi-entry ──
        test_cfg = StrategyConfig(
            zscore_window=best_w, zscore_std_window=best_w,
            entry_levels=(
                (best_z - 0.5, 0.3, True),
                (best_z - 0.2, 0.5, True),
                (best_z, 0.8, False),
                (best_z + 0.5, 1.0, False),
            ),
            z_exit=base_cfg.z_exit,
        )

        test_signals = {}
        for ticker, df in data.items():
            warmup_df = df.loc[:fold["test_end"]]
            if len(warmup_df) < best_w + 60:
                continue
            full_signals = generate_signals(warmup_df, test_cfg, long_only=long_only)
            test_signals[ticker] = full_signals.loc[fold["test_start"]:fold["test_end"]]

        if not test_signals:
            fold_results.append({"fold": fi, "sharpe": 0, "return": 0, "trades": 0})
            continue

        engine = BacktestEngine(test_cfg)
        oos_eq = engine.run(test_signals, pair_correlations)
        oos_trades = engine.get_trades_df()

        all_oos_equity.append(oos_eq)
        if len(oos_trades) > 0:
            oos_trades["fold"] = fi
            all_oos_trades.append(oos_trades)

        oos_metrics = compute_metrics(oos_eq, oos_trades, test_cfg)
        fold_results.append({
            "fold": fi, "sharpe": oos_metrics["Sharpe Ratio"],
            "return": oos_metrics["Total Return"],
            "trades": oos_metrics.get("Total Trades", 0),
            "best_z": best_z, "best_w": best_w,
        })

    print()

    combined_oos_eq = pd.DataFrame()
    if all_oos_equity:
        combined_oos_eq = pd.concat(all_oos_equity)
        combined_oos_eq = combined_oos_eq[~combined_oos_eq.index.duplicated(keep="last")]
        combined_oos_eq.sort_index(inplace=True)

    combined_oos_trades = pd.concat(all_oos_trades) if all_oos_trades else pd.DataFrame()

    return {
        "oos_equity": combined_oos_eq,
        "oos_trades": combined_oos_trades,
        "fold_results": fold_results,
        "best_params_per_fold": best_params_list,
    }
