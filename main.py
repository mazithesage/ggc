"""
Regime-Aware Mean Reversion Strategy v3 — Main Entry Point

Before/After comparison: V2 (low exposure) vs V3 (capital-efficient multi-entry)

Phases:
  1. Load data
  2. Run V2 baseline (for comparison)
  3. Run V3 capital-efficient system
  4. Walk-forward validation (V3)
  5. Before vs After comparison
  6. Visualizations
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from config import StrategyConfig, get_v2_config
from data.loader import load_all
from strategy.signals import generate_signals
from strategy.correlation import compute_rolling_correlations
from backtest.engine import BacktestEngine
from backtest.walkforward import walk_forward_test
from analysis.metrics import compute_metrics, format_metrics


def run_backtest(signals: dict, cfg: StrategyConfig, pair_corr=None):
    engine = BacktestEngine(cfg)
    eq = engine.run(signals, pair_corr)
    trades = engine.get_trades_df()
    return eq, trades


def run_per_asset(all_signals: dict, cfg: StrategyConfig, pair_corr=None) -> dict:
    results = {}
    for ticker, sdf in all_signals.items():
        eq, trades = run_backtest({ticker: sdf}, cfg, pair_corr)
        results[ticker] = {
            "equity": eq, "trades": trades,
            "metrics": compute_metrics(eq, trades, cfg),
        }
    return results


def print_comparison_table(v2_m: dict, v3_m: dict):
    """Print side-by-side V2 vs V3 comparison."""
    w = 62
    lines = [
        "=" * w,
        "  BEFORE (V2) vs AFTER (V3) COMPARISON",
        "=" * w,
        f"  {'Metric':<24s}  {'V2 (Before)':>14s}  {'V3 (After)':>14s}",
        "-" * w,
    ]
    rows = [
        ("CAGR", ".2%"), ("Sharpe Ratio", ".3f"), ("Sortino Ratio", ".3f"),
        ("Calmar Ratio", ".3f"), ("Max Drawdown", ".2%"),
        ("Total Trades", "d"), ("Win Rate", ".2%"),
        ("Profit Factor", ".3f"), ("Avg Exposure", ".2%"),
        ("Exposure-Adj Return", ".2%"),
    ]
    for name, fmt in rows:
        v2_val = v2_m.get(name, 0)
        v3_val = v3_m.get(name, 0)
        if fmt == "d":
            lines.append(f"  {name:<24s}  {int(v2_val):>14d}  {int(v3_val):>14d}")
        else:
            lines.append(f"  {name:<24s}  {v2_val:>14{fmt}}  {v3_val:>14{fmt}}")

    # Derived metrics
    v2_trades_yr = v2_m.get("Trades/Year", 0)
    v3_trades_yr = v3_m.get("Trades/Year", 0)
    lines.append(f"  {'Trades/Year':<24s}  {v2_trades_yr:>14.1f}  {v3_trades_yr:>14.1f}")

    lines.append("=" * w)

    # Improvement summary
    lines.append("")
    lines.append("  IMPROVEMENTS:")
    sharpe_delta = v3_m.get("Sharpe Ratio", 0) - v2_m.get("Sharpe Ratio", 0)
    lines.append(f"    Sharpe: {sharpe_delta:+.3f}")
    exp_delta = v3_m.get("Avg Exposure", 0) - v2_m.get("Avg Exposure", 0)
    lines.append(f"    Exposure: {exp_delta:+.1%}")
    trade_delta = v3_m.get("Total Trades", 0) - v2_m.get("Total Trades", 0)
    lines.append(f"    Trade count: {trade_delta:+d}")
    cagr_delta = v3_m.get("CAGR", 0) - v2_m.get("CAGR", 0)
    lines.append(f"    CAGR: {cagr_delta:+.2%}")

    return "\n".join(lines)


def per_asset_comparison(v2_per: dict, v3_per: dict):
    """Print per-asset comparison."""
    tickers = sorted(set(list(v2_per.keys()) + list(v3_per.keys())))
    w = 76
    lines = [
        "=" * w,
        "  PER-ASSET: V2 vs V3",
        "=" * w,
        f"  {'Asset':<10s} {'V2 Sharpe':>10s} {'V3 Sharpe':>10s} {'V2 Trades':>10s} {'V3 Trades':>10s} {'V2 Exp':>10s} {'V3 Exp':>10s}",
        "-" * w,
    ]
    for t in tickers:
        m2 = v2_per.get(t, {}).get("metrics", {})
        m3 = v3_per.get(t, {}).get("metrics", {})
        lines.append(
            f"  {t:<10s} {m2.get('Sharpe Ratio', 0):>10.3f} {m3.get('Sharpe Ratio', 0):>10.3f}"
            f" {m2.get('Total Trades', 0):>10d} {m3.get('Total Trades', 0):>10d}"
            f" {m2.get('Avg Exposure', 0):>10.1%} {m3.get('Avg Exposure', 0):>10.1%}"
        )
    lines.append("=" * w)
    return "\n".join(lines)


# ═══════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════

def plot_equity_comparison(v2_eq, v3_eq, wf_eq, cfg):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [2, 1, 1]})

    ax = axes[0]
    ax.plot(v2_eq.index, v2_eq["equity"], color="#f85149", lw=1.2, label="V2 (Before)", alpha=0.8)
    ax.plot(v3_eq.index, v3_eq["equity"], color="#58a6ff", lw=1.8, label="V3 (After)")
    if len(wf_eq) > 0:
        ax.plot(wf_eq.index, wf_eq["equity"], color="#d29922", lw=1.2, label="V3 Walk-Forward OOS", alpha=0.8)
    ax.axhline(y=cfg.initial_capital, color="gray", ls="--", alpha=0.5, label="$100K")
    ax.set_title("Equity Curves: V2 vs V3", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Drawdown comparison
    ax2 = axes[1]
    dd_v2 = (v2_eq["equity"] - v2_eq["equity"].cummax()) / v2_eq["equity"].cummax() * 100
    dd_v3 = (v3_eq["equity"] - v3_eq["equity"].cummax()) / v3_eq["equity"].cummax() * 100
    ax2.fill_between(v2_eq.index, dd_v2, 0, color="#f85149", alpha=0.2, label="V2 DD")
    ax2.fill_between(v3_eq.index, dd_v3, 0, color="#58a6ff", alpha=0.3, label="V3 DD")
    ax2.set_ylabel("Drawdown %")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Exposure comparison
    ax3 = axes[2]
    ax3.fill_between(v2_eq.index, v2_eq["exposure"] * 100, 0, color="#f85149", alpha=0.2, label="V2 Exposure")
    ax3.fill_between(v3_eq.index, v3_eq["exposure"] * 100, 0, color="#58a6ff", alpha=0.3, label="V3 Exposure")
    ax3.axhline(y=cfg.target_exposure * 100, color="#3fb950", ls="--", alpha=0.7,
                label=f"Target ({cfg.target_exposure:.0%})")
    ax3.set_ylabel("Exposure %")
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(cfg.output_dir, "v2_vs_v3_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_trade_distribution(v2_trades, v3_trades, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Trade count by asset
    ax = axes[0, 0]
    if len(v2_trades) > 0 and len(v3_trades) > 0:
        v2_counts = v2_trades["ticker"].value_counts().sort_index()
        v3_counts = v3_trades["ticker"].value_counts().sort_index()
        all_tickers = sorted(set(v2_counts.index) | set(v3_counts.index))
        x = np.arange(len(all_tickers))
        ax.bar(x - 0.2, [v2_counts.get(t, 0) for t in all_tickers], 0.4, label="V2", color="#f85149", alpha=0.7)
        ax.bar(x + 0.2, [v3_counts.get(t, 0) for t in all_tickers], 0.4, label="V3", color="#58a6ff", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(all_tickers, rotation=45, fontsize=9)
    ax.set_title("Trade Count by Asset", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Return distribution
    ax = axes[0, 1]
    if len(v2_trades) > 0:
        ax.hist(v2_trades["return_pct"] * 100, bins=30, alpha=0.5, color="#f85149", label="V2")
    if len(v3_trades) > 0:
        ax.hist(v3_trades["return_pct"] * 100, bins=30, alpha=0.5, color="#58a6ff", label="V3")
    ax.set_title("Return Distribution (%)", fontweight="bold")
    ax.set_xlabel("Return %")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bars held distribution
    ax = axes[1, 0]
    if len(v2_trades) > 0:
        ax.hist(v2_trades["bars_held"], bins=20, alpha=0.5, color="#f85149", label="V2")
    if len(v3_trades) > 0:
        ax.hist(v3_trades["bars_held"], bins=20, alpha=0.5, color="#58a6ff", label="V3")
    ax.set_title("Holding Period (Days)", fontweight="bold")
    ax.set_xlabel("Days Held")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Trade PnL by layer (V3 only)
    ax = axes[1, 1]
    if len(v3_trades) > 0 and "layer" in v3_trades.columns:
        layer_pnl = v3_trades.groupby("layer")["pnl"].agg(["sum", "count", "mean"])
        colors = ["#3fb950" if s > 0 else "#f85149" for s in layer_pnl["sum"]]
        ax.bar(layer_pnl.index, layer_pnl["sum"], color=colors, alpha=0.7)
        for i, row in layer_pnl.iterrows():
            ax.text(i, row["sum"], f"n={int(row['count'])}", ha="center", va="bottom" if row["sum"] > 0 else "top", fontsize=9)
    ax.set_title("V3: Total PnL by Entry Layer", fontweight="bold")
    ax.set_xlabel("Entry Layer (0=weakest, 3=strongest)")
    ax.set_ylabel("Total PnL ($)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(cfg.output_dir, "trade_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_asset_detail(signals_df, trades_df, ticker, cfg):
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    ax = axes[0]
    ax.plot(signals_df.index, signals_df["close"], "k-", lw=0.8)
    tt = trades_df[trades_df["ticker"] == ticker] if len(trades_df) > 0 else pd.DataFrame()
    for _, t in tt.iterrows():
        c = "green" if t["pnl"] > 0 else "red"
        ax.annotate("", xy=(t["exit_date"], t["exit_price"]),
                     xytext=(t["entry_date"], t["entry_price"]),
                     arrowprops=dict(arrowstyle="->", color=c, alpha=0.6))
    ax.set_title(f"{ticker} — Price & Trades (V3)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(signals_df.index, signals_df["zscore"], "b-", lw=0.7)
    for z_thresh, _, _ in cfg.entry_levels:
        ax2.axhline(-z_thresh, color="green", ls="--", alpha=0.4, lw=0.5)
        ax2.axhline(z_thresh, color="red", ls="--", alpha=0.4, lw=0.5)
    ax2.axhline(0, color="gray", ls="-", alpha=0.3)
    ax2.set_ylabel("Z-Score")
    ax2.set_ylim(-4, 4)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.fill_between(signals_df.index, signals_df["regime_score"], 0, color="purple", alpha=0.3, label="Regime Score")
    ax3.axhline(cfg.regime_score_threshold, color="red", ls=":", alpha=0.7,
                label=f"Floor ({cfg.regime_score_threshold})")
    ax3.set_ylabel("Regime Score")
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[3]
    ax4.plot(signals_df.index, signals_df["rsi"], "orange", lw=0.7, label="RSI")
    ax4.axhline(cfg.rsi_oversold, color="green", ls=":", alpha=0.7, label=f"Oversold ({cfg.rsi_oversold})")
    ax4.axhline(cfg.rsi_overbought, color="red", ls=":", alpha=0.7, label=f"Overbought ({cfg.rsi_overbought})")
    ax4.set_ylabel("RSI")
    ax4.set_ylim(0, 100)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    path = os.path.join(cfg.output_dir, f"v3_detail_{ticker.replace('-', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_wf_folds(fold_results, cfg):
    if not fold_results:
        return None
    df = pd.DataFrame(fold_results)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3fb950" if s > 0 else "#f85149" for s in df["sharpe"]]
    ax.bar(df["fold"], df["sharpe"], color=colors, alpha=0.8, edgecolor="white", lw=0.5)
    ax.axhline(0, color="gray", ls="-", alpha=0.5)
    ax.axhline(df["sharpe"].mean(), color="blue", ls="--", alpha=0.7,
               label=f"Mean OOS Sharpe: {df['sharpe'].mean():.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("OOS Sharpe Ratio")
    ax.set_title("V3 Walk-Forward: OOS Sharpe by Fold", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(cfg.output_dir, "v3_wf_fold_sharpes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_exposure_timeline(v2_eq, v3_eq, cfg):
    """Detailed exposure over time comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.fill_between(v3_eq.index, v3_eq["exposure"] * 100, 0, color="#58a6ff", alpha=0.4, label="V3 Exposure")
    ax.plot(v3_eq.index, v3_eq["n_positions"], color="#d2a8ff", lw=1, label="# Positions (V3)")
    ax.axhline(y=cfg.target_exposure * 100, color="#3fb950", ls="--", alpha=0.7, label=f"Target ({cfg.target_exposure:.0%})")
    ax.set_ylabel("Exposure % / # Positions")
    ax.set_title("V3 Exposure & Position Count Over Time", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    # Rolling 30-day average exposure
    v2_roll = v2_eq["exposure"].rolling(30).mean() * 100
    v3_roll = v3_eq["exposure"].rolling(30).mean() * 100
    ax2.plot(v2_eq.index, v2_roll, color="#f85149", lw=1.5, label="V2 (30d avg)")
    ax2.plot(v3_eq.index, v3_roll, color="#58a6ff", lw=1.5, label="V3 (30d avg)")
    ax2.axhline(y=cfg.target_exposure * 100, color="#3fb950", ls="--", alpha=0.7)
    ax2.set_ylabel("30-Day Avg Exposure %")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(cfg.output_dir, "exposure_timeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    v3_cfg = StrategyConfig()
    v2_cfg = get_v2_config()
    os.makedirs(v3_cfg.output_dir, exist_ok=True)

    # ══════════════════════════════════════════
    # PHASE 1: LOAD DATA
    # ══════════════════════════════════════════
    print("\n[1/6] Loading data...")
    data = load_all(v3_cfg)

    # Pre-compute correlations for V3
    print("  Computing rolling correlations...")
    pair_corr = compute_rolling_correlations(data, v3_cfg.correlation_window)
    print(f"  {len(pair_corr)} asset pairs tracked")

    # ══════════════════════════════════════════
    # PHASE 2: V2 BASELINE (Before)
    # ══════════════════════════════════════════
    print("\n[2/6] Running V2 baseline (before)...")
    v2_signals = {}
    for ticker, df in data.items():
        v2_signals[ticker] = generate_signals(df, v2_cfg, long_only=True)

    v2_eq, v2_trades = run_backtest(v2_signals, v2_cfg)
    v2_metrics = compute_metrics(v2_eq, v2_trades, v2_cfg)
    print(format_metrics(v2_metrics, "V2 BASELINE (BEFORE)"))

    v2_per_asset = run_per_asset(v2_signals, v2_cfg)

    # ══════════════════════════════════════════
    # PHASE 3: V3 CAPITAL-EFFICIENT (After)
    # ══════════════════════════════════════════
    print("\n[3/6] Running V3 capital-efficient system (after)...")
    v3_signals = {}
    for ticker, df in data.items():
        v3_signals[ticker] = generate_signals(df, v3_cfg, long_only=True)
        rs = v3_signals[ticker]["regime_score"]
        above = (rs >= v3_cfg.regime_score_threshold).mean()
        print(f"  {ticker}: regime eligible on {above:.1%} of days (threshold={v3_cfg.regime_score_threshold})")

    v3_eq, v3_trades = run_backtest(v3_signals, v3_cfg, pair_corr)
    v3_metrics = compute_metrics(v3_eq, v3_trades, v3_cfg)
    print(format_metrics(v3_metrics, "V3 CAPITAL-EFFICIENT (AFTER)"))

    v3_per_asset = run_per_asset(v3_signals, v3_cfg, pair_corr)

    # ══════════════════════════════════════════
    # PHASE 4: BEFORE vs AFTER
    # ══════════════════════════════════════════
    print("\n[4/6] Before vs After comparison...")
    print(print_comparison_table(v2_metrics, v3_metrics))
    print()
    print(per_asset_comparison(v2_per_asset, v3_per_asset))

    # ══════════════════════════════════════════
    # PHASE 5: WALK-FORWARD (V3)
    # ══════════════════════════════════════════
    print("\n[5/6] Walk-forward validation (V3)...")
    wf = walk_forward_test(data, v3_cfg, long_only=True, pair_correlations=pair_corr)
    wf_eq = wf["oos_equity"]
    wf_trades = wf["oos_trades"]

    if len(wf_eq) > 0:
        wf_metrics = compute_metrics(wf_eq, wf_trades, v3_cfg)
        print(format_metrics(wf_metrics, "V3 WALK-FORWARD OOS"))

        print("\n  Per-fold results:")
        for fr in wf["fold_results"]:
            print(f"    Fold {fr['fold']}: Sharpe={fr['sharpe']:+.3f}, "
                  f"Return={fr['return']:+.2%}, Trades={fr['trades']}")

        avg_oos = np.mean([f["sharpe"] for f in wf["fold_results"]])
        print(f"\n  Average OOS Sharpe: {avg_oos:.3f}")
        print(f"  V3 Full-sample Sharpe: {v3_metrics['Sharpe Ratio']:.3f}")

    # ══════════════════════════════════════════
    # PHASE 6: VISUALIZATIONS
    # ══════════════════════════════════════════
    print("\n[6/6] Generating visualizations...")

    p = plot_equity_comparison(v2_eq, v3_eq, wf_eq, v3_cfg)
    print(f"  Saved: {p}")

    p = plot_trade_distribution(v2_trades, v3_trades, v3_cfg)
    print(f"  Saved: {p}")

    p = plot_exposure_timeline(v2_eq, v3_eq, v3_cfg)
    print(f"  Saved: {p}")

    for ticker in v3_signals:
        p = plot_asset_detail(v3_signals[ticker], v3_trades, ticker, v3_cfg)
        print(f"  Saved: {p}")

    p_wf = plot_wf_folds(wf["fold_results"], v3_cfg)
    if p_wf:
        print(f"  Saved: {p_wf}")

    # Save data
    v3_trades.to_csv(os.path.join(v3_cfg.output_dir, "v3_trades.csv"), index=False)
    v3_eq.to_csv(os.path.join(v3_cfg.output_dir, "v3_equity.csv"))
    v2_trades.to_csv(os.path.join(v3_cfg.output_dir, "v2_trades.csv"), index=False)
    v2_eq.to_csv(os.path.join(v3_cfg.output_dir, "v2_equity.csv"))

    # ══════════════════════════════════════════
    # CONCLUSIONS
    # ══════════════════════════════════════════
    print("\n" + "=" * 64)
    print("  V3 CAPITAL EFFICIENCY ANALYSIS")
    print("=" * 64)

    print(f"\n  EXPOSURE IMPROVEMENT:")
    print(f"    V2 avg exposure: {v2_metrics.get('Avg Exposure', 0):.1%}")
    print(f"    V3 avg exposure: {v3_metrics.get('Avg Exposure', 0):.1%}")
    print(f"    Target range:    40-70%")
    v3_exp = v3_metrics.get("Avg Exposure", 0)
    if 0.40 <= v3_exp <= 0.70:
        print(f"    >> TARGET MET")
    elif v3_exp > 0.70:
        print(f"    >> ABOVE TARGET (consider reducing)")
    else:
        print(f"    >> Below target ({v3_exp:.1%} vs 40%)")

    print(f"\n  TRADE FREQUENCY:")
    print(f"    V2: {v2_metrics.get('Total Trades', 0)} trades ({v2_metrics.get('Trades/Year', 0):.1f}/year)")
    print(f"    V3: {v3_metrics.get('Total Trades', 0)} trades ({v3_metrics.get('Trades/Year', 0):.1f}/year)")
    print(f"    Target: 100-300 over 5 years")
    v3_trades_n = v3_metrics.get("Total Trades", 0)
    if 100 <= v3_trades_n <= 300:
        print(f"    >> TARGET MET")
    else:
        print(f"    >> {'Above' if v3_trades_n > 300 else 'Below'} target ({v3_trades_n})")

    print(f"\n  RISK-ADJUSTED RETURNS:")
    print(f"    V2 Sharpe: {v2_metrics.get('Sharpe Ratio', 0):.3f}")
    print(f"    V3 Sharpe: {v3_metrics.get('Sharpe Ratio', 0):.3f}")
    print(f"    Target: > 1.0")

    if len(v3_trades) > 0 and "layer" in v3_trades.columns:
        print(f"\n  LAYER CONTRIBUTION:")
        for layer in sorted(v3_trades["layer"].unique()):
            lt = v3_trades[v3_trades["layer"] == layer]
            pnl = lt["pnl"].sum()
            wr = len(lt[lt["pnl"] > 0]) / len(lt) if len(lt) > 0 else 0
            print(f"    Layer {layer}: {len(lt)} trades, PnL=${pnl:,.0f}, WR={wr:.1%}")

    print(f"\n  CORRELATION CLUSTERS:")
    from strategy.correlation import detect_clusters
    # Show current clusters
    all_dates_list = sorted(set().union(*(df.index for df in data.values())))
    last_date = all_dates_list[-1]
    clusters = detect_clusters(pair_corr, last_date, list(data.keys()), v3_cfg.correlation_threshold)
    for i, cluster in enumerate(clusters):
        print(f"    Cluster {i}: {', '.join(sorted(cluster))}")

    print(f"\n  KEY IMPROVEMENTS:")
    print(f"    1. Continuous regime scaling (no binary filter)")
    print(f"    2. Multi-level entries at Z=[{', '.join(str(z) for z, _, _ in v3_cfg.entry_levels)}]")
    print(f"    3. RSI confirmation on weak signals")
    print(f"    4. Correlation-aware cluster capping ({v3_cfg.max_cluster_exposure:.0%} per cluster)")
    print(f"    5. Regime-adaptive exposure (sideways +{(v3_cfg.sideways_exposure_bonus-1):.0%}, trending -{(1-v3_cfg.trending_exposure_penalty):.0%})")
    print(f"    6. Tighter stops ({v3_cfg.stop_atr_multiple}x ATR) for faster capital recycling")

    print(f"\n  Outputs: {os.path.abspath(v3_cfg.output_dir)}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
