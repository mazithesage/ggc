"""
Visualization: equity curves, Z-scores, regime overlays, sensitivity heatmaps.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from config import StrategyConfig


def plot_equity_curves(
    equity_by_asset: dict[str, pd.DataFrame],
    combined_equity: pd.DataFrame,
    cfg: StrategyConfig,
) -> str:
    """Plot equity curves for each asset and combined. Returns output path."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})

    # Combined equity curve
    ax = axes[0]
    ax.plot(combined_equity.index, combined_equity["equity"], "k-", linewidth=1.5, label="Combined")
    ax.axhline(y=cfg.initial_capital, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Combined Portfolio Equity Curve", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # Drawdown
    ax2 = axes[1]
    cummax = combined_equity["equity"].cummax()
    drawdown = (combined_equity["equity"] - cummax) / cummax * 100
    ax2.fill_between(combined_equity.index, drawdown, 0, color="red", alpha=0.3)
    ax2.set_title("Drawdown (%)", fontsize=12)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.tight_layout()
    path = os.path.join(cfg.output_dir, "equity_curve_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_asset_signals(
    signals_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    ticker: str,
    cfg: StrategyConfig,
) -> str:
    """Plot price, Z-score, half-life, and trade markers for one asset."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Price with trade markers
    ax = axes[0]
    ax.plot(signals_df.index, signals_df["close"], "k-", linewidth=0.8, label="Close")

    ticker_trades = trades_df[trades_df["ticker"] == ticker] if len(trades_df) > 0 else pd.DataFrame()
    if len(ticker_trades) > 0:
        longs = ticker_trades[ticker_trades["direction"] == "LONG"]
        shorts = ticker_trades[ticker_trades["direction"] == "SHORT"]

        for _, t in longs.iterrows():
            color = "green" if t["pnl"] > 0 else "red"
            ax.annotate("", xy=(t["exit_date"], t["exit_price"]),
                       xytext=(t["entry_date"], t["entry_price"]),
                       arrowprops=dict(arrowstyle="->", color=color, alpha=0.6))

        for _, t in shorts.iterrows():
            color = "green" if t["pnl"] > 0 else "red"
            ax.annotate("", xy=(t["exit_date"], t["exit_price"]),
                       xytext=(t["entry_date"], t["entry_price"]),
                       arrowprops=dict(arrowstyle="->", color=color, alpha=0.6, linestyle="--"))

    ax.set_title(f"{ticker} — Price & Trades", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.grid(True, alpha=0.3)

    # Z-score
    ax2 = axes[1]
    ax2.plot(signals_df.index, signals_df["zscore"], "b-", linewidth=0.7)
    ax2.axhline(y=cfg.z_entry_long, color="green", linestyle="--", alpha=0.7, label=f"Long entry (Z={cfg.z_entry_long})")
    ax2.axhline(y=cfg.z_entry_short, color="red", linestyle="--", alpha=0.7, label=f"Short entry (Z={cfg.z_entry_short})")
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.axhspan(cfg.z_exit_short, cfg.z_exit_long * -1, alpha=0.05, color="yellow", label="Exit zone")
    ax2.set_title("Z-Score", fontsize=12)
    ax2.set_ylabel("Z-Score")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-4, 4)

    # Half-life with regime filter
    ax3 = axes[2]
    hl = signals_df["halflife"].copy()
    ax3.plot(signals_df.index, hl, "purple", linewidth=0.7, label="Half-life (days)")
    ax3.axhline(y=cfg.ou_halflife_min, color="green", linestyle=":", alpha=0.7)
    ax3.axhline(y=cfg.ou_halflife_max, color="red", linestyle=":", alpha=0.7)
    ax3.fill_between(
        signals_df.index,
        cfg.ou_halflife_min,
        cfg.ou_halflife_max,
        alpha=0.1,
        color="green",
        label=f"Tradable regime [{cfg.ou_halflife_min}-{cfg.ou_halflife_max}d]",
    )
    ax3.set_title("OU Half-Life (Regime Filter)", fontsize=12)
    ax3.set_ylabel("Half-life (days)")
    ax3.set_ylim(0, 60)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.tight_layout()
    path = os.path.join(cfg.output_dir, f"signals_{ticker.replace('-', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sensitivity_heatmap(
    results: pd.DataFrame,
    metric: str,
    cfg: StrategyConfig,
) -> str:
    """Plot heatmap of metric vs Z-threshold and window length."""
    pivot = results.pivot(index="z_threshold", columns="window", values=metric)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"Parameter Sensitivity: {metric}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Z-Score Window (days)")
    ax.set_ylabel("Z-Score Entry Threshold")

    plt.tight_layout()
    path = os.path.join(cfg.output_dir, f"sensitivity_{metric.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_monthly_returns(equity_df: pd.DataFrame, cfg: StrategyConfig) -> str:
    """Plot monthly returns heatmap."""
    equity = equity_df["equity"]
    monthly = equity.resample("ME").last().pct_change().dropna()
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        pivot * 100,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Monthly Returns (%)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(cfg.output_dir, "monthly_returns.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
