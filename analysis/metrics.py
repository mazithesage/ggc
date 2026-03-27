"""
Performance metrics v2: adds Calmar ratio, exposure-adjusted returns, trade analysis.
"""
import numpy as np
import pandas as pd
from config import StrategyConfig


def compute_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    cfg: StrategyConfig,
) -> dict:
    equity = equity_df["equity"]
    returns = equity.pct_change().dropna()
    ann = cfg.crypto_trading_days
    total_days = (equity.index[-1] - equity.index[0]).days
    years = total_days / 365.25 if total_days > 0 else 1e-6

    m = {}

    # ── Return metrics ──
    total_return = equity.iloc[-1] / equity.iloc[0]
    m["Total Return"] = total_return - 1
    m["CAGR"] = total_return ** (1 / years) - 1 if years > 0 else 0

    # ── Risk metrics ──
    m["Sharpe Ratio"] = (returns.mean() / returns.std() * np.sqrt(ann)) if returns.std() > 0 else 0
    downside = returns[returns < 0]
    ds_std = downside.std() if len(downside) > 1 else 1e-10
    m["Sortino Ratio"] = (returns.mean() / ds_std * np.sqrt(ann)) if ds_std > 0 else 0

    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    m["Max Drawdown"] = drawdown.min()
    m["Max Drawdown Date"] = str(drawdown.idxmin().date()) if len(drawdown) > 0 else "N/A"

    # Calmar Ratio = CAGR / |Max Drawdown|
    m["Calmar Ratio"] = m["CAGR"] / abs(m["Max Drawdown"]) if m["Max Drawdown"] != 0 else 0

    # ── Exposure ──
    if "exposure" in equity_df.columns:
        avg_exposure = equity_df["exposure"].mean()
        m["Avg Exposure"] = avg_exposure
        m["Exposure-Adj Return"] = m["CAGR"] / avg_exposure if avg_exposure > 0 else 0
    else:
        m["Avg Exposure"] = 0
        m["Exposure-Adj Return"] = 0

    # ── Trade metrics ──
    if len(trades_df) > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        m["Total Trades"] = len(trades_df)
        m["Win Rate"] = len(wins) / len(trades_df)
        m["Avg Trade Return"] = trades_df["return_pct"].mean()
        m["Avg Bars Held"] = trades_df["bars_held"].mean()
        m["Trades/Year"] = len(trades_df) / years if years > 0 else 0

        gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        m["Profit Factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        m["Avg Win"] = wins["return_pct"].mean() if len(wins) > 0 else 0
        m["Avg Loss"] = losses["return_pct"].mean() if len(losses) > 0 else 0
        m["Best Trade"] = trades_df["return_pct"].max()
        m["Worst Trade"] = trades_df["return_pct"].min()

        exit_reasons = trades_df["exit_reason"].value_counts().to_dict()
        m["Exit Reasons"] = exit_reasons

        longs = trades_df[trades_df["direction"] == "LONG"]
        shorts = trades_df[trades_df["direction"] == "SHORT"]
        m["Long Trades"] = len(longs)
        m["Short Trades"] = len(shorts)
        m["Long Win Rate"] = len(longs[longs["pnl"] > 0]) / len(longs) if len(longs) > 0 else 0
        m["Short Win Rate"] = len(shorts[shorts["pnl"] > 0]) / len(shorts) if len(shorts) > 0 else 0

        if "regime_score" in trades_df.columns:
            m["Avg Regime Score"] = trades_df["regime_score"].mean()
    else:
        m["Total Trades"] = 0
        m["Win Rate"] = 0
        m["Profit Factor"] = 0
        m["Trades/Year"] = 0

    m["Initial Capital"] = cfg.initial_capital
    m["Final Equity"] = equity.iloc[-1]

    return m


def format_metrics(metrics: dict, title: str = "") -> str:
    lines = []
    w = 57
    lines.append("=" * w)
    lines.append(f"  {title}" if title else "  PERFORMANCE REPORT")
    lines.append("=" * w)
    lines.append(f"  Initial Capital:   ${metrics.get('Initial Capital', 0):>14,.2f}")
    lines.append(f"  Final Equity:      ${metrics.get('Final Equity', 0):>14,.2f}")
    lines.append(f"  Total Return:      {metrics.get('Total Return', 0):>14.2%}")
    lines.append(f"  CAGR:              {metrics.get('CAGR', 0):>14.2%}")
    lines.append("-" * w)
    lines.append(f"  Sharpe Ratio:      {metrics.get('Sharpe Ratio', 0):>14.3f}")
    lines.append(f"  Sortino Ratio:     {metrics.get('Sortino Ratio', 0):>14.3f}")
    lines.append(f"  Calmar Ratio:      {metrics.get('Calmar Ratio', 0):>14.3f}")
    lines.append(f"  Max Drawdown:      {metrics.get('Max Drawdown', 0):>14.2%}")
    lines.append(f"  Max DD Date:       {metrics.get('Max Drawdown Date', 'N/A'):>14s}")
    lines.append(f"  Avg Exposure:      {metrics.get('Avg Exposure', 0):>14.2%}")
    lines.append(f"  Exp-Adj Return:    {metrics.get('Exposure-Adj Return', 0):>14.2%}")
    lines.append("-" * w)
    lines.append(f"  Total Trades:      {metrics.get('Total Trades', 0):>14d}")
    lines.append(f"  Win Rate:          {metrics.get('Win Rate', 0):>14.2%}")
    lines.append(f"  Profit Factor:     {metrics.get('Profit Factor', 0):>14.3f}")
    lines.append(f"  Avg Trade Return:  {metrics.get('Avg Trade Return', 0):>14.4%}")
    lines.append(f"  Avg Bars Held:     {metrics.get('Avg Bars Held', 0):>14.1f}")
    lines.append(f"  Trades/Year:       {metrics.get('Trades/Year', 0):>14.1f}")

    if metrics.get("Total Trades", 0) > 0:
        lines.append("-" * w)
        lines.append(f"  Long / Short:      {metrics.get('Long Trades', 0):>6d} / {metrics.get('Short Trades', 0):<6d}")
        lines.append(f"  Long Win Rate:     {metrics.get('Long Win Rate', 0):>14.2%}")
        lines.append(f"  Short Win Rate:    {metrics.get('Short Win Rate', 0):>14.2%}")
        lines.append(f"  Avg Win / Loss:    {metrics.get('Avg Win', 0):>+.4%} / {metrics.get('Avg Loss', 0):>+.4%}")
        lines.append(f"  Best / Worst:      {metrics.get('Best Trade', 0):>+.4%} / {metrics.get('Worst Trade', 0):>+.4%}")

    exit_reasons = metrics.get("Exit Reasons", {})
    if exit_reasons:
        lines.append("-" * w)
        lines.append("  Exit Reasons:")
        for reason, count in sorted(exit_reasons.items()):
            lines.append(f"    {reason:<20s} {count:>6d}")

    lines.append("=" * w)
    return "\n".join(lines)
