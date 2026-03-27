"""
Statistical analysis of V2/V3 trading strategy results.
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, shapiro, mannwhitneyu, binomtest
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

OUT = "/Users/mac/Documents/GitHub/ggc/output/eda"
DATA = "/Users/mac/Documents/GitHub/ggc/output"

# ── Load ──
v3_trades = pd.read_csv(f"{DATA}/v3_trades.csv")
v2_trades = pd.read_csv(f"{DATA}/v2_trades.csv")
v3_eq = pd.read_csv(f"{DATA}/v3_equity.csv", index_col=0, parse_dates=True)
v2_eq = pd.read_csv(f"{DATA}/v2_equity.csv", index_col=0, parse_dates=True)

print(f"V3: {len(v3_trades)} trades, V2: {len(v2_trades)} trades")
print(f"V3 equity: {len(v3_eq)} days, V2 equity: {len(v2_eq)} days")

report = []
def h(title): report.append(f"\n## {title}\n")
def t(text): report.append(text)
def img(f, c=""): report.append(f"\n![{c}]({f})\n")

report.append("# Statistical Analysis of Trading Strategy Results\n")

# ════════════════════════════════════════════════
# 1. RETURN DISTRIBUTION NORMALITY
# ════════════════════════════════════════════════
h("1. Trade Return Distribution — Normality Tests")

for name, trades in [("V3", v3_trades), ("V2", v2_trades)]:
    r = trades["return_pct"].dropna()
    if len(r) < 8:
        t(f"\n**{name}:** Too few trades ({len(r)}) for normality tests.\n")
        continue

    jb_stat, jb_p = jarque_bera(r)
    sw_stat, sw_p = shapiro(r.values[:5000])
    ad_result = stats.anderson(r.values, dist="norm")

    t(f"\n### {name} ({len(r)} trades)\n")
    t(f"| Test | Statistic | p-value | Normal? |")
    t(f"|------|-----------|---------|---------|")
    t(f"| Jarque-Bera | {jb_stat:.2f} | {jb_p:.4e} | {'No' if jb_p < 0.05 else 'Yes'} |")
    t(f"| Shapiro-Wilk | {sw_stat:.4f} | {sw_p:.4e} | {'No' if sw_p < 0.05 else 'Yes'} |")
    ad_crit = ad_result.critical_values[2]  # 5% level
    t(f"| Anderson-Darling | {ad_result.statistic:.3f} | crit={ad_crit:.3f} | {'No' if ad_result.statistic > ad_crit else 'Yes'} |")
    t(f"\nSkewness: {r.skew():.3f}, Kurtosis: {r.kurtosis():.2f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, trades, color) in zip(axes, [("V3", v3_trades, "#58a6ff"), ("V2", v2_trades, "#f85149")]):
    r = trades["return_pct"].dropna() * 100
    if len(r) < 2: continue
    ax.hist(r, bins=40, color=color, alpha=0.7, density=True, edgecolor="none")
    mu, sigma = r.mean(), r.std()
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "k--", lw=1.5, label="Normal fit")
    ax.axvline(0, color="gray", ls="-", alpha=0.5)
    ax.set_title(f"{name} Trade Returns")
    ax.set_xlabel("Return %")
    ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/stat_01_return_dist.png", bbox_inches="tight")
plt.close()
img("stat_01_return_dist.png", "Trade Return Distributions")
print("  Saved: stat_01_return_dist.png")

# ════════════════════════════════════════════════
# 2. WIN RATE SIGNIFICANCE
# ════════════════════════════════════════════════
h("2. Win Rate — Binomial Test")

for name, trades in [("V3", v3_trades), ("V2", v2_trades)]:
    r = trades["return_pct"].dropna()
    if len(r) < 2: continue
    wins = (r > 0).sum()
    n = len(r)
    wr = wins / n
    result = binomtest(wins, n, 0.5, alternative="greater")
    t(f"\n**{name}:** {wins}/{n} wins ({wr:.1%}), p={result.pvalue:.4e} {'***' if result.pvalue < 0.001 else '**' if result.pvalue < 0.01 else '*' if result.pvalue < 0.05 else 'n.s.'}")
    t(f"  95% CI: [{result.proportion_ci(confidence_level=0.95).low:.3f}, {result.proportion_ci(confidence_level=0.95).high:.3f}]")

# ════════════════════════════════════════════════
# 3. SHARPE RATIO BOOTSTRAP
# ════════════════════════════════════════════════
h("3. Sharpe Ratio — Bootstrap 95% CI")

def bootstrap_sharpe(equity_df, n_boot=10000, seed=42):
    if "equity" not in equity_df.columns:
        return None, None, None
    daily_ret = equity_df["equity"].pct_change().dropna()
    if len(daily_ret) < 30:
        return None, None, None
    observed = daily_ret.mean() / daily_ret.std() * np.sqrt(365)
    rng = np.random.RandomState(seed)
    boot_sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(daily_ret.values, size=len(daily_ret), replace=True)
        s = sample.mean() / sample.std() * np.sqrt(365) if sample.std() > 0 else 0
        boot_sharpes.append(s)
    ci_lo = np.percentile(boot_sharpes, 2.5)
    ci_hi = np.percentile(boot_sharpes, 97.5)
    return observed, ci_lo, ci_hi

for name, eq in [("V3", v3_eq), ("V2", v2_eq)]:
    obs, lo, hi = bootstrap_sharpe(eq)
    if obs is not None:
        t(f"\n**{name}:** Sharpe = {obs:.3f}, 95% CI = [{lo:.3f}, {hi:.3f}]")
        t(f"  {'Statistically significant (CI excludes 0)' if lo > 0 else 'NOT statistically significant (CI includes 0)'}")

# Bootstrap distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, eq, color) in zip(axes, [("V3", v3_eq, "#58a6ff"), ("V2", v2_eq, "#f85149")]):
    if "equity" not in eq.columns: continue
    daily_ret = eq["equity"].pct_change().dropna()
    rng = np.random.RandomState(42)
    boots = []
    for _ in range(10000):
        s = rng.choice(daily_ret.values, size=len(daily_ret), replace=True)
        boots.append(s.mean() / s.std() * np.sqrt(365) if s.std() > 0 else 0)
    ax.hist(boots, bins=60, color=color, alpha=0.7, edgecolor="none")
    ax.axvline(0, color="red", ls="--", lw=1.5, label="Zero")
    ax.axvline(np.percentile(boots, 2.5), color="black", ls=":", lw=1, label="95% CI")
    ax.axvline(np.percentile(boots, 97.5), color="black", ls=":", lw=1)
    ax.set_title(f"{name} Bootstrapped Sharpe")
    ax.set_xlabel("Sharpe Ratio")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/stat_02_sharpe_bootstrap.png", bbox_inches="tight")
plt.close()
img("stat_02_sharpe_bootstrap.png", "Sharpe Bootstrap")
print("  Saved: stat_02_sharpe_bootstrap.png")

# ════════════════════════════════════════════════
# 4. REGIME DEPENDENCE
# ════════════════════════════════════════════════
h("4. Regime Dependence — High vs Low Regime Score Trades")

if "regime_score" in v3_trades.columns:
    rs = v3_trades[["return_pct", "regime_score"]].dropna()
    median_rs = rs["regime_score"].median()
    high = rs[rs["regime_score"] >= median_rs]["return_pct"]
    low = rs[rs["regime_score"] < median_rs]["return_pct"]

    if len(high) > 1 and len(low) > 1:
        mw_stat, mw_p = mannwhitneyu(high, low, alternative="greater")
        t_stat, t_p = stats.ttest_ind(high, low, equal_var=False)

        t(f"\nMedian regime score split: {median_rs:.3f}")
        t(f"High regime trades: {len(high)}, mean return: {high.mean():.4f}")
        t(f"Low regime trades: {len(low)}, mean return: {low.mean():.4f}")
        t(f"\n| Test | Statistic | p-value | Significant? |")
        t(f"|------|-----------|---------|-------------|")
        t(f"| Mann-Whitney U | {mw_stat:.2f} | {mw_p:.4f} | {'Yes' if mw_p < 0.05 else 'No'} |")
        t(f"| Welch's t-test | {t_stat:.3f} | {t_p:.4f} | {'Yes' if t_p < 0.05 else 'No'} |")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot([low.values * 100, high.values * 100], labels=["Low Regime", "High Regime"])
        ax.axhline(0, color="red", ls="--", alpha=0.5)
        ax.set_ylabel("Trade Return %")
        ax.set_title(f"Return by Regime Score (split at {median_rs:.2f})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUT}/stat_03_regime_dependence.png", bbox_inches="tight")
        plt.close()
        img("stat_03_regime_dependence.png", "Regime Dependence")
        print("  Saved: stat_03_regime_dependence.png")

# ════════════════════════════════════════════════
# 5. EQUITY CURVE STATIONARITY
# ════════════════════════════════════════════════
h("5. Equity Curve Stationarity — ADF Test on Daily Returns")

for name, eq in [("V3", v3_eq), ("V2", v2_eq)]:
    if "equity" not in eq.columns: continue
    daily_ret = eq["equity"].pct_change().dropna()
    adf_stat, adf_p, _, _, crit, _ = adfuller(daily_ret, maxlag=20)
    t(f"\n**{name}:**")
    t(f"  ADF statistic: {adf_stat:.4f}")
    t(f"  p-value: {adf_p:.4e}")
    t(f"  Critical values: 1%={crit['1%']:.3f}, 5%={crit['5%']:.3f}")
    t(f"  {'Stationary (reject unit root)' if adf_p < 0.05 else 'Non-stationary'}")

# ════════════════════════════════════════════════
# 6. V2 vs V3 COMPARISON
# ════════════════════════════════════════════════
h("6. V2 vs V3 Paired Comparison")

common = v2_eq.index.intersection(v3_eq.index)
if len(common) > 30:
    v2_ret = v2_eq.loc[common, "equity"].pct_change().dropna()
    v3_ret = v3_eq.loc[common, "equity"].pct_change().dropna()
    common_idx = v2_ret.index.intersection(v3_ret.index)
    v2_r = v2_ret.loc[common_idx]
    v3_r = v3_ret.loc[common_idx]
    diff = v3_r - v2_r

    t_stat, t_p = stats.ttest_rel(v3_r.values, v2_r.values)
    w_stat, w_p = stats.wilcoxon(diff.values)

    t(f"\nPaired comparison on {len(common_idx)} overlapping days:")
    t(f"  V3 mean daily return: {v3_r.mean():.6f}")
    t(f"  V2 mean daily return: {v2_r.mean():.6f}")
    t(f"  Difference: {diff.mean():.6f}")
    t(f"\n| Test | Statistic | p-value | V3 better? |")
    t(f"|------|-----------|---------|-----------|")
    t(f"| Paired t-test | {t_stat:.3f} | {t_p:.4f} | {'Yes' if t_p < 0.05 and diff.mean() > 0 else 'No'} |")
    t(f"| Wilcoxon signed-rank | {w_stat:.1f} | {w_p:.4f} | {'Yes' if w_p < 0.05 and diff.mean() > 0 else 'No'} |")

    # Rolling difference
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    roll_diff = diff.rolling(60).mean() * 365 * 100
    axes[0].plot(roll_diff.index, roll_diff, color="#58a6ff", lw=1.5)
    axes[0].axhline(0, color="red", ls="--")
    axes[0].set_ylabel("V3 - V2 Ann. Return Diff (%)")
    axes[0].set_title("Rolling 60-Day: V3 vs V2 Return Difference")
    axes[0].fill_between(roll_diff.index, roll_diff, 0,
                         where=roll_diff > 0, color="#3fb950", alpha=0.3)
    axes[0].fill_between(roll_diff.index, roll_diff, 0,
                         where=roll_diff < 0, color="#f85149", alpha=0.3)
    axes[0].grid(True, alpha=0.3)

    # Cumulative difference
    cum_diff = diff.cumsum() * 100
    axes[1].plot(cum_diff.index, cum_diff, color="#d2a8ff", lw=1.5)
    axes[1].set_ylabel("Cumulative Return Diff (%)")
    axes[1].set_title("Cumulative: V3 vs V2")
    axes[1].fill_between(cum_diff.index, cum_diff, 0,
                         where=cum_diff > 0, color="#3fb950", alpha=0.2)
    axes[1].fill_between(cum_diff.index, cum_diff, 0,
                         where=cum_diff < 0, color="#f85149", alpha=0.2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/stat_04_v2_v3_diff.png", bbox_inches="tight")
    plt.close()
    img("stat_04_v2_v3_diff.png", "V2 vs V3 Comparison")
    print("  Saved: stat_04_v2_v3_diff.png")

# ════════════════════════════════════════════════
# 7. DRAWDOWN ANALYSIS
# ════════════════════════════════════════════════
h("7. Drawdown Analysis")

def drawdown_stats(eq_df, name):
    eq = eq_df["equity"]
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Drawdown durations
    in_dd = dd < 0
    dd_starts = in_dd & ~in_dd.shift(1, fill_value=False)
    dd_ends = ~in_dd & in_dd.shift(1, fill_value=False)

    durations = []
    depths = []
    start = None
    for i, (date, val) in enumerate(dd.items()):
        if dd_starts.iloc[i]:
            start = date
        if dd_ends.iloc[i] and start is not None:
            dur = (date - start).days
            depth = dd.loc[start:date].min()
            durations.append(dur)
            depths.append(depth)
            start = None
    # Still in drawdown
    if start is not None:
        dur = (dd.index[-1] - start).days
        depth = dd.loc[start:].min()
        durations.append(dur)
        depths.append(depth)

    return {
        "name": name, "max_dd": max_dd,
        "durations": durations, "depths": depths,
        "avg_dur": np.mean(durations) if durations else 0,
        "max_dur": max(durations) if durations else 0,
        "n_dd": len(durations),
    }

for name, eq in [("V3", v3_eq), ("V2", v2_eq)]:
    s = drawdown_stats(eq, name)
    t(f"\n**{name}:**")
    t(f"  Max drawdown: {s['max_dd']:.2%}")
    t(f"  Number of drawdown periods: {s['n_dd']}")
    t(f"  Avg drawdown duration: {s['avg_dur']:.0f} days")
    t(f"  Max drawdown duration: {s['max_dur']} days")

# ════════════════════════════════════════════════
# 8. SERIAL CORRELATION (LJUNG-BOX)
# ════════════════════════════════════════════════
h("8. Serial Correlation — Ljung-Box Test on Trade Returns")

for name, trades in [("V3", v3_trades), ("V2", v2_trades)]:
    r = trades["return_pct"].dropna()
    if len(r) < 20:
        t(f"\n**{name}:** Too few trades for Ljung-Box test.")
        continue
    lb = acorr_ljungbox(r, lags=[5, 10, 15], return_df=True)
    t(f"\n**{name}:**")
    t(f"| Lag | LB Statistic | p-value | Serial Correlation? |")
    t(f"|-----|-------------|---------|---------------------|")
    for lag, row in lb.iterrows():
        t(f"| {lag} | {row['lb_stat']:.2f} | {row['lb_pvalue']:.4f} | {'Yes' if row['lb_pvalue'] < 0.05 else 'No'} |")

t(f"\nNo significant serial correlation means consecutive trade returns are independent — good for strategy robustness.")

# ════════════════════════════════════════════════
# WRITE REPORT
# ════════════════════════════════════════════════
with open(f"{OUT}/statistical_analysis_report.md", "w") as f:
    f.write("\n".join(report))
print(f"\n  Report: {OUT}/statistical_analysis_report.md")
print("  Done!")
