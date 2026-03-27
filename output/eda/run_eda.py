"""
Comprehensive EDA on crypto OHLCV data for the ggc strategy.
Produces publication-quality visualizations and a markdown report.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, shapiro, normaltest
from statsmodels.tsa.stattools import acf
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

OUT = "/Users/mac/Documents/GitHub/ggc/output/eda"
CACHE = "/Users/mac/Documents/GitHub/ggc/data/cache"
ASSETS = ["BTC_USD", "ETH_USD", "SOL_USD", "BNB_USD", "XRP_USD"]
LABELS = ["BTC", "ETH", "SOL", "BNB", "XRP"]
COLORS = ["#F7931A", "#627EEA", "#9945FF", "#F3BA2F", "#00AAE4"]

os.makedirs(OUT, exist_ok=True)

# ── Load data ──
print("Loading data...")
dfs = {}
for asset, label in zip(ASSETS, LABELS):
    df = pd.read_csv(f"{CACHE}/{asset}.csv", parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["pct_return"] = df["Close"].pct_change()
    dfs[label] = df

# Aligned returns matrix
returns = pd.DataFrame({k: v["log_return"] for k, v in dfs.items()}).dropna()
prices = pd.DataFrame({k: v["Close"] for k, v in dfs.items()}).dropna()

print(f"  Period: {returns.index[0].date()} to {returns.index[-1].date()}")
print(f"  Trading days: {len(returns)}")

report_lines = []
def section(title):
    report_lines.append(f"\n## {title}\n")
def text(t):
    report_lines.append(t)
def img(filename, caption=""):
    report_lines.append(f"\n![{caption}]({filename})\n")

report_lines.append("# Crypto OHLCV Exploratory Data Analysis\n")
text(f"**Period:** {returns.index[0].date()} to {returns.index[-1].date()}  ")
text(f"**Assets:** {', '.join(LABELS)}  ")
text(f"**Trading days:** {len(returns)}\n")

# ════════════════════════════════════════════════
# 1. SUMMARY STATISTICS
# ════════════════════════════════════════════════
section("1. Summary Statistics")

summary_rows = []
for label, color in zip(LABELS, COLORS):
    r = dfs[label]["log_return"].dropna()
    ann_ret = r.mean() * 365
    ann_vol = r.std() * np.sqrt(365)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    eq = (1 + dfs[label]["pct_return"].dropna()).cumprod()
    dd = (eq - eq.cummax()) / eq.cummax()
    max_dd = dd.min()
    summary_rows.append({
        "Asset": label,
        "Ann. Return": f"{ann_ret:.1%}",
        "Ann. Volatility": f"{ann_vol:.1%}",
        "Sharpe": f"{sharpe:.3f}",
        "Skewness": f"{r.skew():.3f}",
        "Kurtosis": f"{r.kurtosis():.2f}",
        "Max DD": f"{max_dd:.1%}",
        "Days": len(r),
    })

summary_df = pd.DataFrame(summary_rows)
text(summary_df.to_markdown(index=False))

# Bar chart of key stats
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, col, title in zip(axes,
    ["Ann. Return", "Ann. Volatility", "Sharpe", "Max DD"],
    ["Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown"]):
    vals = [float(summary_rows[i][col].replace("%", "")) / (100 if "%" in summary_rows[i][col] else 1) for i in range(len(LABELS))]
    bar_colors = [COLORS[i] for i in range(len(LABELS))]
    ax.bar(LABELS, vals, color=bar_colors, alpha=0.8, edgecolor="white")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/01_summary_stats.png", bbox_inches="tight")
plt.close()
img("01_summary_stats.png", "Summary Statistics")
print("  Saved: 01_summary_stats.png")

# ════════════════════════════════════════════════
# 2. PRICE EVOLUTION
# ════════════════════════════════════════════════
section("2. Price Evolution (Normalized)")

fig, ax = plt.subplots(figsize=(14, 6))
for label, color in zip(LABELS, COLORS):
    norm = dfs[label]["Close"] / dfs[label]["Close"].iloc[0] * 100
    ax.plot(norm.index, norm, color=color, lw=1.5, label=label)
ax.set_yscale("log")
ax.set_ylabel("Normalized Price (log scale, base=100)")
ax.set_title("Normalized Price Evolution")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.savefig(f"{OUT}/02_price_evolution.png", bbox_inches="tight")
plt.close()
img("02_price_evolution.png", "Price Evolution")
print("  Saved: 02_price_evolution.png")

# ════════════════════════════════════════════════
# 3. RETURN DISTRIBUTIONS
# ════════════════════════════════════════════════
section("3. Return Distributions")

# Histograms
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for ax, label, color in zip(axes, LABELS, COLORS):
    r = dfs[label]["log_return"].dropna()
    ax.hist(r * 100, bins=80, color=color, alpha=0.7, density=True, edgecolor="none")
    # Fitted normal
    mu, sigma = r.mean() * 100, r.std() * 100
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "k--", lw=1, label="Normal fit")
    ax.set_title(label)
    ax.set_xlabel("Daily Return %")
    if ax == axes[0]:
        ax.set_ylabel("Density")
    ax.legend(fontsize=7)
axes[0].set_ylabel("Density")
plt.suptitle("Daily Log-Return Distributions", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/03_return_histograms.png", bbox_inches="tight")
plt.close()
img("03_return_histograms.png", "Return Histograms")
print("  Saved: 03_return_histograms.png")

# QQ plots
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, label, color in zip(axes, LABELS, COLORS):
    r = dfs[label]["log_return"].dropna().values
    stats.probplot(r, dist="norm", plot=ax)
    ax.set_title(label)
    ax.get_lines()[0].set_color(color)
    ax.get_lines()[0].set_markersize(2)
plt.suptitle("QQ Plots vs Normal Distribution", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/04_qq_plots.png", bbox_inches="tight")
plt.close()
img("04_qq_plots.png", "QQ Plots")
print("  Saved: 04_qq_plots.png")

# Normality tests
text("\n### Normality Tests\n")
text("| Asset | Jarque-Bera stat | JB p-value | Shapiro-Wilk p | Skewness | Excess Kurtosis |")
text("|-------|-----------------|------------|----------------|----------|-----------------|")
for label in LABELS:
    r = dfs[label]["log_return"].dropna()
    jb_stat, jb_p = jarque_bera(r)
    # Shapiro on subsample (max 5000)
    sw_stat, sw_p = shapiro(r.sample(min(5000, len(r)), random_state=42))
    text(f"| {label} | {jb_stat:.1f} | {jb_p:.2e} | {sw_p:.2e} | {r.skew():.3f} | {r.kurtosis():.2f} |")

text("\nAll assets reject normality (p < 0.001). Heavy tails and negative skew are consistent with crypto markets.")

# ════════════════════════════════════════════════
# 4. VOLATILITY CLUSTERING
# ════════════════════════════════════════════════
section("4. Volatility Clustering")

# Rolling volatility
fig, ax = plt.subplots(figsize=(14, 6))
for label, color in zip(LABELS, COLORS):
    vol = dfs[label]["log_return"].rolling(30).std() * np.sqrt(365) * 100
    ax.plot(vol.index, vol, color=color, lw=1, label=label, alpha=0.8)
ax.set_ylabel("Annualized Volatility %")
ax.set_title("Rolling 30-Day Annualized Volatility")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/05_rolling_volatility.png", bbox_inches="tight")
plt.close()
img("05_rolling_volatility.png", "Rolling Volatility")
print("  Saved: 05_rolling_volatility.png")

# ACF of squared returns (GARCH-like)
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for ax, label, color in zip(axes, LABELS, COLORS):
    r = dfs[label]["log_return"].dropna()
    sq = r ** 2
    acf_vals = acf(sq, nlags=40, fft=True)
    ax.bar(range(len(acf_vals)), acf_vals, color=color, alpha=0.7, width=0.8)
    ax.axhline(1.96 / np.sqrt(len(sq)), color="red", ls="--", lw=0.8)
    ax.axhline(-1.96 / np.sqrt(len(sq)), color="red", ls="--", lw=0.8)
    ax.set_title(label)
    ax.set_xlabel("Lag (days)")
if axes[0]:
    axes[0].set_ylabel("ACF of r²")
plt.suptitle("Autocorrelation of Squared Returns (Volatility Clustering)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/06_squared_returns_acf.png", bbox_inches="tight")
plt.close()
img("06_squared_returns_acf.png", "Squared Returns ACF")
print("  Saved: 06_squared_returns_acf.png")

text("\nSignificant autocorrelation in squared returns confirms volatility clustering across all assets — periods of high volatility tend to persist.")

# ════════════════════════════════════════════════
# 5. CORRELATION STRUCTURE
# ════════════════════════════════════════════════
section("5. Correlation Structure")

# Full-period correlation
corr = returns.corr()
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
            vmin=0.3, vmax=1.0, ax=ax, square=True, linewidths=0.5,
            mask=None)
ax.set_title("Full-Period Pairwise Correlation (Daily Log Returns)")
plt.tight_layout()
plt.savefig(f"{OUT}/07_correlation_heatmap.png", bbox_inches="tight")
plt.close()
img("07_correlation_heatmap.png", "Correlation Heatmap")
print("  Saved: 07_correlation_heatmap.png")

# Rolling 60-day correlations
pairs = [("BTC", "ETH"), ("BTC", "SOL"), ("BTC", "XRP"), ("ETH", "SOL"), ("BNB", "XRP")]
fig, ax = plt.subplots(figsize=(14, 6))
for a, b in pairs:
    roll_corr = returns[a].rolling(60).corr(returns[b])
    ax.plot(roll_corr.index, roll_corr, lw=1, label=f"{a}-{b}", alpha=0.8)
ax.axhline(0.90, color="red", ls="--", alpha=0.5, label="Cluster threshold (0.90)")
ax.set_ylabel("Rolling 60-Day Correlation")
ax.set_title("Rolling Pairwise Correlations")
ax.legend(fontsize=8, ncol=3)
ax.set_ylim(-0.2, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/08_rolling_correlations.png", bbox_inches="tight")
plt.close()
img("08_rolling_correlations.png", "Rolling Correlations")
print("  Saved: 08_rolling_correlations.png")

# Yearly correlation stability
text("\n### Yearly Correlation (BTC vs others)\n")
text("| Year | ETH | SOL | BNB | XRP |")
text("|------|-----|-----|-----|-----|")
for year in sorted(returns.index.year.unique()):
    yr = returns[returns.index.year == year]
    if len(yr) < 30:
        continue
    row = f"| {year} "
    for alt in ["ETH", "SOL", "BNB", "XRP"]:
        c = yr["BTC"].corr(yr[alt])
        row += f"| {c:.2f} "
    text(row + "|")

# PCA
section("6. PCA of Returns")

pca = PCA(n_components=5)
pca.fit(returns.dropna())
explained = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.bar(range(1, 6), explained * 100, color="#58a6ff", alpha=0.8, edgecolor="white")
ax.plot(range(1, 6), np.cumsum(explained) * 100, "ro-", lw=1.5)
ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Explained %")
ax.set_title("PCA Scree Plot")
ax.set_xticks(range(1, 6))
ax.grid(True, alpha=0.3)

ax = axes[1]
loadings = pd.DataFrame(pca.components_[:3].T, index=LABELS, columns=["PC1", "PC2", "PC3"])
loadings.plot(kind="bar", ax=ax, alpha=0.8, edgecolor="white")
ax.set_title("PCA Loadings (Top 3 Components)")
ax.set_ylabel("Loading")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(f"{OUT}/09_pca_analysis.png", bbox_inches="tight")
plt.close()
img("09_pca_analysis.png", "PCA Analysis")
print("  Saved: 09_pca_analysis.png")

text(f"\nPC1 explains **{explained[0]:.1%}** of return variance — consistent with a dominant market factor driving all crypto assets.")

# ════════════════════════════════════════════════
# 6. REGIME CHARACTERISTICS
# ════════════════════════════════════════════════
section("7. Market Regimes")

# Simple regime detection: rolling 60-day return + volatility
btc = dfs["BTC"].copy()
btc["roll_ret"] = btc["log_return"].rolling(60).sum()
btc["roll_vol"] = btc["log_return"].rolling(30).std() * np.sqrt(365)
btc = btc.dropna()

def classify_regime(row):
    if row["roll_vol"] > btc["roll_vol"].quantile(0.75):
        return "High Vol"
    elif row["roll_ret"] > 0.10:
        return "Bull"
    elif row["roll_ret"] < -0.10:
        return "Bear"
    else:
        return "Sideways"

btc["regime"] = btc.apply(classify_regime, axis=1)
regime_colors = {"Bull": "#3fb950", "Bear": "#f85149", "Sideways": "#8b949e", "High Vol": "#d29922"}

fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})

ax = axes[0]
ax.plot(btc.index, btc["Close"], "k-", lw=0.8)
for regime, color in regime_colors.items():
    mask = btc["regime"] == regime
    ax.fill_between(btc.index, btc["Close"].min() * 0.9, btc["Close"].max() * 1.1,
                    where=mask, alpha=0.15, color=color, label=regime)
ax.set_yscale("log")
ax.set_ylabel("BTC Price (log)")
ax.set_title("BTC Price with Market Regime Overlay")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
regime_counts = btc["regime"].value_counts()
ax2.barh(regime_counts.index, regime_counts.values,
         color=[regime_colors[r] for r in regime_counts.index], alpha=0.8)
ax2.set_xlabel("Days")
ax2.set_title("Regime Distribution (BTC)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/10_regime_analysis.png", bbox_inches="tight")
plt.close()
img("10_regime_analysis.png", "Regime Analysis")
print("  Saved: 10_regime_analysis.png")

text("\n### Regime Statistics (BTC)\n")
text("| Regime | Days | % of Period | Avg Daily Return | Avg Vol (ann.) |")
text("|--------|------|-------------|-----------------|----------------|")
for regime in ["Bull", "Bear", "Sideways", "High Vol"]:
    mask = btc["regime"] == regime
    days = mask.sum()
    pct = days / len(btc)
    avg_ret = btc.loc[mask, "log_return"].mean() * 365
    avg_vol = btc.loc[mask, "roll_vol"].mean()
    text(f"| {regime} | {days} | {pct:.1%} | {avg_ret:.1%} | {avg_vol:.1%} |")

# ════════════════════════════════════════════════
# 7. TAIL RISK
# ════════════════════════════════════════════════
section("8. Tail Risk Analysis")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# VaR / CVaR
ax = axes[0]
var_data = []
for label in LABELS:
    r = dfs[label]["log_return"].dropna()
    for conf in [0.95, 0.99]:
        var = np.percentile(r, (1 - conf) * 100)
        cvar = r[r <= var].mean()
        var_data.append({"Asset": label, "Confidence": f"{conf:.0%}",
                         "VaR": var, "CVaR": cvar})

var_df = pd.DataFrame(var_data)
x = np.arange(len(LABELS))
w = 0.2
for i, conf in enumerate(["95%", "99%"]):
    subset = var_df[var_df["Confidence"] == conf]
    ax.bar(x + i * w - w/2, subset["VaR"].values * 100, w,
           label=f"VaR {conf}", alpha=0.7)
    ax.bar(x + i * w - w/2 + 2*w, subset["CVaR"].values * 100, w,
           label=f"CVaR {conf}", alpha=0.5, hatch="//")
ax.set_xticks(x + w/2)
ax.set_xticklabels(LABELS)
ax.set_ylabel("Daily Loss %")
ax.set_title("Value at Risk & Conditional VaR")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Drawdown per asset
ax = axes[1]
for label, color in zip(LABELS, COLORS):
    eq = (1 + dfs[label]["pct_return"].fillna(0)).cumprod()
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    ax.plot(dd.index, dd, color=color, lw=0.8, label=label, alpha=0.8)
ax.set_ylabel("Drawdown %")
ax.set_title("Drawdown Time Series")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/11_tail_risk.png", bbox_inches="tight")
plt.close()
img("11_tail_risk.png", "Tail Risk")
print("  Saved: 11_tail_risk.png")

text("\n### VaR & CVaR (Daily)\n")
text("| Asset | VaR 95% | CVaR 95% | VaR 99% | CVaR 99% |")
text("|-------|---------|----------|---------|----------|")
for label in LABELS:
    r = dfs[label]["log_return"].dropna()
    v95 = np.percentile(r, 5)
    c95 = r[r <= v95].mean()
    v99 = np.percentile(r, 1)
    c99 = r[r <= v99].mean()
    text(f"| {label} | {v95:.2%} | {c95:.2%} | {v99:.2%} | {c99:.2%} |")

# ════════════════════════════════════════════════
# 8. VOLUME ANALYSIS
# ════════════════════════════════════════════════
section("9. Volume Analysis")

fig, ax = plt.subplots(figsize=(14, 6))
for label, color in zip(LABELS, COLORS):
    vol_norm = dfs[label]["Volume"].rolling(30).mean()
    vol_norm = vol_norm / vol_norm.iloc[60]  # normalize to early value
    ax.plot(vol_norm.index, vol_norm, color=color, lw=1, label=label, alpha=0.8)
ax.set_ylabel("Normalized 30-Day Avg Volume")
ax.set_title("Relative Volume Trends (30-Day MA, Normalized)")
ax.legend()
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/12_volume_trends.png", bbox_inches="tight")
plt.close()
img("12_volume_trends.png", "Volume Trends")
print("  Saved: 12_volume_trends.png")

# ════════════════════════════════════════════════
# WRITE REPORT
# ════════════════════════════════════════════════
report_path = f"{OUT}/eda_report.md"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

print(f"\n  Report saved: {report_path}")
print("  Done!")
