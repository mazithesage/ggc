"""
Correlation-aware allocation: rolling pairwise correlations and cluster detection.

Used to prevent redundant exposure across highly correlated assets.
"""
import numpy as np
import pandas as pd
from config import StrategyConfig


def compute_rolling_correlations(
    price_data: dict[str, pd.DataFrame],
    window: int = 60,
) -> dict[str, pd.DataFrame]:
    """
    Compute rolling pairwise return correlations.

    Returns dict mapping each date → correlation matrix DataFrame.
    For efficiency, returns a dict of {(ticker_a, ticker_b): pd.Series of correlations}.
    """
    # Build returns DataFrame
    returns = {}
    for ticker, df in price_data.items():
        close = df["Close"] if "Close" in df.columns else df["close"]
        returns[ticker] = close.pct_change()

    returns_df = pd.DataFrame(returns).dropna()
    tickers = list(returns_df.columns)

    # Compute rolling pairwise correlations
    pair_correlations = {}
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if j <= i:
                continue
            pair_key = (t1, t2)
            pair_correlations[pair_key] = (
                returns_df[t1]
                .rolling(window, min_periods=window // 2)
                .corr(returns_df[t2])
            )

    return pair_correlations


def get_correlation_at_date(
    pair_correlations: dict[tuple, pd.Series],
    date: pd.Timestamp,
    ticker: str,
    existing_tickers: list[str],
    threshold: float = 0.75,
) -> float:
    """
    Return the max absolute correlation between ticker and any existing position.
    Used to determine if a new entry would create redundant exposure.
    """
    max_corr = 0.0
    for existing in existing_tickers:
        pair = tuple(sorted([ticker, existing]))
        if pair in pair_correlations:
            series = pair_correlations[pair]
            if date in series.index:
                c = series.loc[date]
                if not np.isnan(c):
                    max_corr = max(max_corr, abs(c))
    return max_corr


def detect_clusters(
    pair_correlations: dict[tuple, pd.Series],
    date: pd.Timestamp,
    tickers: list[str],
    threshold: float = 0.75,
) -> list[set]:
    """
    Group tickers into correlated clusters at a given date.
    Uses simple greedy union: if A~B and B~C, then {A,B,C} is one cluster.
    """
    # Build adjacency
    parent = {t: t for t in tickers}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (t1, t2), series in pair_correlations.items():
        if t1 in tickers and t2 in tickers:
            if date in series.index:
                c = series.loc[date]
                if not np.isnan(c) and abs(c) > threshold:
                    union(t1, t2)

    # Group by root
    clusters = {}
    for t in tickers:
        root = find(t)
        clusters.setdefault(root, set()).add(t)

    return list(clusters.values())


def compute_cluster_exposure(
    clusters: list[set],
    position_tickers: dict[str, float],
) -> dict[str, float]:
    """
    Given clusters and current position exposures per ticker,
    return total exposure per cluster root.
    """
    cluster_exp = {}
    for cluster in clusters:
        cluster_key = sorted(cluster)[0]
        total = sum(position_tickers.get(t, 0.0) for t in cluster)
        cluster_exp[cluster_key] = total
    return cluster_exp


def get_cluster_for_ticker(clusters: list[set], ticker: str) -> set:
    """Find which cluster a ticker belongs to."""
    for cluster in clusters:
        if ticker in cluster:
            return cluster
    return {ticker}
