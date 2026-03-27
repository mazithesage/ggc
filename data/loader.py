"""
Data loader: downloads OHLCV from yfinance with local CSV caching.
"""
import os
import pandas as pd
import yfinance as yf
from config import StrategyConfig


def load_asset(ticker: str, cfg: StrategyConfig) -> pd.DataFrame:
    """Download or load cached daily OHLCV data for a single asset."""
    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache_path = os.path.join(cfg.cache_dir, f"{ticker.replace('-', '_')}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
    else:
        df = yf.download(
            ticker,
            start=cfg.start_date,
            end=cfg.end_date,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(cache_path)

    df = df.loc[cfg.start_date:cfg.end_date].copy()

    # Validate
    if df.empty:
        raise ValueError(f"No data for {ticker} in [{cfg.start_date}, {cfg.end_date}]")

    nan_pct = df["Close"].isna().mean()
    if nan_pct > 0.05:
        raise ValueError(f"{ticker}: {nan_pct:.1%} NaN in Close prices — too many gaps")

    df["Close"] = df["Close"].ffill()
    return df


def load_all(cfg: StrategyConfig) -> dict[str, pd.DataFrame]:
    """Load data for all configured assets. Returns {ticker: DataFrame}."""
    data = {}
    for ticker in cfg.assets:
        print(f"Loading {ticker}...")
        data[ticker] = load_asset(ticker, cfg)
        print(f"  {len(data[ticker])} bars from {data[ticker].index[0].date()} to {data[ticker].index[-1].date()}")
    return data
