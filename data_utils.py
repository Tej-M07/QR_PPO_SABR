"""Utilities for efficient empirical chain sampling from parquet data."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChainSpec:
    """A unique identifier for a chain within a ticker."""

    ticker: str
    contract_id: str


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling configuration for evaluation and training."""

    min_dte: int = 21
    max_dte: int = 252
    min_chain_len: int = 20
    max_paths: int = 200
    seed: int = 0
    max_tickers_scan: int | None = 60


OPTION_COLUMNS: Sequence[str] = (
    "contract_id",
    "symbol",
    "expiration",
    "strike",
    "type",
    "mark",
    "bid",
    "ask",
    "date",
    "implied_volatility",
    "delta",
    "gamma",
    "vega",
)

UNDERLYING_COLUMNS: Sequence[str] = ("symbol", "date", "close")


def list_tickers(data_dir: str = "data") -> List[str]:
    """List ticker directories under `data_dir`."""
    tickers: List[str] = []
    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path):
            tickers.append(name)
    tickers.sort()
    return tickers


def _read_parquet(
    path: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Read parquet using pyarrow backend and only selected columns."""
    return pd.read_parquet(path, engine="pyarrow", columns=list(columns))


def _ensure_datetime(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        return
    if not np.issubdtype(df[column].dtype, np.datetime64):
        df[column] = pd.to_datetime(df[column], errors="coerce")


def _compute_dte_days(df: pd.DataFrame) -> pd.Series:
    """Compute DTE in calendar days from expiration and date."""
    exp = df["expiration"]
    dt = df["date"]
    dte = (exp - dt).dt.total_seconds() / (24.0 * 3600.0)
    return dte


def _clean_chain_df(df: pd.DataFrame, numeric_cols: Sequence[str]) -> pd.DataFrame:
    """Defensively clean NaNs and missing greeks for environment consumption."""
    df = df.sort_values("date").reset_index(drop=True).copy()

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward-fill within a chain, then back-fill, finally fill with zeros.
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0.0)

    # Ensure bid/ask consistency.
    if "bid" in df.columns:
        df["bid"] = pd.to_numeric(df["bid"], errors="coerce").fillna(0.0)
    if "ask" in df.columns:
        df["ask"] = pd.to_numeric(df["ask"], errors="coerce").fillna(0.0)
    if "mark" in df.columns:
        df["mark"] = pd.to_numeric(df["mark"], errors="coerce").fillna((df["bid"] + df["ask"]) / 2.0)

    return df


def _compute_skew_and_vol_of_vol(chain: pd.DataFrame, iv_col: str = "implied_volatility") -> pd.DataFrame:
    """Compute skew and vol-of-vol proxies from the IV time series."""
    chain = chain.copy()
    iv = chain[iv_col].astype(float)

    # Skew proxy: deviation from rolling mean IV.
    skew_window = min(20, max(5, len(chain) // 4))
    rolling_mean = iv.rolling(window=skew_window, min_periods=1).mean()
    chain["skew_proxy"] = (iv - rolling_mean).clip(-2.0, 2.0).astype(float)

    # Vol-of-vol proxy: rolling std of IV changes (proxy for stochastic vol).
    iv_chg = iv.diff().fillna(0.0)
    vol_window = min(20, max(5, len(chain) // 4))
    rolling_std = iv_chg.rolling(window=vol_window, min_periods=1).std().fillna(0.0)
    chain["vol_of_vol_proxy"] = rolling_std.clip(0.0, 2.0).astype(float)

    return chain


def build_chain(
    ticker_dir: str,
    chain: ChainSpec,
    dte_min: int,
    dte_max: int,
) -> Optional[pd.DataFrame]:
    """Build a time-ordered chain df for one contract_id within a ticker.

    The returned dataframe includes columns required by `HistoricalHedgingEnv`.
    Returns None if the chain can't be built (too short or missing data).
    """
    options_path = os.path.join(ticker_dir, "options.parquet")
    underlying_path = os.path.join(ticker_dir, "underlying.parquet")

    options_df = _read_parquet(options_path, OPTION_COLUMNS)
    _ensure_datetime(options_df, "date")
    _ensure_datetime(options_df, "expiration")
    options_df["dte"] = _compute_dte_days(options_df)

    contract_rows = options_df[options_df["contract_id"] == chain.contract_id].copy()
    contract_rows = contract_rows[
        (contract_rows["dte"] >= float(dte_min)) & (contract_rows["dte"] <= float(dte_max))
    ]
    contract_rows = contract_rows.sort_values("date")
    if contract_rows.empty or len(contract_rows) < 3:
        return None

    # Load underlying close once per chain build (we can cache externally if needed).
    underlying_df = _read_parquet(underlying_path, UNDERLYING_COLUMNS)
    _ensure_datetime(underlying_df, "date")
    underlying_df = underlying_df.sort_values("date")

    # Merge underlying close onto the chain dates.
    contract_rows = contract_rows.merge(
        underlying_df[["date", "close"]].rename(columns={"close": "underlying_close"}),
        on="date",
        how="left",
    )

    # Defensive cleaning.
    numeric_cols = [
        "strike",
        "implied_volatility",
        "delta",
        "gamma",
        "vega",
        "bid",
        "ask",
        "mark",
        "underlying_close",
        "dte",
    ]
    contract_rows = _clean_chain_df(contract_rows, numeric_cols=numeric_cols)

    contract_rows["dte"] = contract_rows["dte"].clip(0.0, 10_000.0)
    contract_rows = _compute_skew_and_vol_of_vol(contract_rows)

    # Required columns by env: symbol, expiration, strike, date, greeks, bids/marks.
    required = [
        "symbol",
        "expiration",
        "strike",
        "date",
        "implied_volatility",
        "delta",
        "gamma",
        "vega",
        "bid",
        "ask",
        "mark",
        "underlying_close",
        "dte",
        "skew_proxy",
        "vol_of_vol_proxy",
    ]
    for col in required:
        if col not in contract_rows.columns:
            contract_rows[col] = 0.0

    return contract_rows[required].copy()


def enumerate_chain_candidates(
    ticker_dir: str,
    dte_min: int,
    dte_max: int,
    min_chain_len: int = 20,
) -> List[ChainSpec]:
    """Enumerate chain contract_ids eligible for sampling within a ticker."""
    ticker = os.path.basename(ticker_dir)
    options_path = os.path.join(ticker_dir, "options.parquet")
    options_df = _read_parquet(options_path, OPTION_COLUMNS)
    _ensure_datetime(options_df, "date")
    _ensure_datetime(options_df, "expiration")
    options_df["dte"] = _compute_dte_days(options_df)

    eligible = options_df[
        (options_df["dte"] >= float(dte_min)) & (options_df["dte"] <= float(dte_max))
    ]
    if eligible.empty:
        return []

    # Keep only contract_ids with sufficient number of time points.
    counts = eligible.groupby("contract_id", observed=True)["date"].count()
    keep = counts[counts >= float(min_chain_len)].index.astype(str).tolist()
    return [ChainSpec(ticker=ticker, contract_id=cid) for cid in keep]


def sample_chain_specs(
    data_dir: str,
    cfg: SamplingConfig,
) -> List[ChainSpec]:
    """Randomly sample chain specs across all tickers."""
    rng = np.random.default_rng(cfg.seed)
    ticker_dirs = [os.path.join(data_dir, t) for t in list_tickers(data_dir)]
    if cfg.max_tickers_scan is not None and cfg.max_tickers_scan < len(ticker_dirs):
        idx = rng.choice(len(ticker_dirs), size=cfg.max_tickers_scan, replace=False)
        ticker_dirs = [ticker_dirs[i] for i in idx]

    all_candidates: List[ChainSpec] = []
    for tdir in ticker_dirs:
        all_candidates.extend(
            enumerate_chain_candidates(
                tdir,
                cfg.min_dte,
                cfg.max_dte,
                min_chain_len=cfg.min_chain_len,
            )
        )

    if not all_candidates:
        return []

    n = min(cfg.max_paths, len(all_candidates))
    idx = rng.choice(len(all_candidates), size=n, replace=False)
    return [all_candidates[i] for i in idx]


def build_chains_for_specs(
    specs: List[ChainSpec],
    data_dir: str,
    dte_min: int = 21,
    dte_max: int = 252,
) -> List[Tuple[ChainSpec, pd.DataFrame]]:
    """Build multiple chains efficiently by caching per-ticker parquet reads.

    Args:
        specs: List of chain specs to build.
        data_dir: Root data directory containing ticker subfolders.
        dte_min: Minimum DTE (inclusive) in calendar days.
        dte_max: Maximum DTE (inclusive) in calendar days.

    Returns:
        List of (spec, chain_df) pairs. Specs that can't be built are skipped.
    """
    if not specs:
        return []

    specs_by_ticker: Dict[str, List[ChainSpec]] = {}
    for spec in specs:
        specs_by_ticker.setdefault(spec.ticker, []).append(spec)

    built: List[Tuple[ChainSpec, pd.DataFrame]] = []
    for ticker, ticker_specs in specs_by_ticker.items():
        ticker_dir = os.path.join(data_dir, ticker)
        options_path = os.path.join(ticker_dir, "options.parquet")
        underlying_path = os.path.join(ticker_dir, "underlying.parquet")

        options_df = _read_parquet(options_path, OPTION_COLUMNS)
        _ensure_datetime(options_df, "date")
        _ensure_datetime(options_df, "expiration")
        options_df["dte"] = _compute_dte_days(options_df)

        underlying_df = _read_parquet(underlying_path, UNDERLYING_COLUMNS)
        _ensure_datetime(underlying_df, "date")
        underlying_df = underlying_df.sort_values("date")
        underlying_small = underlying_df[["date", "close"]].rename(columns={"close": "underlying_close"})

        contract_ids = [s.contract_id for s in ticker_specs]
        subset = options_df[options_df["contract_id"].isin(contract_ids)].copy()

        # Apply DTE filter per-row before slicing chains.
        subset = subset[(subset["dte"] >= float(dte_min)) & (subset["dte"] <= float(dte_max))]

        for spec in ticker_specs:
            chain_rows = subset[subset["contract_id"] == spec.contract_id].copy()
            chain_rows = chain_rows.sort_values("date")
            if chain_rows.empty or len(chain_rows) < 3:
                continue

            chain_rows = chain_rows.merge(underlying_small, on="date", how="left")

            numeric_cols = [
                "strike",
                "implied_volatility",
                "delta",
                "gamma",
                "vega",
                "bid",
                "ask",
                "mark",
                "underlying_close",
                "dte",
            ]
            chain_rows = _clean_chain_df(chain_rows, numeric_cols=numeric_cols)
            chain_rows["dte"] = chain_rows["dte"].clip(0.0, 10_000.0)
            chain_rows = _compute_skew_and_vol_of_vol(chain_rows)

            required = [
                "symbol",
                "expiration",
                "strike",
                "date",
                "implied_volatility",
                "delta",
                "gamma",
                "vega",
                "bid",
                "ask",
                "mark",
                "underlying_close",
                "dte",
                "skew_proxy",
                "vol_of_vol_proxy",
            ]
            for col in required:
                if col not in chain_rows.columns:
                    chain_rows[col] = 0.0
            built.append((spec, chain_rows[required].copy()))

    return built

