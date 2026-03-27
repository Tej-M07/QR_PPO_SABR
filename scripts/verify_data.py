"""Validate local data directory structure and parquet schemas.

Usage:
    python scripts/verify_data.py --data-dir data --max-tickers 20
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import pandas as pd


OPTION_REQUIRED_COLUMNS = {
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
}

UNDERLYING_REQUIRED_COLUMNS = {"symbol", "date", "close"}


def _fmt_missing(missing: Iterable[str]) -> str:
    items = sorted(set(missing))
    return ", ".join(items) if items else "-"


def _read_columns(path: str) -> set[str]:
    # Reads only schema metadata through pyarrow backend.
    empty_df = pd.read_parquet(path, engine="pyarrow", columns=[])
    return set(empty_df.columns)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=0,
        help="Limit number of ticker folders checked (0 = all).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        return 1

    tickers = [
        name
        for name in sorted(os.listdir(args.data_dir))
        if os.path.isdir(os.path.join(args.data_dir, name))
    ]
    if args.max_tickers > 0:
        tickers = tickers[: args.max_tickers]

    if not tickers:
        print(f"[ERROR] No ticker folders found under: {args.data_dir}")
        return 1

    print(f"[INFO] Checking {len(tickers)} ticker folder(s) under '{args.data_dir}'")

    errors = 0
    for ticker in tickers:
        tdir = os.path.join(args.data_dir, ticker)
        options_path = os.path.join(tdir, "options.parquet")
        underlying_path = os.path.join(tdir, "underlying.parquet")

        if not os.path.exists(options_path):
            print(f"[ERROR] {ticker}: missing options.parquet")
            errors += 1
            continue
        if not os.path.exists(underlying_path):
            print(f"[ERROR] {ticker}: missing underlying.parquet")
            errors += 1
            continue

        try:
            option_cols = _read_columns(options_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {ticker}: cannot read options.parquet schema: {exc}")
            errors += 1
            continue

        try:
            underlying_cols = _read_columns(underlying_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {ticker}: cannot read underlying.parquet schema: {exc}")
            errors += 1
            continue

        option_missing = OPTION_REQUIRED_COLUMNS - option_cols
        underlying_missing = UNDERLYING_REQUIRED_COLUMNS - underlying_cols
        if option_missing or underlying_missing:
            print(
                f"[ERROR] {ticker}: missing columns | "
                f"options: {_fmt_missing(option_missing)} | "
                f"underlying: {_fmt_missing(underlying_missing)}"
            )
            errors += 1
            continue

        print(f"[OK] {ticker}")

    print("-" * 72)
    if errors:
        print(f"[FAIL] Validation completed with {errors} error(s).")
        return 1

    print(f"[PASS] Validation succeeded for {len(tickers)} ticker folder(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
