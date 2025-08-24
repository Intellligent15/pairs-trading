# analysis.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def _validate_closes(closes: pd.DataFrame) -> None:
    """
    Ensure the input is a 2-column, date-indexed DataFrame with no duplicate dates.
    """
    if not isinstance(closes, pd.DataFrame):
        raise TypeError("closes must be a pandas DataFrame with two columns (the two ETFs).")
    if closes.shape[1] != 2:
        raise ValueError(f"Expected exactly 2 columns (two ETFs), got {closes.shape[1]}.")
    if closes.index.has_duplicates:
        raise ValueError("Index has duplicate dates. De-duplicate before calling.")
    if not closes.index.is_monotonic_increasing:
        # Keep it explicit; reordering silently can mask data issues
        raise ValueError("Index must be sorted in ascending date order.")


def log_prices_from_closes(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log prices: ln(P_t).
    Returns a DataFrame with same shape/columns/index.
    """
    _validate_closes(closes)
    if (closes <= 0).any().any():
        raise ValueError("Close prices must be strictly positive to take logarithms.")
    return np.log(closes)


def log_returns_from_closes(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns: r_t = ln(P_t / P_{t-1}).
    Equivalent to: log_prices.diff().
    Returns a DataFrame aligned to closes.index (first row NaN).
    """
    _validate_closes(closes)
    # Either of these are fine; use log1p(pct_change) for numerical stability
    # return np.log(closes).diff()
    return np.log1p(closes.pct_change())


def return_correlation(
    log_returns: pd.DataFrame,
    lookback: int = 120,
    drop_na: bool = True
) -> float:
    """
    Pearson correlation of the two return series over the most recent 'lookback' bars.
    Returns a single float in [-1, 1]. Raises if insufficient history.
    """
    if log_returns.shape[1] != 2:
        raise ValueError("log_returns must have exactly 2 columns.")
    if len(log_returns) < lookback:
        raise ValueError(f"Insufficient history for lookback={lookback}. Have {len(log_returns)} rows.")
    window = log_returns.tail(lookback)
    if drop_na:
        window = window.dropna(how="any")
    if window.empty or (len(window) < 2):
        raise ValueError("Not enough non-NaN points to compute correlation.")
    c = window.corr().iloc[0, 1]
    # In rare cases correlation matrix can yield NaN (e.g., constant series)
    if pd.isna(c):
        raise ValueError("Correlation is NaN (one or both series may be constant).")
    return float(c)


def rolling_return_correlation(
    log_returns: pd.DataFrame,
    lookback: int = 120
) -> pd.Series:
    """
    Rolling Pearson correlation over a moving 'lookback' window.
    Returns a pandas Series indexed by date.
    """
    if log_returns.shape[1] != 2:
        raise ValueError("log_returns must have exactly 2 columns.")
    a = log_returns.iloc[:, 0]
    b = log_returns.iloc[:, 1]
    # Use pandas' built-in rolling corr for efficiency/clarity
    return a.rolling(lookback).corr(b)


def prep_pair(
    closes: pd.DataFrame,
    lookback: int = 120
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Convenience one-liner for your workflow steps 1-2.
    Input:
        closes: aligned close prices (two ETF columns, ascending date index)
        lookback: lookback window for the correlation check
    Output:
        (log_prices, log_returns, return_corr_lookback)
    """
    lp = log_prices_from_closes(closes)
    lr = log_returns_from_closes(closes)
    corr_lb = return_correlation(lr, lookback=lookback)
    return lp, lr, corr_lb
