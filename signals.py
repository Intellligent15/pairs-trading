# signals.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# --------------------- Validation ---------------------

def _validate_spread(spread: pd.Series) -> None:
    """
    Ensure 'spread' is a 1-D pandas Series, date-indexed, strictly increasing,
    without duplicate timestamps, and finite where not NaN.
    """
    if not isinstance(spread, pd.Series):
        raise TypeError("spread must be a pandas Series (the residual/spread over time).")
    if spread.index.has_duplicates:
        raise ValueError("Spread index has duplicate timestamps. De-duplicate before calling.")
    if not spread.index.is_monotonic_increasing:
        raise ValueError("Spread index must be sorted in ascending order.")
    # Allow NaNs (e.g., at the very start), but no infinities
    finite_mask = np.isfinite(spread.values) | pd.isna(spread.values)
    if not finite_mask.all():
        raise ValueError("Spread contains +/-inf values. Clean the data first.")
    if spread.dtype.kind not in {"f", "i"}:
        # Try to coerce to float; raise if impossible
        try:
            spread.astype(float)
        except Exception as e:
            raise TypeError("Spread must be numeric (float or int).") from e


# --------------------- Core rolling z-score ---------------------

def rolling_mean_std(
    spread: pd.Series,
    window: int = 60,
    ddof: int = 1,
    avoid_lookahead: bool = True,
    min_periods: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute rolling mean and std of the spread with optional 1-step shift to avoid look-ahead.

    Parameters
    ----------
    spread : pd.Series
        Residual/spread series S_t (e.g., from Engle-Granger).
    window : int
        Lookback window length for rolling stats.
    ddof : int
        Delta degrees of freedom for std; ddof=1 matches sample std.
    avoid_lookahead : bool
        If True, shift the rolling stats by 1 bar so values at time t use data through t-1.
    min_periods : Optional[int]
        Minimum observations in window required to have a value; defaults to window.

    Returns
    -------
    (mu_roll, sigma_roll) : Tuple[pd.Series, pd.Series]
        Rolling mean and std, aligned to spread.index (and optionally shifted).
    """
    _validate_spread(spread)
    if window <= 1:
        raise ValueError("window must be > 1 for a meaningful rolling std.")
    if min_periods is None:
        min_periods = window

    mu = spread.rolling(window=window, min_periods=min_periods).mean()
    sig = spread.rolling(window=window, min_periods=min_periods).std(ddof=ddof)

    if avoid_lookahead:
        mu = mu.shift(1)
        sig = sig.shift(1)

    mu.name = "mu_roll"
    sig.name = "sigma_roll"
    return mu, sig


def rolling_zscore(
    spread: pd.Series,
    window: int = 60,
    ddof: int = 1,
    avoid_lookahead: bool = True,
    min_periods: Optional[int] = None,
    clip_at: Optional[float] = None,
    return_components: bool = False,
) -> pd.Series | Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Standardize the spread into a rolling z-score: Z_t = (S_t - mu_roll) / sigma_roll.

    By default, the rolling mean/std are shifted by 1 bar so that z_t at time t uses
    only information available up to t-1 (no look-ahead).

    Parameters
    ----------
    spread : pd.Series
        Residual/spread series S_t.
    window : int
        Lookback for rolling mean/std (typical: 60-90).
    ddof : int
        Std degrees of freedom; ddof=1 (sample std) is typical.
    avoid_lookahead : bool
        Shift rolling stats by 1 bar if True.
    min_periods : Optional[int]
        Required observations per window; default = window.
    clip_at : Optional[float]
        If provided, clip z-scores to +/- clip_at to reduce extreme outliers.
    return_components : bool
        If True, return (z, mu_roll, sigma_roll) instead of just z.

    Returns
    -------
    z : pd.Series  (or (z, mu_roll, sigma_roll) if return_components=True)
        Rolling z-score series aligned to spread.index.
    """
    mu, sig = rolling_mean_std(
        spread,
        window=window,
        ddof=ddof,
        avoid_lookahead=avoid_lookahead,
        min_periods=min_periods,
    )

    # Avoid division by zero: where sigma == 0, z is NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (spread - mu) / sig

    z = z.astype(float)
    z.name = "z"

    if clip_at is not None:
        z = z.clip(lower=-abs(clip_at), upper=abs(clip_at))

    if return_components:
        return z, mu, sig
    return z
