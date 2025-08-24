# adaptive.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from analysis import log_prices_from_closes
from stats import fit_engle_granger
from signals import rolling_zscore


# -------- Half-life (AR(1)) --------

def spread_half_life(spread: pd.Series) -> float:
    """
    Estimate mean-reversion half-life using AR(1):
        S_t = a + b*S_{t-1} + e_t
      half-life = -ln(2) / ln(b)

    Returns np.inf if b is outside (0,1) or insufficient data.
    """
    s = spread.dropna()
    if len(s) < 10:
        return np.inf
    y = s.iloc[1:].values
    x = s.shift(1).iloc[1:].values
    # Guard against all-zeros/constant
    if np.allclose(np.std(x), 0.0):
        return np.inf
    # Simple OLS slope
    b, a = np.polyfit(x, y, 1)
    if b <= 0.0 or b >= 1.0 or not np.isfinite(b):
        return np.inf
    return float(-np.log(2.0) / np.log(b))


@dataclass(frozen=True)
class AdaptiveConfig:
    train_window: int = 252          # OLS/ADF lookback for each re-fit
    refit_every: int = 10            # re-fit cadence (in trading days)
    z_window_mult: float = 5.0       # z-window ≈ z_window_mult * half-life
    z_window_min: int = 20           # clamp z-window
    z_window_max: int = 120
    adf_regression: str = "n"        # 'n' for residuals
    fallback_z_window: int = 60      # if half-life is inf/NaN
    # Optional: you can disable half-life adaptation by setting z_window_mult=None


@dataclass(frozen=True)
class AdaptiveOutputs:
    spread: pd.Series                 # piecewise spread using rolling (alpha,beta)
    z: pd.Series                      # adaptive z-score with no look-ahead
    beta_series: pd.Series            # piecewise-constant beta used at each date
    alpha_series: pd.Series           # piecewise-constant alpha used at each date
    hl_series: pd.Series              # half-life estimated at each re-fit date
    z_window_series: pd.Series        # z-window chosen at each re-fit date
    regimes: List[Tuple[pd.Timestamp, pd.Timestamp, float, float, float, int]]
    # list of (seg_start, seg_end, alpha, beta, half_life, z_window)


def _clip_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(x))))


def rolling_refit_adaptive(
    closes: pd.DataFrame,
    cfg: AdaptiveConfig = AdaptiveConfig()
) -> AdaptiveOutputs:
    """
    Rolling re-fit of (alpha, beta) every cfg.refit_every days.
    For each re-fit:
      - Fit Engle–Granger on the last `train_window` log prices (through *t*),
      - Compute training residuals -> estimate half-life,
      - Choose z-window ≈ mult * half-life, clamped to [min,max],
      - From the *next day* onward, apply the new (alpha,beta) until next re-fit.

    No look-ahead:
      - z-score computation uses rolling stats shifted by 1 (inside rolling_zscore),
      - The regime parameters chosen at day t apply from t+1 onward.

    Returns spread, z, and piecewise constants (alpha/beta) plus diagnostics.
    """
    if closes.shape[1] != 2:
        raise ValueError("closes must have exactly 2 columns (X first, Y second).")
    if not closes.index.is_monotonic_increasing:
        raise ValueError("closes index must be ascending.")
    if (closes <= 0).any().any():
        raise ValueError("closes must be strictly positive.")

    lp = log_prices_from_closes(closes)
    idx = lp.index
    n = len(lp)
    tw = cfg.train_window
    step = cfg.refit_every

    # First re-fit index (inclusive) is where we have tw points available
    first_refit = tw - 1
    if n <= first_refit:
        raise ValueError(f"Need > {tw} rows to start rolling re-fit; have {n}.")

    # Build anchor points for re-fits
    anchors = list(range(first_refit, n, step))
    # Ensure last anchor is included
    if anchors[-1] != n - 1:
        anchors.append(n - 1)

    # Storage
    alpha_series = pd.Series(index=idx, dtype=float)
    beta_series  = pd.Series(index=idx, dtype=float)
    hl_points    = []
    zw_points    = []
    regimes: List[Tuple[pd.Timestamp, pd.Timestamp, float, float, float, int]] = []

    # We will build spread with piecewise alpha/beta applied from (anchor+1) .. next_anchor
    spread = pd.Series(index=idx, dtype=float)

    for j, anc in enumerate(anchors):
        # Training slice ends at 'anc' (inclusive)
        start_i = anc - tw + 1
        if start_i < 0:
            continue
        train_slice = lp.iloc[start_i:anc + 1]

        # Fit EG on the training window
        eg, resid_train, _ = fit_engle_granger(train_slice, train_window=len(train_slice), adf_regression=cfg.adf_regression)

        # Half-life on training residuals
        hl = spread_half_life(resid_train)
        if cfg.z_window_mult is not None and np.isfinite(hl):
            zwin = _clip_int(cfg.z_window_mult * hl, cfg.z_window_min, cfg.z_window_max)
        else:
            zwin = cfg.fallback_z_window

        hl_points.append((idx[anc], float(hl)))
        zw_points.append((idx[anc], int(zwin)))

        # Regime applies from next day (anc+1) up to (next_anchor)
        seg_start_i = anc + 1
        seg_end_i   = anchors[j + 1] if j + 1 < len(anchors) else (n - 1)
        if seg_start_i > seg_end_i:
            continue  # nothing to apply (end of series)

        # Fill alpha/beta piecewise for this segment
        alpha_series.iloc[seg_start_i:seg_end_i + 1] = eg.alpha
        beta_series.iloc[seg_start_i:seg_end_i + 1]  = eg.beta

        # Compute spread on this segment (using future day's prices but yesterday's parameters)
        X = lp.iloc[:, 0]
        Y = lp.iloc[:, 1]
        seg_idx = idx[seg_start_i: seg_end_i + 1]
        spread.loc[seg_idx] = X.loc[seg_idx] - (eg.alpha + eg.beta * Y.loc[seg_idx])

        # Track regime metadata
        regimes.append((idx[seg_start_i], idx[seg_end_i], eg.alpha, eg.beta, float(hl), int(zwin)))

    # Backfill any leading NaNs (before first regime) with initial fixed-parameter spread (optional)
    # For strict no-lookahead, we leave the pre-first-regime values as NaN.
    # It won’t be tradable anyway (no parameters available).
    # alpha_series/beta_series may have NaNs for the early period; that's expected.

    # Build an adaptive z-score by stitching per-regime windows
    z = pd.Series(index=idx, dtype=float)
    for (seg_start, seg_end, _a, _b, _hl, zwin) in regimes:
        seg = slice(seg_start, seg_end)
        # compute z on full series with this window, then take the segment
        z_full = rolling_zscore(spread, window=zwin, avoid_lookahead=True)
        z.loc[seg] = z_full.loc[seg]

    # Diagnostics series at anchor dates
    hl_series = pd.Series({t: v for t, v in hl_points}, name="half_life")
    z_window_series = pd.Series({t: v for t, v in zw_points}, name="z_window")

    # Ensure names
    spread.name = "spread_adaptive"
    z.name = "z_adaptive"
    alpha_series.name = "alpha_piecewise"
    beta_series.name = "beta_piecewise"

    return AdaptiveOutputs(spread, z, beta_series, alpha_series, hl_series, z_window_series, regimes)
