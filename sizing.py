# sizing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------- Validation -------------------------

def _validate_prices(px: pd.DataFrame) -> None:
    """
    Expect a 2-column DataFrame of positive prices with an ascending Date index.
    Column 0 is 'X' (first ETF), column 1 is 'Y' (second ETF).
    """
    if not isinstance(px, pd.DataFrame):
        raise TypeError("px must be a pandas DataFrame.")
    if px.shape[1] != 2:
        raise ValueError(f"px must have exactly 2 columns, got {px.shape[1]}.")
    if not px.index.is_monotonic_increasing:
        raise ValueError("px index must be sorted ascending.")
    if (px <= 0).any().any():
        raise ValueError("px must contain strictly positive prices.")
    if px.index.has_duplicates:
        raise ValueError("px index has duplicate timestamps.")


def _validate_signal(sig: pd.Series, px_index: pd.Index) -> None:
    if not isinstance(sig, pd.Series):
        raise TypeError("signal must be a pandas Series.")
    if not sig.index.equals(px_index):
        raise ValueError("signal.index must exactly match prices.index.")
    if set(np.unique(sig.dropna().values)).difference({-1, 0, 1}):
        raise ValueError("signal must only contain -1 (short), 0 (flat), +1 (long).")


# ------------------------- Sizing primitives -------------------------

def unit_neutral_quantities(price_x: float, price_y: float, beta: float) -> Tuple[float, float]:
    """
    One 'unit' of the pair: +1 share of X and +beta shares of Y.
    We return *absolute* quantities (positive numbers). Apply signs later based on signal.
    """
    if price_x <= 0 or price_y <= 0:
        raise ValueError("Prices must be positive.")
    if not np.isfinite(beta):
        raise ValueError("beta must be finite.")
    qx = 1.0
    qy = float(beta)
    return qx, qy


def dollar_neutral_quantities(price_x: float, price_y: float, beta: float, leg_capital: float) -> Tuple[float, float]:
    """
    Dollar-neutral sizing that respects hedge ratio beta.
    Put 'leg_capital' notional on the X leg and 'beta * leg_capital' on the Y leg.
    Return *absolute* share counts; apply signs with the signal later.
    """
    if min(price_x, price_y) <= 0:
        raise ValueError("Prices must be positive.")
    if leg_capital <= 0:
        raise ValueError("leg_capital must be positive.")
    if not np.isfinite(beta):
        raise ValueError("beta must be finite.")

    qx = leg_capital / price_x               # shares of X
    qy = (beta * leg_capital) / price_y      # shares of Y
    return float(qx), float(qy)


# ------------------------- Volatility targeting -------------------------

def _pair_pnl_per_unit(px: pd.DataFrame, beta: float, mode: str) -> pd.Series:
    """
    Build a synthetic daily P&L series for ONE 'unit' of the pair.
    - mode == 'dollar': one unit means leg_capital=1 on X and beta*1 on Y.
      That implies qx=1/px_x, qy=beta/px_y (recomputed each day) — which is
      equivalent to evaluating today's P&L using yesterday->today price changes.
    - mode == 'unit'  : one unit means 1 share X vs beta shares Y.

    Returns a Series of daily P&L for the long-pair orientation:
        dPNL_unit = (qx*dPx) - (qy*dPy)
    """
    _validate_prices(px)
    if mode not in {"dollar", "unit"}:
        raise ValueError("mode must be 'dollar' or 'unit'.")

    x = px.iloc[:, 0]
    y = px.iloc[:, 1]
    dPx = x.diff()
    dPy = y.diff()

    if mode == "dollar":
        # per-$1 leg capital on X (and beta*$1 on Y)
        qx = 1.0 / x.shift(1)            # shares per $1 on X (yesterday’s price for no look-ahead)
        qy = beta / y.shift(1)           # shares per $beta on Y
    else:
        # fixed shares per unit
        qx = pd.Series(1.0, index=x.index)
        qy = pd.Series(float(beta), index=y.index)

    dPNL_unit = qx * dPx - qy * dPy
    dPNL_unit.name = "dPNL_unit"
    return dPNL_unit


def estimate_pair_vol(
    px: pd.DataFrame,
    beta: float,
    mode: str = "dollar",
    lookback: int = 60
) -> pd.Series:
    """
    Rolling estimate of daily *dollar* volatility for ONE unit of the pair.
    Uses a 1-bar shift on quantities for 'dollar' mode to avoid look-ahead.
    Returns a Series 'vol_unit' (rolling std of dPNL_unit).
    """
    dPNL_unit = _pair_pnl_per_unit(px, beta=beta, mode=mode)
    vol = dPNL_unit.rolling(lookback, min_periods=lookback).std(ddof=1)
    vol.name = "vol_unit"
    # shift by 1 day so today's capital decision uses info up to t-1
    return vol.shift(1)


def capital_for_target_vol(
    px: pd.DataFrame,
    beta: float,
    target_daily_vol: float = 0.01,
    mode: str = "dollar",
    lookback: int = 60,
    cap_bounds: Tuple[float, float] = (1_000.0, 1_000_000.0)
) -> pd.Series:
    """
    Compute the leg_capital (for 'dollar' mode) or 'unit multiplier' (for 'unit' mode)
    that would achieve approximately 'target_daily_vol' *equity* volatility for the pair.

    For 'dollar' mode:
      - If vol_unit[t] is the std of dPNL when leg_capital=1,
        then leg_capital[t] ≈ target_vol / vol_unit[t].

    For 'unit' mode:
      - We return a scalar multiplier of the unit position sizing.

    Returns a Series aligned to px.index, shifted (no look-ahead), and clipped to cap_bounds.
    """
    if target_daily_vol <= 0:
        raise ValueError("target_daily_vol must be positive.")
    vol_unit = estimate_pair_vol(px, beta=beta, mode=mode, lookback=lookback)

    # Avoid division by zero / NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = target_daily_vol / vol_unit

    # Clip to sane bounds
    scale = scale.clip(lower=cap_bounds[0], upper=cap_bounds[1])
    scale.name = "capital_scale"
    return scale


# ------------------------- Daily quantity schedules -------------------------

@dataclass(frozen=True)
class SizingConfig:
    mode: str = "dollar"                  # 'dollar' or 'unit'
    base_leg_capital: float = 10_000.0    # used only for 'dollar' mode if no vol-targeting
    target_daily_vol: Optional[float] = None  # e.g., 0.01 for 1%/day. If None, disable.
    lookback: int = 60                    # for volatility estimate
    cap_bounds: Tuple[float, float] = (1_000.0, 1_000_000.0)


def quantity_schedule(
    px: pd.DataFrame,
    beta: float,
    config: SizingConfig = SizingConfig()
) -> pd.DataFrame:
    """
    Produce a per-day schedule of *absolute* quantities (no sign) to hold
    when a position is active: columns ['qx', 'qy'].

    If config.target_daily_vol is set:
      - In 'dollar' mode, this function computes a *daily leg_capital*
        using volatility targeting (shifted by 1 bar).
      - In 'unit' mode, it returns a unit multiplier.

    Returns a DataFrame with the same index as px, columns ['qx', 'qy'].
    You apply the TRADE DIRECTION (+/-) later using your signal.
    """
    _validate_prices(px)

    x = px.iloc[:, 0]
    y = px.iloc[:, 1]
    idx = px.index

    if config.target_daily_vol:
        scale = capital_for_target_vol(
            px, beta=beta, target_daily_vol=config.target_daily_vol,
            mode=config.mode, lookback=config.lookback, cap_bounds=config.cap_bounds
        )
    else:
        # constant scale
        scale = pd.Series(
            config.base_leg_capital if config.mode == "dollar" else 1.0,
            index=idx, name="capital_scale"
        )

    qx = pd.Series(index=idx, dtype=float)
    qy = pd.Series(index=idx, dtype=float)

    if config.mode == "dollar":
        # per-day quantities based on that day's prices and scaled leg_capital
        qx = scale / x
        qy = (beta * scale) / y
    else:
        # 'unit' mode: 1 share X, beta shares Y, then scale = unit multiplier
        qx = 1.0 * scale
        qy = float(beta) * scale

    out = pd.DataFrame({"qx": qx, "qy": qy})
    return out


def signed_quantities_from_signal(
    px: pd.DataFrame,
    signal: pd.Series,
    beta: float,
    config: SizingConfig = SizingConfig()
) -> pd.DataFrame:
    """
    Combine the quantity schedule with a trading signal {-1,0,+1}
    to yield *signed* share quantities per day.

    Convention:
      signal = +1  -> LONG PAIR  -> +qx on X, -qy on Y
      signal = -1  -> SHORT PAIR -> -qx on X, +qy on Y
      signal = 0   -> FLAT       ->  0, 0

    Returns DataFrame with columns ['qx_signed', 'qy_signed'].
    """
    _validate_prices(px)
    _validate_signal(signal, px.index)

    abs_qty = quantity_schedule(px, beta=beta, config=config)
    s = signal.reindex(px.index).fillna(0.0)

    qx_signed = s * abs_qty["qx"]
    qy_signed = -s * abs_qty["qy"]  # opposite sign to maintain the pair structure

    out = pd.DataFrame({"qx_signed": qx_signed, "qy_signed": qy_signed})
    return out
