# rules.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


# --------------------- Validation ---------------------

def _validate_z(z: pd.Series) -> None:
    """
    Ensure 'z' is a 1-D pandas Series, date-indexed, ascending, numeric.
    """
    if not isinstance(z, pd.Series):
        raise TypeError("z must be a pandas Series (rolling z-score over time).")
    if z.index.has_duplicates:
        raise ValueError("z index has duplicate timestamps. De-duplicate before calling.")
    if not z.index.is_monotonic_increasing:
        raise ValueError("z index must be sorted in ascending order.")
    if z.dtype.kind not in {"f", "i"}:
        try:
            z.astype(float)
        except Exception as e:
            raise TypeError("z must be numeric (float or int).") from e


# --------------------- Core: band entry/exit ---------------------

def band_signals(
    z: pd.Series,
    enter: float = 2.0,
    exit: float = 0.5,
    require_exit_cross: bool = False,
) -> pd.Series:
    """
    Convert z-score series into discrete trading signals with hysteresis bands.

    Conventions:
      - +1 = LONG PAIR  (long X / short beta*Y) when z <= -enter
      - -1 = SHORT PAIR (short X / long  beta*Y) when z >= +enter
      -  0 = FLAT when |z| <= exit

    Parameters
    ----------
    z : pd.Series
        Rolling z-score series (no look-ahead already baked in).
    enter : float
        Entry threshold magnitude. Trade when z >= +enter or z <= -enter.
    exit : float
        Exit-to-flat band: flatten when |z| <= exit.
    require_exit_cross : bool
        If True, require z to *cross* the exit band boundary from the
        outside before flattening (filters noise if z hovers exactly at exit).

    Returns
    -------
    signal : pd.Series
        Series in {-1, 0, +1}; stateful, one position at a time.
    """
    _validate_z(z)
    if not (enter > exit >= 0):
        raise ValueError("Require enter > exit >= 0, e.g., enter=2.0, exit=0.5.")

    signal = pd.Series(index=z.index, dtype=float)
    state = 0  # current position: -1 short, 0 flat, +1 long
    prev_z = np.nan

    for ts, val in z.items():
        if np.isnan(val):
            # propagate prior state (no new info)
            signal.loc[ts] = state
            prev_z = val
            continue

        if state == 0:
            # look for entries
            if val <= -enter:
                state = +1  # LONG PAIR
            elif val >= +enter:
                state = -1  # SHORT PAIR

        elif state == +1:
            # in a long-pair; look for exit (reversion toward 0)
            if require_exit_cross:
                # must *cross* from below -exit to above -exit, or from below +exit to <= +exit
                # simpler: flatten only when |z| <= exit and previous |z| > exit
                if abs(val) <= exit and (not np.isnan(prev_z)) and abs(prev_z) > exit:
                    state = 0
            else:
                if abs(val) <= exit:
                    state = 0

        elif state == -1:
            # in a short-pair; look for exit (reversion toward 0)
            if require_exit_cross:
                if abs(val) <= exit and (not np.isnan(prev_z)) and abs(prev_z) > exit:
                    state = 0
            else:
                if abs(val) <= exit:
                    state = 0

        signal.loc[ts] = state
        prev_z = val

    signal.name = "signal"
    return signal


# --------------------- Utilities (nice-to-haves) ---------------------

def signal_changes(signal: pd.Series) -> pd.Series:
    """
    +1 -> 0, 0 -> -1, etc. Detects points in time where the regime changes.
    Useful to derive entry/exit timestamps without scanning the whole series.

    Returns a series where:
      - +1 means we *entered* a long pair at this bar
      - -1 means we *entered* a short pair at this bar
      - +2 means we *exited* a short (since -1 -> 0 => diff = +1, but we normalize exits to +2/-2 below)
      - -2 means we *exited* a long (since +1 -> 0)
    (We normalize exits to +/-2 to distinguish them from entries.)
    """
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series.")
    d = signal.diff()
    d.iloc[0] = signal.iloc[0]  # first bar change = initial state

    # Normalize exits to +/-2 for readability in logs
    # Entry long: 0->+1 => +1
    # Entry short: 0->-1 => -1
    # Exit long:  +1->0 => -1 => map to -2
    # Exit short: -1->0 => +1 => map to +2
    out = d.copy()
    out[(d == -1) & (signal.shift(1) == +1)] = -2  # exit long
    out[(d == +1) & (signal.shift(1) == -1)] = +2  # exit short
    out.name = "signal_change"
    return out


def entry_exit_table(signal: pd.Series) -> pd.DataFrame:
    """
    Produce a tidy table of entry/exit events (timestamps & side).
    This is purely for inspection/debug and doesn’t execute trades.
    """
    ch = signal_changes(signal)
    entries_long = ch[ch == +1].index
    entries_short = ch[ch == -1].index
    exits_long = ch[ch == -2].index
    exits_short = ch[ch == +2].index

    rows = []
    for t in entries_long:
        rows.append({"time": t, "event": "enter_long"})
    for t in entries_short:
        rows.append({"time": t, "event": "enter_short"})
    for t in exits_long:
        rows.append({"time": t, "event": "exit_long"})
    for t in exits_short:
        rows.append({"time": t, "event": "exit_short"})

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df