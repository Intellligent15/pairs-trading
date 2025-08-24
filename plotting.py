# plotting.py
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

def _event_indices(df: pd.DataFrame) -> tuple[pd.Index, pd.Index, pd.Index]:
    """
    Return (enter_idx, exit_idx, stop_like_idx) from the backtest timeline.
    """
    events = df["event"].fillna("")
    enter_idx = events[events == "enter"].index
    exit_idx  = events[events == "exit"].index
    # hard/time stop variants for visibility
    stop_idx  = events[events.isin(["stop", "time_exit"])].index
    return enter_idx, exit_idx, stop_idx


def plot_equity(bt_df: pd.DataFrame, title: str = "Equity Curve") -> None:
    """
    Plot equity over time. Marks entries/exits as vertical lines.
    """
    if "equity" not in bt_df:
        raise ValueError("bt_df must contain an 'equity' column.")
    enter_idx, exit_idx, stop_idx = _event_indices(bt_df)

    plt.figure()
    bt_df["equity"].plot()
    for t in enter_idx:
        plt.axvline(t, linestyle="--", linewidth=1, alpha=0.7)
    for t in exit_idx:
        plt.axvline(t, linestyle=":", linewidth=1, alpha=0.7)
    for t in stop_idx:
        plt.axvline(t, linestyle="-.", linewidth=1, alpha=0.7)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()


def plot_z_with_bands(z: pd.Series, enter: float = 2.0, exit: float = 0.5,
                      events_df: pd.DataFrame | None = None,
                      title: str = "Z-Score with Bands") -> None:
    """
    Plot the z-score and draw horizontal entry/exit bands.
    Optionally mark entry/exit timestamps from an events table.
    """
    if not isinstance(z, pd.Series):
        raise TypeError("z must be a pandas Series.")

    plt.figure()
    z.plot()
    # bands
    for lvl in (+enter, -enter, +exit, -exit, 0.0):
        plt.axhline(lvl, linestyle="--", linewidth=1, alpha=0.7)

    # optional event markers
    if events_df is not None and "time" in events_df and "event" in events_df:
        for _, row in events_df.iterrows():
            t = row["time"]
            e = row["event"]
            if e == "enter_long" or e == "enter_short":
                plt.axvline(t, linestyle=":", linewidth=1, alpha=0.7)
            elif e in ("exit_long", "exit_short"):
                plt.axvline(t, linestyle="-.", linewidth=1, alpha=0.7)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Z-score")
    plt.tight_layout()
    plt.show()


def plot_positions(bt_df: pd.DataFrame, title: str = "Signed Positions (shares)") -> None:
    """
    Plot signed share holdings qx (X leg) and qy (Y leg) over time.
    """
    if not {"qx", "qy"}.issubset(bt_df.columns):
        raise ValueError("bt_df must contain 'qx' and 'qy' columns.")

    plt.figure()
    bt_df["qx"].plot(label="qx")
    bt_df["qy"].plot(label="qy")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Shares")
    plt.legend()
    plt.tight_layout()
    plt.show()
