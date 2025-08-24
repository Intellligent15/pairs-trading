# backtest.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

from sizing import SizingConfig, quantity_schedule


# --------------------- Config ---------------------

@dataclass(frozen=True)
class BacktestConfig:
    fee_bps: float = 1.0                 # per-leg transaction fee/slippage (bps) per share change
    slip_bps: float = 0.0                # extra slippage bps (added to fee_bps)
    short_borrow_bps_per_year: float = 50.0  # borrow cost on the short leg (annualized bps)
    starting_equity: float = 100_000.0
    hard_stop_z: Optional[float] = 3.5   # None to disable
    time_stop_bars: Optional[int] = 20   # None to disable
    rebalance: bool = False              # adjust quantities daily to target schedule?
    # notes:
    # - If rebalance=False: shares are set at entry and held until exit.
    # - If rebalance=True : daily share changes are traded to match schedule (incurs fees).


# --------------------- Validation helpers ---------------------

def _validate_inputs(
    px: pd.DataFrame,
    z: pd.Series,
    signal: pd.Series,
) -> None:
    if not isinstance(px, pd.DataFrame) or px.shape[1] != 2:
        raise ValueError("px must be a 2-column DataFrame of prices (X first, Y second).")
    if (px <= 0).any().any():
        raise ValueError("px must be strictly positive.")
    if px.index.has_duplicates or not px.index.is_monotonic_increasing:
        raise ValueError("px index must be unique and ascending.")
    for name, s in {"z": z, "signal": signal}.items():
        if not isinstance(s, pd.Series):
            raise TypeError(f"{name} must be a pandas Series.")
        if s.index.has_duplicates or not s.index.is_monotonic_increasing:
            raise ValueError(f"{name} index must be unique and ascending.")


# --------------------- Core backtest ---------------------

def backtest_pair(
    px: pd.DataFrame,
    z: pd.Series,
    signal: pd.Series,
    beta: float,
    sizing_cfg: SizingConfig,
    bt_cfg: BacktestConfig = BacktestConfig(),
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Backtest the pair using:
      - prices px (2 cols: X first, Y second),
      - z-score (no look-ahead; e.g., rolling stats already shifted),
      - raw 'signal' in {-1,0,+1} (long/flat/short), computed with info through t-1,
      - alpha/beta already fitted elsewhere; we only need beta for sizing,
      - SizingConfig for abs quantities, BacktestConfig for costs & risk controls.

    Execution rule:
      - Use 'exec_signal' := signal.shift(1).fillna(0)  (decision at t-1, trade at t close).

    Returns:
      - df: detailed timeline (equity, pnl, positions, events)
      - stats: summary metrics (sharpe, max_drawdown, cagr, win_rate, exposure, trades)
    """
    _validate_inputs(px, z, signal)

    # Align and drop rows without complete info
    df = pd.concat(
        [
            px.rename(columns={px.columns[0]: "X", px.columns[1]: "Y"}),
            z.rename("z"),
            signal.rename("signal"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    # Execution signal - trade today using yesterday's computed signal
    df["exec_signal"] = df["signal"].shift(1).fillna(0.0)

    # Precompute target abs-quantities schedule (for optional rebalancing)
    abs_qty = quantity_schedule(px=df[["X", "Y"]], beta=beta, config=sizing_cfg)
    abs_qty = abs_qty.reindex(df.index)
    df["qx_target"] = abs_qty["qx"]
    df["qy_target"] = abs_qty["qy"]

    # State variables
    equity = bt_cfg.starting_equity
    pos = 0                 # -1 short pair, 0 flat, +1 long pair
    days_in_trade = 0
    qx = 0.0                # signed shares held on X
    qy = 0.0                # signed shares held on Y
    entry_price_x = np.nan  # for reference only
    entry_price_y = np.nan

    # Outputs
    df["qx"] = 0.0
    df["qy"] = 0.0
    df["pnl_day"] = 0.0
    df["trade_cost"] = 0.0
    df["borrow_cost"] = 0.0
    df["equity"] = np.nan
    df["event"] = ""        # 'enter', 'exit', 'rebalance', 'stop', 'time_exit', ''

    # Cost model
    txn_bps = (bt_cfg.fee_bps + bt_cfg.slip_bps) / 1e4
    borrow_daily_rate = (bt_cfg.short_borrow_bps_per_year / 1e4) / 252.0

    X = df["X"].values
    Y = df["Y"].values
    Z = df["z"].values
    exec_sig = df["exec_signal"].values
    idx = df.index

    for i in range(1, len(df)):
        today = idx[i]

        # 1) Decide desired state for today (based on exec_signal)
        desired = int(exec_sig[i])

        # Optional: risk controls override desired state
        if bt_cfg.hard_stop_z is not None and pos != 0:
            if abs(Z[i]) >= bt_cfg.hard_stop_z:
                desired = 0  # force exit on hard stop
                reason = "stop"
            else:
                reason = ""
        else:
            reason = ""

        if bt_cfg.time_stop_bars is not None and pos != 0 and days_in_trade >= bt_cfg.time_stop_bars:
            desired = 0  # force exit on time stop
            reason = "time_exit"

        trade_cost = 0.0
        event = ""

        # 2) Open/close/rebalance at today's close
        if pos == 0 and desired != 0:
            # ENTER new position at today's close
            target_abs_qx = df["qx_target"].iloc[i]
            target_abs_qy = df["qy_target"].iloc[i]
            qx_new = desired * target_abs_qx       # + for long pair, - for short
            qy_new = -desired * target_abs_qy      # opposite leg

            # Shares to trade = new - old
            dqx = qx_new - qx
            dqy = qy_new - qy
            notional_traded = abs(dqx) * X[i] + abs(dqy) * Y[i]
            trade_cost = txn_bps * notional_traded

            qx, qy = qx_new, qy_new
            pos = desired
            days_in_trade = 0
            entry_price_x, entry_price_y = X[i], Y[i]
            event = "enter"

        elif pos != 0 and desired == 0:
            # EXIT current position at today's close (rule exit / stop / time exit)
            dqx = -qx
            dqy = -qy
            notional_traded = abs(dqx) * X[i] + abs(dqy) * Y[i]
            trade_cost = txn_bps * notional_traded

            qx, qy = 0.0, 0.0
            pos = 0
            days_in_trade = 0
            event = "exit" if reason == "" else reason

        elif pos != 0 and desired == pos and bt_cfg.rebalance:
            # REBALANCE to today's target quantities (optional)
            target_abs_qx = df["qx_target"].iloc[i]
            target_abs_qy = df["qy_target"].iloc[i]
            qx_new = desired * target_abs_qx
            qy_new = -desired * target_abs_qy

            dqx = qx_new - qx
            dqy = qy_new - qy
            notional_traded = abs(dqx) * X[i] + abs(dqy) * Y[i]
            trade_cost = txn_bps * notional_traded

            qx, qy = qx_new, qy_new
            event = "rebalance"
            days_in_trade += 1

        elif pos != 0:
            # HOLD position, no share change
            days_in_trade += 1

        # 3) Daily P&L from price moves (yesterday->today)
        dPx = X[i] - X[i - 1]
        dPy = Y[i] - Y[i - 1]
        pnl_mark = qx * dPx + qy * dPy

        # 4) Borrow cost on short leg's notional (if in position)
        short_notional = 0.0
        if pos != 0:
            # Short leg is the one with negative signed shares
            short_notional = 0.0
            if qx < 0:
                short_notional += abs(qx) * X[i]
            if qy < 0:
                short_notional += abs(qy) * Y[i]
        borrow_cost = borrow_daily_rate * short_notional

        # 5) Update equity
        pnl_day = pnl_mark - trade_cost - borrow_cost
        equity += pnl_day

        # 6) Save
        df.at[today, "qx"] = qx
        df.at[today, "qy"] = qy
        df.at[today, "pnl_day"] = pnl_day
        df.at[today, "trade_cost"] = trade_cost
        df.at[today, "borrow_cost"] = borrow_cost
        df.at[today, "equity"] = equity
        if event:
            df.at[today, "event"] = event

    # Finalize returns & stats
    df["equity"] = df["equity"].ffill().fillna(bt_cfg.starting_equity)
    df["ret_day"] = df["equity"].pct_change().fillna(0.0)

    stats = _summary_stats(df, bt_cfg.starting_equity)
    return df, stats


# --------------------- Stats ---------------------

def _summary_stats(df: pd.DataFrame, starting_equity: float) -> Dict[str, float]:
    """
    Compute common performance stats from the backtest timeline.
    """
    ann = 252.0
    r = df["ret_day"]
    mu = r.mean() * ann
    sig = r.std(ddof=1) * np.sqrt(ann)
    sharpe = mu / sig if sig > 0 else np.nan

    # Max drawdown (on equity)
    eq = df["equity"]
    roll_max = eq.cummax()
    dd = (eq / roll_max) - 1.0
    max_dd = dd.min()

    # CAGR
    n_days = max(len(df), 1)
    cagr = (df["equity"].iloc[-1] / starting_equity) ** (ann / n_days) - 1.0

    # Trade counting & win rate
    events = df["event"].fillna("")
    entries = (events == "enter").sum()
    exits = ((events == "exit") | (events == "stop") | (events == "time_exit")).sum()
    trades = int(min(entries, exits))

    # Approx win rate: compare cumulative equity change over each completed trade
    win_rate = np.nan
    if trades > 0:
        # Build per-trade P&L
        pnl_list = []
        in_trade = False
        eq_start = 0.0
        for _, row in df.iterrows():
            if row["event"] == "enter" and not in_trade:
                in_trade = True
                eq_start = row["equity"]
            elif row["event"] in ("exit", "stop", "time_exit") and in_trade:
                in_trade = False
                pnl_list.append(row["equity"] - eq_start)
        if pnl_list:
            pnl_arr = np.array(pnl_list, dtype=float)
            win_rate = float((pnl_arr > 0).mean())

    exposure = float((df["qx"].abs() + df["qy"].abs() > 0).mean())  # % of days in a position

    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "cagr": float(cagr),
        "exposure": exposure,
        "trades": float(trades),
        "win_rate": float(win_rate) if win_rate == win_rate else np.nan,
    }
