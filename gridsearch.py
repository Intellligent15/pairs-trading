# gridsearch.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from adaptive import AdaptiveConfig, rolling_refit_adaptive
from rules import band_signals
from sizing import SizingConfig
from backtest import BacktestConfig, backtest_pair


@dataclass(frozen=True)
class Grid:
    enter: Iterable[float] = (1.5, 2.0, 2.5, 3.0)
    exit: Iterable[float] = (0.25, 0.5, 0.75)
    hard_stop_z: Iterable[float] = (3.0, 3.5, 4.0)
    time_stop_mult_hl: Iterable[float] = (2.0, 3.0)  # multiply median half-life


def run_grid(
    closes: pd.DataFrame,
    adaptive_cfg: AdaptiveConfig,
    sizing_cfg: SizingConfig,
    bt_cfg_template: BacktestConfig,
    grid: Grid = Grid(),
) -> pd.DataFrame:
    """
    Run a small grid search over entry/exit bands and stops on top of the adaptive z.
    We compute the adaptive outputs ONCE, then vary only rules/stops.

    time_stop_bars is derived as: int(mult * median_half_life), clipped to [5, 60].

    Returns a DataFrame with one row per combo and key stats.
    """
    # 1) Compute adaptive spread/z once
    ao = rolling_refit_adaptive(closes, cfg=adaptive_cfg)

    # Use beta_series at each day; for sizing we just need beta to compute quantities.
    # We choose the *same* beta piecewise series, but for backtest we need a single beta value.
    # Practically, we use the *beta in force* on each day; the backtester expects a scalar beta.
    # Easiest workaround: compute quantities from prices directly (quantity_schedule) using that scalar beta.
    # For simplicity here, we use the *median beta* across regimes, which is robust in ETF pairs.
    # If you want exact piecewise beta sizing, we can add that to the backtester later.
    beta_scalar = float(np.nanmedian(ao.beta_series.values))

    # Derive a reasonable time stop from half-lives
    hl_med = float(np.nanmedian(ao.hl_series.values)) if len(ao.hl_series) else np.inf

    rows: List[Dict[str, Any]] = []
    for ent in grid.enter:
        for ex in grid.exit:
            if not (ent > ex >= 0.0):
                continue
            sig = band_signals(ao.z, enter=ent, exit=ex)

            for hs in grid.hard_stop_z:
                for mult in grid.time_stop_mult_hl:
                    if np.isfinite(hl_med):
                        ts = int(max(5, min(60, round(mult * hl_med))))
                    else:
                        ts = 20  # fallback

                    bt_cfg = BacktestConfig(
                        fee_bps=bt_cfg_template.fee_bps,
                        slip_bps=bt_cfg_template.slip_bps,
                        short_borrow_bps_per_year=bt_cfg_template.short_borrow_bps_per_year,
                        starting_equity=bt_cfg_template.starting_equity,
                        hard_stop_z=hs,
                        time_stop_bars=ts,
                        rebalance=bt_cfg_template.rebalance,
                    )

                    bt_df, stats = backtest_pair(
                        px=closes,
                        z=ao.z,
                        signal=sig,
                        beta=beta_scalar,
                        sizing_cfg=sizing_cfg,
                        bt_cfg=bt_cfg,
                    )

                    rows.append({
                        "enter": ent,
                        "exit": ex,
                        "hard_stop_z": hs,
                        "time_stop_bars": ts,
                        "beta_used": beta_scalar,
                        **stats
                    })

    out = pd.DataFrame(rows)
    # Sort best first by Sharpe, then by drawdown (less negative is better), then trades
    if not out.empty:
        out = out.sort_values(["sharpe", "max_drawdown", "trades"], ascending=[False, False, False]).reset_index(drop=True)
    return out
