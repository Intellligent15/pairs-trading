# main.py
from __future__ import annotations

import sys
import numpy as np
from pprint import pprint

# --- project modules ---
from data.polygon_data import PolygonClient
from analysis import prep_pair
from stats import fit_engle_granger
from signals import rolling_zscore
from rules import band_signals, entry_exit_table
from sizing import SizingConfig
from backtest import BacktestConfig, backtest_pair
from plotting import plot_equity, plot_z_with_bands, plot_positions

# adaptive + grid
from adaptive import AdaptiveConfig, rolling_refit_adaptive
from gridsearch import Grid, run_grid


# ===================== User-configurable knobs =====================

# Pair (X first, Y second)
TICKER_X = "QQQ"
TICKER_Y = "SPY"

# Data range
START_DATE = "2023-01-01"
END_DATE = None  # None = today

# Fixed pipeline params
RET_CORR_LOOKBACK = 120   # step 2 correlation lookback (returns)
TRAIN_WINDOW = 252        # step 3 OLS/ADF training window (log prices)
Z_WINDOW = 60             # step 5 rolling z-score window (fixed pipeline)

# Entry/exit bands (used for both fixed and adaptive unless overridden)
ENTER_Z = 2.0
EXIT_Z = 0.5

# Sizing
SIZING_CFG = SizingConfig(
    mode="dollar",           # "dollar" or "unit"
    base_leg_capital=10_000.0,
    target_daily_vol=None,   # e.g., 0.01 for ~1%/day; None disables vol targeting
    lookback=60
)

# Backtest & risk controls
BT_CFG = BacktestConfig(
    fee_bps=1.0,
    slip_bps=0.5,
    short_borrow_bps_per_year=50.0,
    starting_equity=100_000.0,
    hard_stop_z=3.5,        # exit if |z| >= this while in trade
    time_stop_bars=20,      # force exit after N bars
    rebalance=False         # set True to rebalance daily to target quantities
)

# Adaptive (rolling re-fit + half-life)
ADAPTIVE_CFG = AdaptiveConfig(
    train_window=252,
    refit_every=10,
    z_window_mult=5.0,      # z-window ≈ mult * half-life
    z_window_min=20,
    z_window_max=120,
    adf_regression="n",
    fallback_z_window=60
)

# Grid search (around adaptive z)
GRID = Grid(
    enter=(1.5, 2.0, 2.5, 3.0),
    exit=(0.25, 0.5, 0.75),
    hard_stop_z=(3.0, 3.5, 4.0),
    time_stop_mult_hl=(2.0, 3.0)
)


# ========================= Pipelines =========================

def run_fixed_pipeline(closes):
    """Original fixed-parameter pipeline (steps 0→9)."""
    # Steps 1–2
    log_prices, log_returns, corr_lb = prep_pair(closes, lookback=RET_CORR_LOOKBACK)
    print(f"[Fixed] Return corr ({RET_CORR_LOOKBACK}d): {corr_lb:.3f}")

    # Step 3
    eg_result, resid_train, resid_full = fit_engle_granger(
        log_prices, train_window=TRAIN_WINDOW, adf_regression="n"
    )
    print(f"[Fixed] Beta: {eg_result.beta:.4f} | ADF p: {eg_result.adf_pvalue:.4f} "
          f"| Cointegrated@5%? {eg_result.cointegrated(0.05)}")

    # Step 4: spread
    spread = resid_full

    # Step 5: z-score (no look-ahead)
    z = rolling_zscore(spread, window=Z_WINDOW, avoid_lookahead=True)

    # Step 6: signals
    signal = band_signals(z, enter=ENTER_Z, exit=EXIT_Z)

    # Steps 8–9: backtest
    bt_df, stats = backtest_pair(
        px=closes, z=z, signal=signal, beta=eg_result.beta,
        sizing_cfg=SIZING_CFG, bt_cfg=BT_CFG
    )

    print("\n=== [Fixed] Summary stats ===")
    pprint(stats)
    print("\n=== [Fixed] Tail (equity, z, exec_signal, qx, qy, event) ===")
    print(bt_df[["equity", "z", "exec_signal", "qx", "qy", "event"]].tail())

    # Plots
    events = entry_exit_table(signal)
    plot_equity(bt_df, title=f"[Fixed] Equity Curve ({TICKER_X}/{TICKER_Y})")
    plot_z_with_bands(z, enter=ENTER_Z, exit=EXIT_Z,
                      events_df=events, title=f"[Fixed] Z-Score with Bands ({TICKER_X}/{TICKER_Y})")
    plot_positions(bt_df, title=f"[Fixed] Signed Positions ({TICKER_X}/{TICKER_Y})")

    return bt_df, stats


def run_adaptive_pipeline(closes):
    """Adaptive pipeline: rolling re-fit + half-life-based z-window, then backtest."""
    ao = rolling_refit_adaptive(closes, cfg=ADAPTIVE_CFG)

    # Print last few regimes for visibility
    print("\nAdaptive regimes (last 5):")
    for (s, e, a, b, hl, zw) in ao.regimes[-5:]:
        print(f"  {s.date()} → {e.date()} | alpha={a:.4f} beta={b:.4f} "
              f"half-life={hl:.2f} z-window={zw}")

    # Signals on adaptive z
    signal = band_signals(ao.z, enter=ENTER_Z, exit=EXIT_Z)

    # For sizing, use a robust single beta (median across regimes); optional: extend backtester to accept beta series.
    beta_used = float(np.nanmedian(ao.beta_series.values))

    bt_df, stats = backtest_pair(
        px=closes, z=ao.z, signal=signal, beta=beta_used,
        sizing_cfg=SIZING_CFG, bt_cfg=BT_CFG
    )

    print("\n=== [Adaptive] Summary stats ===")
    pprint(stats)
    print(f"[Adaptive] beta_used (median of piecewise β): {beta_used:.4f}")
    print("\n=== [Adaptive] Tail (equity, z, exec_signal, qx, qy, event) ===")
    print(bt_df[["equity", "z", "exec_signal", "qx", "qy", "event"]].tail())

    # Plots
    events = entry_exit_table(signal)
    plot_equity(bt_df, title=f"[Adaptive] Equity Curve ({TICKER_X}/{TICKER_Y})")
    plot_z_with_bands(ao.z, enter=ENTER_Z, exit=EXIT_Z,
                      events_df=events, title=f"[Adaptive] Z-Score with Bands ({TICKER_X}/{TICKER_Y})")
    plot_positions(bt_df, title=f"[Adaptive] Signed Positions ({TICKER_X}/{TICKER_Y})")

    return bt_df, stats, ao


def run_adaptive_grid(closes):
    """Run a small grid search of rules on top of the adaptive z."""
    results = run_grid(
        closes=closes,
        adaptive_cfg=ADAPTIVE_CFG,
        sizing_cfg=SIZING_CFG,
        bt_cfg_template=BT_CFG,
        grid=GRID
    )
    print("\n=== Grid Search (top 15 by Sharpe) ===")
    print(results.head(15))
    return results


# ========================= Orchestration =========================

def run_pipeline() -> None:
    client = PolygonClient()

    # Step 0: pull aligned adjusted closes (and ETF validation)
    closes = client.get_aligned_etf_closes(TICKER_X, TICKER_Y, start=START_DATE, end=END_DATE)

    # --- Fixed pipeline ---
    bt_df_fixed, stats_fixed = run_fixed_pipeline(closes)

    # --- Adaptive pipeline ---
    bt_df_ad, stats_ad, ao = run_adaptive_pipeline(closes)

    # --- Grid search on adaptive z ---
    _grid_results = run_adaptive_grid(closes)


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print("❌ Pipeline failed:", e, file=sys.stderr)
        raise