import os
import sys
import traceback
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import numpy as np

# Optional: try to add project directory to path so local modules import correctly
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

# Try importing your existing modules
modules_status = []
def _try_import(name):
    try:
        mod = __import__(name)
        modules_status.append(f"Imported {name}")
        return mod
    except Exception as e:
        modules_status.append(f"Could not import {name}: {e}")
        return None

# Adjust names if your files live in a src/ folder etc.
backtest = _try_import("backtest")
signals = _try_import("signals")
analysis = _try_import("analysis")
adaptive = _try_import("adaptive")
sizing = _try_import("sizing")
rules = _try_import("rules")
stats = _try_import("stats")
plotting = _try_import("plotting")
main_module = _try_import("main")

st.set_page_config(page_title="ETF Pairs Mean-Reversion", layout="wide")

st.title("ETF Pairs Mean-Reversion Backtest")

# with st.expander("Module import status", expanded=False):
#     for line in modules_status:
#         st.write(line)

# Sidebar inputs
st.sidebar.header("Inputs")
ticker_x = st.sidebar.text_input("ETF X (Long on underperformance)", value="QQQ")
ticker_y = st.sidebar.text_input("ETF Y (Short the relative outperformance)", value="SPY")

today = date.today()
default_start = today - timedelta(days=365*2)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=today)

lookback = st.sidebar.number_input("Lookback window (days)", min_value=20, max_value=252*2, value=120, step=5)
entry_z = st.sidebar.number_input("Entry Z-score (open trade when |z| ≥)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
exit_z = st.sidebar.number_input("Exit Z-score (close trade when |z| ≤)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

run_btn = st.sidebar.button("Run backtest")

# placeholder_schema = {
#     "summary": {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "trades": 0, "win_rate": 0.0, "exposure": 0.0},
#     "equity_curve": "DataFrame[Date, equity]",
#     "signals": "DataFrame[Date, z, exec_signal, qty_x, qty_y]",
# }
# with st.expander("Expected output schema (example)", expanded=False):
#     st.json(placeholder_schema)

def simple_demo_pairs(ticker_x: str, ticker_y: str, start: date, end: date, lookback: int, entry_z: float, exit_z: float):
    """
    Fallback demo strategy using yfinance (no dependency on your modules).
    You can delete this once wired to your code.
    """
    import yfinance as yf

    px_x = yf.download(ticker_x, start=start, end=end, progress=False)["Adj Close"].rename(ticker_x)
    px_y = yf.download(ticker_y, start=start, end=end, progress=False)["Adj Close"].rename(ticker_y)
    df = pd.concat([px_x, px_y], axis=1).dropna()
    log_x = np.log(df[ticker_x])
    log_y = np.log(df[ticker_y])

    # OLS hedge ratio via rolling regression (fast approximation)
    # Using rolling beta of X ~ Y — adjust per your project conventions
    roll = lookback
    cov = log_x.rolling(roll).cov(log_y)
    var = log_y.rolling(roll).var()
    beta = (cov / var).dropna()
    beta = beta.reindex(df.index).ffill().bfill()

    spread = log_x - beta * log_y
    spread_mean = spread.rolling(roll).mean()
    spread_std = spread.rolling(roll).std()
    z = (spread - spread_mean) / spread_std
    z = z.dropna()

    # Generate entry/exit signals: long X/short Y when z < -entry, close when |z| <= exit
    pos = 0  # -1 short spread, +1 long spread
    signals = []
    equity = 100000.0
    equity_curve = []
    x_shares = y_shares = 0

    for i in range(len(z)):
        idx = z.index[i]
        zi = z.iloc[i]
        px_i_x = df.loc[idx, ticker_x]
        px_i_y = df.loc[idx, ticker_y]
        beta_i = float(beta.loc[idx])

        signal = 0

        if pos == 0:
            if zi <= -entry_z:
                pos = +1  # long spread: long X, short Y*beta
                signal = +1
                # simple 1 unit notionals
                notional = equity * 0.05
                x_shares = notional / px_i_x
                y_shares = (notional * beta_i) / px_i_y
            elif zi >= entry_z:
                pos = -1  # short spread
                signal = -1
                notional = equity * 0.05
                x_shares = notional / px_i_x
                y_shares = (notional * beta_i) / px_i_y
        else:
            if abs(zi) <= exit_z:
                pos = 0
                signal = 0
                x_shares = 0
                y_shares = 0

        # Mark-to-market PnL (very rough; you should replace with your backtester)
        if pos == +1:
            # long X, short Y*beta
            pnl = x_shares * px_i_x - y_shares * px_i_y
        elif pos == -1:
            # short X, long Y*beta
            pnl = -x_shares * px_i_x + y_shares * px_i_y
        else:
            pnl = 0

        equity_curve.append((idx, equity + pnl))
        signals.append((idx, zi, signal, x_shares if pos!=0 else 0.0, (y_shares if pos!=0 else 0.0) * (-1 if pos==+1 else +1)))

    eq_df = pd.DataFrame(equity_curve, columns=["Date", "equity"]).set_index("Date")
    sig_df = pd.DataFrame(signals, columns=["Date", "z", "exec_signal", "qx", "qy"]).set_index("Date")
    # quick stats
    ret = eq_df["equity"].pct_change().dropna()
    cagr = (eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0]) ** (252/len(eq_df)) - 1 if len(eq_df) > 1 else 0.0
    sharpe = np.sqrt(252) * (ret.mean() / ret.std() if ret.std() != 0 else 0.0)
    mdd = (eq_df["equity"] / eq_df["equity"].cummax() - 1).min()
    trades = int((sig_df["exec_signal"] != 0).sum())
    wins = 0  # placeholder
    exposure = (sig_df["exec_signal"] != 0).mean()

    return {
        "summary": {
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": float(mdd),
            "trades": trades,
            "win_rate": float(wins),
            "exposure": float(exposure),
        },
        "equity_df": eq_df,
        "signals_df": sig_df,
    }

def run_with_your_code(ticker_x, ticker_y, start_date, end_date, lookback, entry_z, exit_z):
    """
    Adaptive pipeline: rolling re-fit of (alpha, beta) + half-life-based z-window.
    Returns the dict schema expected by the Streamlit app.
    """
    # Lazy imports so the app loads even if modules are missing at import time
    from data.polygon_data import PolygonClient
    from adaptive import AdaptiveConfig, rolling_refit_adaptive
    from rules import band_signals
    from sizing import SizingConfig
    from backtest import BacktestConfig, backtest_pair
    import numpy as np
    import pandas as pd

    # Convert Streamlit dates (date objects) to ISO strings for our client
    start_str = start_date.isoformat() if start_date else None
    end_str = end_date.isoformat() if end_date else None

    # ---- Step 0: pull data
    client = PolygonClient()
    closes = client.get_aligned_etf_closes(ticker_x, ticker_y, start=start_str, end=end_str)

    # ---- Adaptive config
    # We’ll still respect the sidebar "lookback" by mapping it to the z-window multiplier
    # approximately: lookback ≈ z_window_mult * half_life  => z_window_mult ≈ lookback / half_life
    # Since half-life varies by regime, we keep our robust defaults and clamp windows.
    ad_cfg = AdaptiveConfig(
        train_window=min(252, len(closes)),
        refit_every=10,
        z_window_mult=5.0,     # half-life multiplier; robust default
        z_window_min=20,
        z_window_max=120,
        adf_regression="n",
        fallback_z_window=int(max(20, min(120, lookback)))  # fall back to the UI lookback if needed
    )

    # ---- Run adaptive rolling re‑fit (no look‑ahead inside)
    ao = rolling_refit_adaptive(closes, cfg=ad_cfg)

    # ---- Signals from adaptive z using the UI thresholds
    signal = band_signals(ao.z, enter=float(entry_z), exit=float(exit_z))

    # ---- Sizing & backtest
    # Use a robust scalar beta for sizing (median of piecewise β); we can extend to β series later.
    beta_used = float(np.nanmedian(ao.beta_series.values))

    sizing_cfg = SizingConfig(
        mode="dollar",
        base_leg_capital=10_000.0,
        target_daily_vol=None,
        lookback=60,
    )
    bt_cfg = BacktestConfig(
        fee_bps=1.0,
        slip_bps=0.5,
        short_borrow_bps_per_year=50.0,
        starting_equity=100_000.0,
        hard_stop_z=3.5,
        time_stop_bars=20,
        rebalance=False,
    )

    bt_df, stats = backtest_pair(
        px=closes,
        z=ao.z,
        signal=signal,
        beta=beta_used,
        sizing_cfg=sizing_cfg,
        bt_cfg=bt_cfg,
    )

    # ---- Shape outputs to the schema the app expects
    eq_df = bt_df[["equity"]].copy()
    sig_df = bt_df[["z", "exec_signal", "qx", "qy"]].copy()
    eq_df.index.name = "Date"
    sig_df.index.name = "Date"

    return {
        "summary": {
            "cagr": float(stats.get("cagr", 0.0)),
            "sharpe": float(stats.get("sharpe", 0.0)),
            "max_drawdown": float(stats.get("max_drawdown", 0.0)),
            "trades": int(stats.get("trades", 0)),
            "win_rate": float(stats.get("win_rate", 0.0)),
            "exposure": float(stats.get("exposure", 0.0)),
        },
        "equity_df": eq_df,
        "signals_df": sig_df,
    }


if run_btn:
    try:
        try:
            results = run_with_your_code(
                ticker_x, ticker_y, start_date, end_date, lookback, entry_z, exit_z
            )
        except NotImplementedError:
            st.info("Using fallback internal demo (yfinance). To use your code, edit `run_with_your_code()`.")
            results = simple_demo_pairs(ticker_x, ticker_y, start_date, end_date, lookback, entry_z, exit_z)

        summary = results["summary"]
        eq_df = results["equity_df"]
        sig_df = results["signals_df"]

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("CAGR", f"{summary['cagr']:.2%}")
        c2.metric("Sharpe", f"{summary['sharpe']:.2f}")
        c3.metric("Max DD", f"{summary['max_drawdown']:.2%}")
        c4.metric("Trades", f"{summary['trades']}")
        c5.metric("Win rate", f"{summary['win_rate']:.1%}")
        c6.metric("Exposure", f"{summary['exposure']:.1%}")

        st.subheader("Equity Curve")
        st.line_chart(eq_df["equity"])

        st.subheader("Z‑score & Signals")
        st.area_chart(sig_df[["z"]])
        with st.expander("Signals table", expanded=False):
            st.dataframe(sig_df.tail(200))
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.code("".join(traceback.format_exc()))
else:
    st.info("Set your inputs and click **Run backtest** from the sidebar.")
