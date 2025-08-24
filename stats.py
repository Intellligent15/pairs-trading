# stats.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import adfuller


# --------------------- Validation ---------------------

def _validate_log_prices(lp: pd.DataFrame) -> None:
    """
    Ensure lp is a 2-column DataFrame of log prices, indexed by ascending dates,
    with no duplicate dates and finite values.
    """
    if not isinstance(lp, pd.DataFrame):
        raise TypeError("lp must be a pandas DataFrame with two columns (log prices for two ETFs).")
    if lp.shape[1] != 2:
        raise ValueError(f"Expected exactly 2 columns, got {lp.shape[1]}.")
    if lp.index.has_duplicates:
        raise ValueError("Index has duplicate dates. De-duplicate before calling.")
    if not lp.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted ascending by date.")
    if not np.isfinite(lp.values).all():
        raise ValueError("lp contains non-finite values (NaN/inf). Clean the data first.")


# --------------------- Results container ---------------------

@dataclass(frozen=True)
class EngleGrangerResult:
    """
    Container for Engle-Granger outputs.
    """
    alpha: float               # intercept from OLS
    beta: float                # hedge ratio (slope) from OLS
    adf_pvalue: float          # ADF p-value on residuals (null: unit root)
    adf_stat: float            # ADF test statistic
    adf_crit: Dict[str, float] # Critical values (e.g., {'1%': -3.43, '5%': -2.86, ...})
    train_start: pd.Timestamp  # first date used in the regression window
    train_end: pd.Timestamp    # last date used in the regression window
    nobs: int                  # number of observations used in OLS/ADF

    def cointegrated(self, alpha_level: float = 0.05) -> bool:
        """Convenience: True if p-value < alpha_level."""
        return self.adf_pvalue < alpha_level


# --------------------- Core Engle–Granger ---------------------

def fit_engle_granger(
    log_prices: pd.DataFrame,
    train_window: int = 252,
    adf_regression: str = "n"  # "n" = no constant (recommended for residuals)
) -> Tuple[EngleGrangerResult, pd.Series, pd.Series]:
    """
    Run Engle-Granger two-step test on the last `train_window` rows of `log_prices`.

    Model: X_t = alpha + beta * Y_t + e_t
    - X := first column of `log_prices`
    - Y := second column of `log_prices`
    - OLS fit over the training window
    - ADF test on residuals e_t (by default with regression="n")

    Parameters
    ----------
    log_prices : pd.DataFrame
        2-column DataFrame (date-indexed, ascending) of log prices.
    train_window : int
        Number of most recent observations to use for OLS + ADF.
    adf_regression : {"n", "c", "ct", "ctt"}
        ADF deterministic terms. Residuals are mean-zero by construction, so "n" is typical.

    Returns
    -------
    (result, resid_train, resid_full)
      - result: EngleGrangerResult (alpha, beta, ADF p-value/stat/critvals, window info)
      - resid_train: residuals over the *training* window (used for ADF)
      - resid_full: residuals recomputed over the *entire* `log_prices` using the fitted alpha/beta
    """
    _validate_log_prices(log_prices)

    if len(log_prices) < train_window:
        raise ValueError(f"Need at least train_window={train_window} rows; have {len(log_prices)}.")

    # Define X (dependent) and Y (independent) from the *same* last train_window
    X = log_prices.iloc[-train_window:, 0]
    Y = log_prices.iloc[-train_window:, 1]

    # Add intercept and fit OLS: X = alpha + beta*Y + e
    X_with_const = add_constant(Y)
    ols_fit = OLS(X, X_with_const).fit()
    alpha = float(ols_fit.params["const"])
    beta = float(ols_fit.params[Y.name])

    # Residuals from training period (used for ADF)
    resid_train = ols_fit.resid.copy()
    resid_train.name = "resid_train"

    # ADF on residuals (null: unit root; reject -> stationary -> cointegration)
    adf_out = adfuller(resid_train.values, regression=adf_regression, autolag="AIC")
    adf_stat, adf_p, used_lag, nobs, crit_vals, icbest = adf_out

    # Residuals over the entire series using the trained alpha/beta
    resid_full = compute_spread(log_prices, alpha=alpha, beta=beta)
    resid_full.name = "resid_full"

    # Build result object
    result = EngleGrangerResult(
        alpha=alpha,
        beta=beta,
        adf_pvalue=float(adf_p),
        adf_stat=float(adf_stat),
        adf_crit={k: float(v) for k, v in crit_vals.items()},
        train_start=X.index[0],
        train_end=X.index[-1],
        nobs=int(nobs),
    )
    return result, resid_train, resid_full


# --------------------- Helpers ---------------------

def compute_spread(log_prices: pd.DataFrame, alpha: float, beta: float) -> pd.Series:
    """
    Compute residuals/spread S_t = X_t - (alpha + beta * Y_t)
    from full log price series using provided alpha/beta.
    """
    _validate_log_prices(log_prices)
    X = log_prices.iloc[:, 0]
    Y = log_prices.iloc[:, 1]
    spread = X - (alpha + beta * Y)
    spread.name = "spread"
    return spread


def refit_on_window(
    log_prices: pd.DataFrame,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    adf_regression: str = "n"
) -> Tuple[EngleGrangerResult, pd.Series, pd.Series]:
    """
    Fit Engle-Granger on an explicit [start:end] slice (inclusive by labels).
    Useful for walk-forward or diagnostics.
    """
    _validate_log_prices(log_prices)

    # Label-based slicing; if None, pandas will default to full range.
    lp_win = log_prices.loc[start:end]
    if lp_win.shape[0] < 10:
        raise ValueError(f"Window too small: need at least ~10 rows for a meaningful OLS/ADF; got {lp_win.shape[0]}.")

    X = lp_win.iloc[:, 0]
    Y = lp_win.iloc[:, 1]

    X_with_const = add_constant(Y)
    ols_fit = OLS(X, X_with_const).fit()
    alpha = float(ols_fit.params["const"])
    beta = float(ols_fit.params[Y.name])

    resid_train = ols_fit.resid.copy()
    resid_train.name = "resid_train"

    adf_out = adfuller(resid_train.values, regression=adf_regression, autolag="AIC")
    adf_stat, adf_p, used_lag, nobs, crit_vals, icbest = adf_out

    resid_full = compute_spread(log_prices, alpha=alpha, beta=beta)
    resid_full.name = "resid_full"

    result = EngleGrangerResult(
        alpha=alpha,
        beta=beta,
        adf_pvalue=float(adf_p),
        adf_stat=float(adf_stat),
        adf_crit={k: float(v) for k, v in crit_vals.items()},
        train_start=X.index[0],
        train_end=X.index[-1],
        nobs=int(nobs),
    )
    return result, resid_train, resid_full
