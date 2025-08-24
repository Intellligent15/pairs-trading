# data/polygon_data.py
from __future__ import annotations

import os
import time
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from keys import POLYGON_API_KEY  # Ensure this is set in your environment

import requests
import pandas as pd

# -------- Helpers --------
# Normalizes date inputs into strings Polygon accepts
def _datestr(d: Optional[str | dt.date]) -> str:
    if d is None:
        return dt.date.today().isoformat()
    if isinstance(d, dt.date):
        return d.isoformat()
    return str(d)

@dataclass
class PolygonClient:
    api_key: str = POLYGON_API_KEY
    base_v2: str = "https://api.polygon.io/v2"
    base_v3: str = "https://api.polygon.io/v3"
    timeout: int = 20
    max_retries: int = 3
    backoff_seconds: float = 1.5

    def __post_init__(self):
        self.session = requests.Session()
        self.session.params = {"apiKey": self.api_key}

    # ---- Low-level request with simple retry/backoff ----
    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        for attempt in range(1, self.max_retries + 1):
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 429:
                # rate-limited -> backoff and retry
                time.sleep(self.backoff_seconds * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        # Final try without 429 handled
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ---- Reference: is this ticker an ETF? ----
    def is_etf(self, ticker: str) -> bool:
        """
        Uses v3 reference /tickers/{ticker} to verify type == 'ETF'.
        """
        t = ticker.upper().strip()
        url = f"{self.base_v3}/reference/tickers/{t}"
        data = self._get(url)
        result = data.get("results") or {}
        # Polygon returns fields like 'type' (e.g., 'ETF'), 'asset_class' (e.g., 'stocks')
        return (result.get("type") or "").upper() == "ETF"

    # ---- Daily aggregates (adjusted close) ----
    def fetch_daily_adjusted_closes(
        self,
        ticker: str,
        start: str | dt.date,
        end: str | dt.date,
        adjusted: bool = True,
        limit: int = 50000,
    ) -> pd.Series:
        """
        Fetch daily bars from v2 aggregates, returns a pandas Series of adjusted close.
        Index is timezone-naive date (UTC->date).
        """
        t = ticker.upper().strip()
        start_s = _datestr(start)
        end_s = _datestr(end)

        url = f"{self.base_v2}/aggs/ticker/{t}/range/1/day/{start_s}/{end_s}"
        params = {
            "adjusted": "true" if adjusted else "false",
            "sort": "asc",
            "limit": limit,
        }
        data = self._get(url, params=params)

        results: List[Dict[str, Any]] = data.get("results") or []
        if not results:
            # Return empty series with proper dtype
            return pd.Series(name=t, dtype="float64")

        # 't' = timestamp in ms; 'c' = close; Polygon already applies adj when adjusted=true
        dates = [dt.datetime.utcfromtimestamp(bar["t"] / 1000).date() for bar in results]
        closes = [float(bar["c"]) for bar in results]
        s = pd.Series(closes, index=pd.to_datetime(dates), name=t)
        s.index = s.index.tz_localize(None)  # ensure tz-naive
        return s

    # ---- High-level: validate both tickers are ETFs, pull aligned closes ----
    def get_aligned_etf_closes(
        self,
        t1: str,
        t2: str,
        start: str | dt.date,
        end: str | dt.date,
        zfill: bool = False,
    ) -> pd.DataFrame:
        """
        Validates ETF type for both tickers, fetches adjusted closes, and returns an aligned DataFrame.
        - If zfill=True, forward-fills missing values after concatenation (rare for ETFs).
        """
        if not self.is_etf(t1):
            raise ValueError(f"{t1} is not classified by Polygon as an ETF.")
        if not self.is_etf(t2):
            raise ValueError(f"{t2} is not classified by Polygon as an ETF.")

        s1 = self.fetch_daily_adjusted_closes(t1, start, end, adjusted=True)
        s2 = self.fetch_daily_adjusted_closes(t2, start, end, adjusted=True)

        df = pd.concat([s1, s2], axis=1)
        # Drop rows with any NA by default to ensure alignment; ETFs usually have complete data
        if zfill:
            df = df.ffill().dropna(how="any")
        else:
            df = df.dropna(how="any")

        # Name columns cleanly and ensure Date index
        df.index.name = "Date"
        df.columns = [t1.upper(), t2.upper()]
        return df
