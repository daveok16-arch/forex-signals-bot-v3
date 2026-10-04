"""
Market Data Providers
=====================
Real OHLCV data acquisition for the forex/metals universe.

Providers (in priority order):
- OANDA REST API (requires OANDA_API_KEY; live/practice)
- Yahoo Finance chart API (no credentials; daily bars)
- Local files under data/raw (csv/parquet)

The previous behaviour - silently fabricating synthetic random-walk data when
no source was configured - is intentionally removed. Callers must opt in via
``allow_synthetic``.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests


class MarketDataError(RuntimeError):
    """Raised when no market data source could satisfy a request."""


# Pipeline symbol -> Yahoo Finance ticker
YAHOO_SYMBOLS: Dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
    "XAUUSD": "GC=F",  # COMEX gold front-month future
}

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121 Safari/537.36",
]


def _normalize_ohlcv(df: pd.DataFrame, pair: str, spread: float = 0.0) -> pd.DataFrame:
    """Return a clean, chronologically sorted OHLCV frame with a DatetimeIndex."""
    if df is None or len(df) == 0:
        return df

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise MarketDataError(f"{pair}: expected a DatetimeIndex from provider")

    # Drop tz, sort, de-duplicate
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    core = ["open", "high", "low", "close"]
    for col in core:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=core)

    # Enforce OHLC consistency, dropping malformed bars
    ok = (
        (df["high"] >= df["low"])
        & (df["high"] >= df["close"]) & (df["close"] >= df["low"])
        & (df["high"] >= df["open"]) & (df["open"] >= df["low"])
    )
    df = df[ok]

    if "volume" not in df.columns:
        df["volume"] = 0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    # FX has no true volume; keep a positive placeholder so downstream
    # volume indicators are well-defined.
    df.loc[df["volume"] <= 0, "volume"] = 1

    if "spread" not in df.columns:
        df["spread"] = spread

    return df


class YahooFinanceProvider:
    """Keyless daily OHLCV from the Yahoo Finance chart API."""

    def __init__(self, range_: str = "10y", interval: str = "1d",
                 timeout: int = 20, max_retries: int = 6):
        self.range = range_
        self.interval = interval
        self.timeout = timeout
        self.max_retries = max_retries

    def _request(self, symbol: str) -> Optional[dict]:
        hosts = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
        last_status: Any = None
        for attempt in range(self.max_retries):
            host = hosts[attempt % len(hosts)]
            ua = _USER_AGENTS[attempt % len(_USER_AGENTS)]
            try:
                resp = requests.get(
                    f"https://{host}/v8/finance/chart/{symbol}",
                    params={"interval": self.interval, "range": self.range},
                    headers={"User-Agent": ua, "Accept": "application/json"},
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    return resp.json()
                last_status = resp.status_code
            except requests.RequestException as exc:
                last_status = str(exc)
            time.sleep(1.5 * (attempt + 1))
        raise MarketDataError(f"Yahoo Finance request for {symbol} failed: {last_status}")

    def fetch(self, pair: str, spread: float = 0.0) -> pd.DataFrame:
        symbol = YAHOO_SYMBOLS.get(pair, f"{pair}=X")
        payload = self._request(symbol)
        result = (payload or {}).get("chart", {}).get("result")
        if not result:
            raise MarketDataError(f"Yahoo Finance returned no result for {symbol}")

        result = result[0]
        timestamps = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        if not timestamps or not quote:
            raise MarketDataError(f"Yahoo Finance returned empty bars for {symbol}")

        index = pd.to_datetime(timestamps, unit="s", utc=True)
        df = pd.DataFrame(
            {
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume"),
            },
            index=index,
        )
        return _normalize_ohlcv(df, pair, spread)


class OandaProvider:
    """OHLCV from the OANDA v3 REST API (requires credentials)."""

    def __init__(self, api_key: str, account_id: str, environment: str = "practice",
                 granularity: str = "D", count: int = 5000, timeout: int = 30):
        self.api_key = api_key
        self.account_id = account_id
        self.granularity = granularity
        self.count = count
        self.timeout = timeout
        host = "api-fxtrade.oanda.com" if environment == "live" else "api-fxpractice.oanda.com"
        self.base_url = f"https://{host}/v3"

    def fetch(self, pair: str, spread: float = 0.0) -> pd.DataFrame:
        instrument = pair[:3] + "_" + pair[3:]
        resp = requests.get(
            f"{self.base_url}/instruments/{instrument}/candles",
            params={"granularity": self.granularity, "count": self.count, "price": "M"},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise MarketDataError(
                f"OANDA {instrument} error {resp.status_code}: {resp.text[:200]}")
        candles = resp.json().get("candles", [])
        rows = []
        for c in candles:
            if not c.get("complete", True):
                continue
            mid = c["mid"]
            rows.append({
                "timestamp": pd.to_datetime(c["time"], utc=True),
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c.get("volume", 0)),
            })
        if not rows:
            raise MarketDataError(f"OANDA returned no complete candles for {instrument}")
        df = pd.DataFrame(rows).set_index("timestamp")
        return _normalize_ohlcv(df, pair, spread)


def load_from_directory(path: Path) -> Dict[str, pd.DataFrame]:
    """Load OHLCV csv/parquet files from a directory."""
    data: Dict[str, pd.DataFrame] = {}
    if not path.exists():
        return data
    for file in sorted(path.glob("*")):
        if file.suffix not in (".csv", ".parquet", ".pkl"):
            continue
        pair = file.stem.upper().replace("_", "").replace("-", "")
        try:
            if file.suffix == ".csv":
                df = pd.read_csv(file)
                ts_col = next((c for c in df.columns if c.lower() in ("timestamp", "time", "date", "datetime")), None)
                if ts_col:
                    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                    df = df.set_index(ts_col)
            elif file.suffix == ".parquet":
                df = pd.read_parquet(file)
            else:
                df = pd.read_pickle(file)
            data[pair] = _normalize_ohlcv(df, pair)
        except Exception:
            continue
    return data
