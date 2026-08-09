"""Data layer: reproducible prices with real-fetch + caching + synthetic fallback.

Design intent (see docs/DESIGN_DOC.md, section "Data & Leakage Controls"):

* Prefer REAL adjusted prices from Yahoo Finance (auto-adjusted for splits and
  dividends, which removes a large source of spurious signal).
* Cache to a local parquet/csv so results reproduce offline and reviewers do not
  hammer the API.
* If the network is unavailable (or a reviewer is offline), fall back to a
  clearly-labelled SYNTHETIC generator. The synthetic data embeds a common
  market factor, sector factors, a small mean-reversion component and a slow
  momentum component, so the strategies have something real to find while the
  pipeline stays fully deterministic.

The synthetic path is never silently mixed with real data; the returned object
records `is_synthetic` so the tear sheet and dashboard can flag it honestly.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, DATA_DIR


@dataclass
class PriceData:
    close: pd.DataFrame        # adjusted close, index=date, cols=tickers
    volume: pd.DataFrame       # share volume (for liquidity/turnover context)
    is_synthetic: bool
    source: str

    @property
    def returns(self) -> pd.DataFrame:
        return self.close.pct_change()


def _cache_path(cfg: Config, kind: str) -> "os.PathLike":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{kind}_{cfg.start}_{cfg.end}_{len(cfg.universe)}"
    return DATA_DIR / f"{tag}.parquet"


def _try_load_cache(cfg: Config) -> PriceData | None:
    cp = _cache_path(cfg, "close")
    vp = _cache_path(cfg, "volume")
    if cp.exists() and vp.exists():
        try:
            close = pd.read_parquet(cp)
            volume = pd.read_parquet(vp)
            return PriceData(close, volume, is_synthetic=False, source="cache")
        except Exception:
            return None
    return None


def _save_cache(cfg: Config, close: pd.DataFrame, volume: pd.DataFrame) -> None:
    try:
        close.to_parquet(_cache_path(cfg, "close"))
        volume.to_parquet(_cache_path(cfg, "volume"))
    except Exception as exc:  # parquet engine missing, etc. -- non-fatal
        warnings.warn(f"Could not cache data: {exc!r}")


def _fetch_real(cfg: Config) -> PriceData | None:
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        raw = yf.download(
            cfg.universe, start=cfg.start, end=cfg.end,
            auto_adjust=True, progress=False, threads=False,
        )
        if raw is None or len(raw) == 0:
            return None
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()
        # Drop tickers with too little history (listing gaps, etc.).
        good = close.columns[close.notna().mean() > 0.9]
        close, volume = close[good], volume[good]
        close = close.ffill().dropna(how="all")
        if close.shape[1] < 10 or close.shape[0] < 250:
            return None
        return PriceData(close, volume, is_synthetic=False, source="yfinance")
    except Exception:
        return None


def _make_synthetic(cfg: Config) -> PriceData:
    """Deterministic synthetic panel with factor structure + embedded signals.

    Return model for each name i on day t:
        r_it = beta_i * f_market_t + sector_load * f_sector_t
             - phi * (short_term_excess_return)     # mean reversion
             + gamma * (slow_trend)                 # momentum
             + eps_it
    The reversion and momentum terms are what the strategies are designed to
    capture; everything else is common/idiosyncratic noise they must survive.
    """
    rng = np.random.default_rng(cfg.synthetic_seed)
    dates = pd.bdate_range(cfg.start, cfg.end)
    tickers = list(cfg.universe)
    n_t, n_i = len(dates), len(tickers)

    betas = rng.uniform(0.6, 1.4, n_i)
    n_sectors = 6
    sector_id = rng.integers(0, n_sectors, n_i)
    sector_load = rng.uniform(0.3, 0.8, n_i)

    market = rng.normal(0.0003, 0.010, n_t)               # market factor
    sectors = rng.normal(0.0, 0.008, (n_t, n_sectors))    # sector factors

    rets = np.zeros((n_t, n_i))
    slow_trend = np.zeros(n_i)
    short_excess = np.zeros(n_i)
    phi, gamma = 0.06, 0.03
    for t in range(n_t):
        common = betas * market[t] + sector_load * sectors[t, sector_id]
        idio = rng.normal(0.0, 0.015, n_i)
        r = common - phi * short_excess + gamma * slow_trend + idio
        rets[t] = r
        # update latent states
        short_excess = 0.5 * short_excess + (r - common)   # recent idio move
        slow_trend = 0.98 * slow_trend + 0.02 * (r - common)

    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    close = pd.DataFrame(prices, index=dates, columns=tickers)
    vol = pd.DataFrame(
        rng.lognormal(15, 0.6, (n_t, n_i)).astype(np.int64),
        index=dates, columns=tickers,
    )
    return PriceData(close, vol, is_synthetic=True, source="synthetic")


def load_prices(cfg: Config | None = None, force_synthetic: bool = False) -> PriceData:
    """Return a PriceData bundle following the cache -> real -> synthetic order."""
    cfg = cfg or Config()
    if force_synthetic:
        return _make_synthetic(cfg)

    cached = _try_load_cache(cfg)
    if cached is not None:
        return cached

    real = _fetch_real(cfg)
    if real is not None:
        _save_cache(cfg, real.close, real.volume)
        return real

    if cfg.use_synthetic_if_needed:
        warnings.warn(
            "Falling back to SYNTHETIC data (no network / no cache). "
            "Results are illustrative and clearly flagged as synthetic."
        )
        return _make_synthetic(cfg)

    raise RuntimeError("No cached or live data available and synthetic disabled.")
