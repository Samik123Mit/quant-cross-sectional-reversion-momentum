"""Signal construction: cross-sectional reversion and momentum.

Every function returns a DataFrame aligned to (date x ticker). Signals are
constructed so they are *point-in-time*: the signal available on day t only uses
information up to and including day t, and is later shifted by one day before it
touches returns (see backtest.py) so there is no look-ahead.

Rationale for each choice is documented inline and, in full, in the design doc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _winsorize_cs(df: pd.DataFrame, z: float) -> pd.DataFrame:
    """Cross-sectional winsorization at +/- z standard deviations."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    z_df = df.sub(mu, axis=0).div(sd, axis=0)
    z_clip = z_df.clip(-z, z)
    return z_clip.mul(sd, axis=0).add(mu, axis=0)


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize each row (day) to mean 0, std 1 across names."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def reversion_signal(close: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Short-horizon cross-sectional mean reversion.

    Idea (Lo & MacKinlay 1990; Jegadeesh 1990): over a few days, relative winners
    tend to give back part of the move and relative losers bounce. We therefore
    go LONG recent losers and SHORT recent winners.

    Construction:
      1. r_k = k-day return (the recent move).
      2. normalize by rolling volatility so a 2% move in a calm name is treated
         as bigger than a 2% move in a wild name.
      3. NEGATE (reversion), winsorize, and cross-sectionally z-score.
    """
    k = cfg.reversion_lookback
    past_ret = close.pct_change(k)
    vol = close.pct_change().rolling(cfg.vol_lookback).std()
    norm = past_ret / vol.replace(0, np.nan)
    sig = -norm  # long losers, short winners
    sig = _winsorize_cs(sig, cfg.winsor_z)
    return cross_sectional_zscore(sig)


def momentum_signal(close: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Intermediate-horizon cross-sectional momentum (12-1 style, here 6-1).

    Idea (Jegadeesh & Titman 1993): past 6-12 month relative winners keep
    outperforming for a few months. We skip the most recent month (`momentum_gap`)
    because that horizon is dominated by short-term REVERSAL and would fight the
    reversion signal.

    Construction:
      1. cumulative return from t-(L+gap) to t-gap.
      2. normalize by rolling vol (risk-adjusted momentum).
      3. winsorize + cross-sectional z-score (no negation: long winners).
    """
    L, g = cfg.momentum_lookback, cfg.momentum_gap
    px_gap = close.shift(g)
    mom = px_gap / px_gap.shift(L) - 1.0
    vol = close.pct_change().rolling(cfg.vol_lookback).std()
    norm = mom / (vol.replace(0, np.nan) * np.sqrt(L))
    sig = _winsorize_cs(norm, cfg.winsor_z)
    return cross_sectional_zscore(sig)


def combine_signals(rev: pd.DataFrame, mom: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Weighted blend of the two z-scored signals, then re-standardized.

    Because reversion (days) and momentum (months) operate on different horizons
    they are weakly correlated, so blending them diversifies the alpha. The blend
    weight is a single interpretable knob (`combo_weight_reversion`).
    """
    w = cfg.combo_weight_reversion
    combo = w * rev + (1.0 - w) * mom
    return cross_sectional_zscore(combo)


def _project_weights(w: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Iterative projection onto {dollar-neutral, gross=target, |w|<=cap}.

    These three constraints mildly conflict, so we alternate the projections a
    few times (Dykstra-style) and finish on the CAP so single-name concentration
    is the binding, strictly-satisfied constraint. Dollar-neutrality and gross
    then hold to a few basis points -- documented as soft constraints in the
    design doc.
    """
    for _ in range(12):
        w = w.sub(w.mean(axis=1), axis=0)                 # dollar neutral
        gross = w.abs().sum(axis=1).replace(0, np.nan)
        w = w.div(gross, axis=0) * cfg.gross_exposure     # gross target
        w = w.clip(-cfg.max_weight, cfg.max_weight)       # cap (final op each pass)
    return w.fillna(0.0)


def signal_to_weights(signal: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Map a cross-sectional score to dollar-neutral, capped portfolio weights.

    Steps:
      1. demean each day  -> dollar neutral (sum of weights = 0).
      2. scale so gross exposure (sum |w|) = cfg.gross_exposure each day.
      3. OPTIONAL smoothing: average target weights over `signal_smoothing` days.
         Smoothing is a turnover control -- it trades a small amount of signal
         freshness for a large reduction in trading cost, which is the single
         biggest driver of whether a daily signal survives costs in practice.
      4. iterative projection so the book is dollar-neutral, gross-normalized and
         position-capped simultaneously (see `_project_weights`).
    The result is a market-neutral long/short book: no net market bet, position
    size proportional to conviction, single-name concentration controlled.
    """
    s = signal.sub(signal.mean(axis=1), axis=0)
    gross = s.abs().sum(axis=1).replace(0, np.nan)
    w = s.div(gross, axis=0) * cfg.gross_exposure

    if cfg.signal_smoothing and cfg.signal_smoothing > 1:
        w = w.rolling(cfg.signal_smoothing, min_periods=1).mean()

    return _project_weights(w, cfg)
