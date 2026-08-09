"""Honest performance & risk metrics (net of cost unless stated).

We deliberately report the metrics an interviewer will ask about: annualized
return/vol, Sharpe, Sortino, max drawdown, hit rate, turnover, and the
information coefficient (rank correlation of signal with forward return). A
single high-Sharpe number is never presented on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def ann_return(r: pd.Series, ann: int) -> float:
    return float((1 + r).prod() ** (ann / max(len(r), 1)) - 1)


def ann_vol(r: pd.Series, ann: int) -> float:
    return float(r.std() * np.sqrt(ann))


def sharpe(r: pd.Series, ann: int) -> float:
    sd = r.std()
    return float(np.sqrt(ann) * r.mean() / sd) if sd > 0 else np.nan


def sortino(r: pd.Series, ann: int) -> float:
    downside = r[r < 0].std()
    return float(np.sqrt(ann) * r.mean() / downside) if downside > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1
    return float(dd.min())


def hit_rate(r: pd.Series) -> float:
    nz = r[r != 0]
    return float((nz > 0).mean()) if len(nz) else np.nan


def information_coefficient(signal: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional rank IC = Spearman(signal_t, return_{t->t+1}).

    Vectorized: Spearman == Pearson on cross-sectional RANKS, so we rank each row
    and take the row-wise correlation. This is ~100x faster than a per-day
    scipy.spearmanr loop and gives identical values. We use the *lagged* signal
    against the same-day forward return to stay point-in-time.
    """
    sig = signal.shift(1)
    common = sig.index.intersection(fwd_returns.index)
    sig = sig.loc[common]
    ret = fwd_returns.loc[common]
    # mask to names present in both, per row
    mask = sig.notna() & ret.notna()
    sig = sig.where(mask)
    ret = ret.where(mask)
    rs = sig.rank(axis=1)
    rr = ret.rank(axis=1)
    rs = rs.sub(rs.mean(axis=1), axis=0)
    rr = rr.sub(rr.mean(axis=1), axis=0)
    num = (rs * rr).sum(axis=1)
    den = np.sqrt((rs ** 2).sum(axis=1) * (rr ** 2).sum(axis=1))
    ic = (num / den.replace(0, np.nan))
    # require at least 5 names that day
    enough = mask.sum(axis=1) >= 5
    return ic[enough].dropna()


def summarize(returns: pd.Series, turnover: pd.Series | None,
              cfg: Config, label: str = "") -> dict:
    d = {
        "label": label,
        "n_days": int(len(returns)),
        "ann_return": ann_return(returns, cfg.ann_factor),
        "ann_vol": ann_vol(returns, cfg.ann_factor),
        "sharpe": sharpe(returns, cfg.ann_factor),
        "sortino": sortino(returns, cfg.ann_factor),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
    }
    if turnover is not None and len(turnover):
        d["avg_daily_turnover"] = float(turnover.mean())
        d["ann_turnover"] = float(turnover.mean() * cfg.ann_factor)
    return d


def ic_summary(ic: pd.Series) -> dict:
    if len(ic) == 0:
        return {"ic_mean": np.nan, "ic_std": np.nan, "ic_ir": np.nan,
                "ic_tstat": np.nan, "ic_hit": np.nan}
    mean, sd = ic.mean(), ic.std()
    return {
        "ic_mean": float(mean),
        "ic_std": float(sd),
        "ic_ir": float(mean / sd) if sd > 0 else np.nan,          # info ratio of IC
        "ic_tstat": float(mean / sd * np.sqrt(len(ic))) if sd > 0 else np.nan,
        "ic_hit": float((ic > 0).mean()),
    }
