"""End-to-end research pipeline: data -> signals -> weights -> backtest -> metrics.

This is the single place that composes the building blocks. `run_all` returns a
dictionary of results for every leg (reversion, momentum, combined), split into
in-sample (IS) and out-of-sample (OS) windows, plus IC diagnostics. The scripts
and the dashboard both call into here so there is exactly one source of truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data import load_prices, PriceData
from . import signals as sg
from .backtest import run_backtest, BacktestResult
from . import metrics as mt


def build_signals(close: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    rev = sg.reversion_signal(close, cfg)
    mom = sg.momentum_signal(close, cfg)
    combo = sg.combine_signals(rev, mom, cfg)
    return {"reversion": rev, "momentum": mom, "combined": combo}


def _split(series_index: pd.DatetimeIndex, cfg: Config):
    split = pd.Timestamp(cfg.train_end)
    return series_index <= split, series_index > split


def run_all(cfg: Config | None = None, data: PriceData | None = None,
            compute_ic: bool = True) -> dict:
    cfg = cfg or Config()
    data = data or load_prices(cfg)
    close = data.close
    fwd = close.pct_change().shift(-1)   # return from t -> t+1, indexed at t

    sigs = build_signals(close, cfg)
    out = {"is_synthetic": data.is_synthetic, "source": data.source,
           "legs": {}, "config": cfg.to_dict()}

    for name, sig in sigs.items():
        w = sg.signal_to_weights(sig, cfg)
        bt = run_backtest(w, fwd.reindex_like(w), cfg)
        ic = mt.information_coefficient(sig, fwd) if compute_ic else pd.Series(dtype=float)

        idx = bt.net_returns.index
        is_mask, os_mask = _split(idx, cfg)
        out["legs"][name] = {
            "backtest": bt,
            "ic": ic,
            "full": mt.summarize(bt.net_returns, bt.turnover, cfg, f"{name}-net-full"),
            "full_gross": mt.summarize(bt.gross_returns, bt.turnover, cfg, f"{name}-gross-full"),
            "is": mt.summarize(bt.net_returns[is_mask], bt.turnover[is_mask], cfg, f"{name}-net-IS"),
            "os": mt.summarize(bt.net_returns[os_mask], bt.turnover[os_mask], cfg, f"{name}-net-OS"),
            "ic_stats": mt.ic_summary(ic),
        }
    return out


def parameter_sensitivity(cfg: Config | None = None, data: PriceData | None = None,
                          leg: str = "combined") -> pd.DataFrame:
    """Sweep the key knobs; record net IS/OS Sharpe and turnover.

    IC is skipped here (compute_ic=False) for speed -- this grid only needs the
    risk-adjusted return surface to demonstrate robustness / absence of a lone
    lucky spike (overfitting check).
    """
    cfg = cfg or Config()
    data = data or load_prices(cfg)
    rows = []
    for rev_lb in [3, 5, 10]:
        for mom_lb in [63, 126, 252]:
            for w in [0.3, 0.5, 0.7]:
                c = Config(**{**cfg.to_dict(),
                              "reversion_lookback": rev_lb,
                              "momentum_lookback": mom_lb,
                              "combo_weight_reversion": w})
                l = run_all(c, data, compute_ic=False)["legs"][leg]
                rows.append({
                    "reversion_lookback": rev_lb,
                    "momentum_lookback": mom_lb,
                    "combo_weight_reversion": w,
                    "sharpe_IS": l["is"]["sharpe"],
                    "sharpe_OS": l["os"]["sharpe"],
                    "ann_turnover": l["full"].get("ann_turnover", np.nan),
                })
    return pd.DataFrame(rows)
