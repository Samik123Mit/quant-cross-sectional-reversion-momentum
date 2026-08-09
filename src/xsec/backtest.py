"""Vectorized cross-sectional backtest with explicit leakage controls & costs.

Key modelling decisions (all defended in the design doc):

* LEAKAGE CONTROL: weights formed from information up to day t are applied to
  the return from t -> t+1. Concretely, `weights.shift(1)` multiplies today's
  return, so no signal ever sees the return it is trying to predict.

* TRANSACTION COSTS: we charge cost on the *traded* notional each day,
  turnover_t = sum |w_t - w_{t-1}|, priced at (cost_bps_per_side + slippage_bps).
  This is what kills naive high-Sharpe daily signals, so it is modelled up front.

* NET-OF-COST is the headline; gross is shown only for diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class BacktestResult:
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame
    cost_drag: pd.Series

    @property
    def gross_equity(self) -> pd.Series:
        return (1 + self.gross_returns).cumprod()

    @property
    def net_equity(self) -> pd.Series:
        return (1 + self.net_returns).cumprod()


def run_backtest(weights: pd.DataFrame, fwd_returns: pd.DataFrame,
                 cfg: Config) -> BacktestResult:
    """Apply weights to forward returns with a one-day implementation lag."""
    # Align.
    weights = weights.reindex_like(fwd_returns).fillna(0.0)

    # ---- LEAKAGE CONTROL: trade on yesterday's signal ----
    held = weights.shift(1).fillna(0.0)

    # Gross P&L = sum over names of weight * that day's return.
    gross = (held * fwd_returns).sum(axis=1)

    # Turnover = traded notional between consecutive rebalances.
    turnover = (held - held.shift(1)).abs().sum(axis=1).fillna(0.0)

    # Costs on traded notional.
    cost_rate = (cfg.cost_bps_per_side + cfg.slippage_bps) / 1e4
    cost_drag = turnover * cost_rate

    net = gross - cost_drag

    # Trim the warm-up region where signals are undefined.
    valid = gross.replace([np.inf, -np.inf], np.nan).notna()
    return BacktestResult(
        gross_returns=gross[valid],
        net_returns=net[valid],
        turnover=turnover[valid],
        weights=held.loc[valid.index[valid]],
        cost_drag=cost_drag[valid],
    )
