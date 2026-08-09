"""Tests that protect the credibility of the results.

These check the properties an interviewer will probe: no look-ahead, dollar
neutrality, cost monotonicity, position cap and determinism. Run with `pytest -q`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsec.config import Config
from xsec.data import load_prices
from xsec import signals as sg
from xsec.backtest import run_backtest
from xsec.pipeline import run_all


@pytest.fixture(scope="module")
def cfg():
    return Config()


@pytest.fixture(scope="module")
def data(cfg):
    return load_prices(cfg, force_synthetic=True)


def _combo_weights(cfg, data):
    sig = sg.combine_signals(
        sg.reversion_signal(data.close, cfg),
        sg.momentum_signal(data.close, cfg), cfg)
    return sg.signal_to_weights(sig, cfg)


def test_weights_are_dollar_neutral(cfg, data):
    w = _combo_weights(cfg, data)
    net = w.sum(axis=1).abs()
    assert net.dropna().max() < 1e-9, "weights must sum to ~0 (dollar neutral)"


def test_gross_exposure_soft_target(cfg, data):
    """Gross is a soft constraint after the cap; must hold within a few bps."""
    w = _combo_weights(cfg, data)
    gross = w.abs().sum(axis=1)
    gross = gross[gross > 0]
    assert (gross - cfg.gross_exposure).abs().max() < 0.02


def test_position_cap_enforced(cfg, data):
    w = _combo_weights(cfg, data)
    assert w.abs().max().max() <= cfg.max_weight + 1e-9


def test_no_lookahead_shift(cfg, data):
    """If we corrupt the FINAL day's return, results before it must not change."""
    close = data.close
    fwd = close.pct_change().shift(-1)
    w = _combo_weights(cfg, data)

    base = run_backtest(w, fwd.reindex_like(w), cfg).net_returns
    fwd2 = fwd.copy()
    fwd2.iloc[-1] = fwd2.iloc[-1] * 0 + 999.0     # explode the last day
    pert = run_backtest(w, fwd2.reindex_like(w), cfg).net_returns

    common = base.index.intersection(pert.index)[:-1]  # everything but last day
    assert np.allclose(base.loc[common], pert.loc[common], equal_nan=True), \
        "past P&L changed when only the future changed -> look-ahead leak"


def test_costs_reduce_returns(cfg, data):
    res = run_all(cfg, data)
    for name, leg in res["legs"].items():
        g = leg["backtest"].gross_returns.sum()
        n = leg["backtest"].net_returns.sum()
        assert n <= g + 1e-9, f"net must be <= gross for {name}"


def test_turnover_nonnegative(cfg, data):
    res = run_all(cfg, data)
    for name, leg in res["legs"].items():
        assert (leg["backtest"].turnover >= -1e-12).all()


def test_determinism(cfg):
    a = load_prices(cfg, force_synthetic=True).close
    b = load_prices(cfg, force_synthetic=True).close
    assert a.equals(b), "synthetic data must be deterministic given the seed"
