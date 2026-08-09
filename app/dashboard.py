"""Interactive research dashboard for the cross-sectional reversion+momentum study.

Run:
    streamlit run app/dashboard.py

Lets a reviewer inspect equity curves (gross vs net), rolling IC, turnover, the
in-sample/out-of-sample split, and parameter sensitivity -- and change the key
knobs live to see how costs and turnover respond. Every panel is honest: gross
and net are shown side by side and the data source (real vs synthetic) is
flagged at the top.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsec.config import Config
from xsec.data import load_prices
from xsec.pipeline import run_all

st.set_page_config(page_title="XSec Reversion + Momentum", layout="wide")
st.title("Cross-Sectional Reversion + Momentum -- Research Dashboard")

st.sidebar.header("Configuration")
synthetic = st.sidebar.checkbox("Force synthetic data", value=False)
rev_lb = st.sidebar.slider("Reversion lookback (days)", 2, 20, 5)
mom_lb = st.sidebar.select_slider("Momentum lookback (days)", [63, 126, 189, 252], value=126)
w_rev = st.sidebar.slider("Blend weight on reversion", 0.0, 1.0, 0.35, 0.05)
smooth = st.sidebar.slider("Weight smoothing (days)", 1, 20, 5)
cost = st.sidebar.slider("Cost per side (bps)", 0.0, 20.0, 5.0, 0.5)


@st.cache_data(show_spinner=True)
def _run(synthetic, rev_lb, mom_lb, w_rev, smooth, cost):
    cfg = Config(reversion_lookback=rev_lb, momentum_lookback=mom_lb,
                 combo_weight_reversion=w_rev, signal_smoothing=smooth,
                 cost_bps_per_side=cost)
    data = load_prices(cfg, force_synthetic=synthetic)
    res = run_all(cfg, data)
    return res


res = _run(synthetic, rev_lb, mom_lb, w_rev, smooth, cost)

flag = "SYNTHETIC (illustrative)" if res["is_synthetic"] else f"REAL ({res['source']})"
st.info(f"Data source: **{flag}**  |  split date (IS/OS): {res['config']['train_end']}")

legs = res["legs"]
tab1, tab2, tab3, tab4 = st.tabs(["Equity curves", "Metrics", "IC & turnover", "About"])

with tab1:
    st.subheader("Net equity curves (growth of 1)")
    eq = pd.DataFrame({name: leg["backtest"].net_equity for name, leg in legs.items()})
    st.line_chart(eq)
    st.subheader("Combined: gross vs net (cost impact)")
    bt = legs["combined"]["backtest"]
    st.line_chart(pd.DataFrame({"gross": bt.gross_equity, "net": bt.net_equity}))

with tab2:
    rows = []
    for name, leg in legs.items():
        for h in ["full", "is", "os"]:
            d = dict(leg[h]); d["leg"] = name; d["horizon"] = h
            rows.append(d)
    df = pd.DataFrame(rows)[
        ["leg", "horizon", "ann_return", "ann_vol", "sharpe", "sortino",
         "max_drawdown", "hit_rate", "ann_turnover"]]
    st.dataframe(df.round(3), use_container_width=True)
    st.caption("IS = in-sample (<= split), OS = out-of-sample (> split). Net of cost.")

with tab3:
    st.subheader("Combined signal: 63-day rolling mean IC")
    ic = legs["combined"]["ic"]
    if len(ic):
        st.line_chart(ic.rolling(63).mean())
        st.write({k: round(v, 4) for k, v in legs["combined"]["ic_stats"].items()})
    st.subheader("Daily turnover (combined)")
    st.line_chart(legs["combined"]["backtest"].turnover)

with tab4:
    st.markdown(
        """
        This dashboard accompanies a study of two classic cross-sectional equity
        signals -- short-horizon **mean reversion** and intermediate-horizon
        **momentum** -- combined into a **dollar-neutral** long/short book.

        The point is not a headline Sharpe. It is an **honest** evaluation:
        gross vs net of realistic costs, in-sample vs out-of-sample, information
        coefficient with a t-stat, turnover, and parameter sensitivity. See the
        design doc and tear sheet in `docs/` for the full reasoning and the
        known limitations (factor decay, survivorship, capacity).
        """
    )
