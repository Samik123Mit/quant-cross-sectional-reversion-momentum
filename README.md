# Cross-Sectional Reversion + Momentum (US Equities)

A small, reproducible research package that builds two classic cross-sectional
equity signals -- short-horizon **mean reversion** and intermediate-horizon
**momentum** -- and combines them into a **dollar-neutral, market-neutral**
long/short book. The goal is not a flashy Sharpe; it is an **honest, defensible
evaluation** of whether these signals survive out-of-sample and after realistic
transaction costs.

> Personal research / educational project. Not investment advice. See
> [DISCLAIMER.md](DISCLAIMER.md).

## Why this project

Systematic equity funds build thousands of weak signals ("alphas") and combine
them into market-neutral portfolios. This repo reproduces that workflow at small
scale on a liquid US universe, with the leakage controls, cost model, and
out-of-sample discipline that separate a real signal study from an overfit
backtest.

## Headline result (honest)

On ~97 liquid US large caps, 2015-2024 (adjusted close, split at 2020-12-31):

| Signal | IC mean | IC t-stat | Net Sharpe (full) | Net Sharpe IS / OS | Ann. turnover |
|--------|--------:|----------:|------------------:|-------------------:|--------------:|
| Reversion (5d) | 0.003 | 0.80 | -0.56 | -0.11 / -1.19 | ~87x |
| Momentum (6-1) | 0.013 | 2.57 | +0.05 | +0.31 / -0.34 | ~16x |
| Combined       | 0.013 | 2.68 | -0.04 | +0.37 / -0.65 | ~37x |

**Reading of the result.** Momentum and the blend have **statistically
significant** raw predictive power (IC t-stat > 2). But on this liquid,
large-cap universe the edge is **economically thin**, is **eaten by costs**
(gross combined Sharpe +0.27 -> net -0.04), and has **decayed out-of-sample**
(positive IS, negative OS after 2020). Short-horizon reversion adds no
significant IC here and its ~87x turnover makes it uneconomic. This is a
realistic finding about crowded, well-known factors on liquid names -- and the
repo is built to show *why* rather than to hide it. See
[docs/RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md).

The same pipeline run on **synthetic data with a known injected reversion
signal** recovers positive reversion IC, confirming the machinery detects alpha
when alpha exists (a sanity check on the whole stack).

## What is in here

```
src/xsec/          core library
  config.py        all tunable choices in one dataclass
  data.py          real fetch (yfinance) + cache + synthetic fallback
  signals.py       reversion, momentum, blend, weight construction
  backtest.py      vectorized backtest w/ 1-day lag + cost model
  metrics.py       Sharpe/Sortino/MDD/turnover + vectorized rank IC
  pipeline.py      composes everything; IS/OS split; parameter sweep
scripts/
  run_backtest.py  full run -> results/ (tables + figures)
  sweep_params.py  parameter sensitivity -> heatmap
tests/             leakage, dollar-neutrality, cost, cap, determinism
app/dashboard.py   Streamlit dashboard (interactive)
docs/              DESIGN_DOC (A-to-Z reasoning) + RESEARCH_REPORT (tear sheet)
results/           generated figures + summary tables
```

## Reproduce in one command

```bash
pip install -r requirements.txt
python scripts/run_backtest.py            # real data if online, else synthetic
python scripts/run_backtest.py --synthetic  # force the deterministic synthetic path
python scripts/sweep_params.py            # parameter sensitivity
pytest -q                                 # leakage & invariance tests
streamlit run app/dashboard.py            # interactive dashboard
```

## Dashboard deployment

Deploy the dashboard through Streamlit Community Cloud:

[![Deploy](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Samik123Mit/quant-cross-sectional-reversion-momentum&branch=main&mainModule=app/dashboard.py)

Entrypoint: `app/dashboard.py`  
Live dashboard: [Open the deployed dashboard](https://samik123mit-quant-cross-sectional-reversion-appdashboard-m0kksz.streamlit.app/)

If Yahoo Finance is unreachable the pipeline **automatically** falls back to a
deterministic synthetic panel and flags it everywhere, so results always
reproduce offline.

## Leakage & bias controls (the important part)

- **No look-ahead:** weights formed on day `t` are applied to the `t -> t+1`
  return (`weights.shift(1)`); a unit test corrupts the last day's return and
  asserts prior P&L is unchanged.
- **Point-in-time signals:** every rolling window uses only past data.
- **Costs modelled up front:** turnover x (5 bps/side + 1 bp slippage), charged
  on traded notional each day; net-of-cost is the headline.
- **Out-of-sample split:** metrics reported separately for pre/post 2020-12-31.
- **Survivorship caveat:** a fixed modern universe has survivorship bias; this
  is stated openly and its likely direction discussed in the report.

## Key design choices

- Volatility-normalized signals so risk, not raw price move, drives sizing.
- Cross-sectional winsorization + z-scoring for fat-tailed robustness.
- Dollar-neutral, gross-normalized, position-capped book via an iterative
  projection (constraints satisfied simultaneously).
- Weight smoothing as an explicit **turnover control** (freshness vs cost).

## Author

Samiksha Mitra -- IIT Guwahati. Built as part of a quantitative research
portfolio (WorldQuant BRAIN and Multyfi internship background).
