# Design Doc -- Cross-Sectional Reversion + Momentum

This document explains the project A-to-Z: where the idea came from, the
research behind it, every non-trivial design decision, the alternatives I
rejected and why, the metrics, and the honest impact and limitations. It is
written so I can defend every line in an interview.

---

## 1. Where the idea came from

During my WorldQuant BRAIN consultancy I built cross-sectional US-equity alphas
(short-horizon mean reversion, intermediate-horizon momentum) and learned that
"stability matters more than peak performance" and "turnover reduction improves
real-world viability more than raw alpha." At Multyfi I ran mean-reversion and
linear-regression-slope strategies end-to-end on futures and learned that exit
logic and risk control dominate entries.

This project is the equity, cross-sectional distillation of both experiences,
rebuilt as clean, reproducible, testable code with the evaluation rigor a
systematic fund expects. It mirrors the industry workflow: *build weak signals,
combine them, neutralize market exposure, and judge them honestly.*

## 2. The hypotheses (stated before testing)

- **H1 (reversion):** Over a few days, relative winners in a liquid cross-section
  give back part of the move and relative losers bounce, so a portfolio that is
  long recent losers / short recent winners earns a positive return.
  *Economic basis:* liquidity provision / overreaction (Lo & MacKinlay 1990;
  Jegadeesh 1990).
- **H2 (momentum):** Past 6-12 month relative winners keep outperforming for a
  few months (Jegadeesh & Titman 1993). *Economic basis:* under-reaction to
  information, behavioral herding.
- **H3 (blend):** Because the two operate on different horizons they are weakly
  correlated, so a blend has a higher information ratio than either alone
  (diversification of alpha).

Stating hypotheses first is deliberate: it stops me from reverse-fitting a story
onto whatever the backtest happens to show.

## 3. Data and leakage controls

**Source.** Yahoo Finance adjusted close (auto-adjusted for splits/dividends).
Adjustment matters: raw prices create fake jumps on ex-dividend/split days that
a naive signal would trade. Cached to parquet for offline reproducibility.

**Universe.** ~97-100 liquid US large caps across sectors, fixed list. Fixed so
the demo is reproducible.

**Synthetic fallback.** If offline, a deterministic generator produces a panel
with a market factor, sector factors, an injected mean-reversion term and a slow
momentum term. Purpose: (a) results always reproduce; (b) it is a *positive
control* -- the pipeline should recover the injected reversion signal, proving
the machinery works. It is always flagged as synthetic; never mixed with real
data.

**Leakage controls (the decisions that make results trustworthy):**

1. *Point-in-time signals.* Every rolling computation uses only data up to day t.
2. *Implementation lag.* The backtest trades on `weights.shift(1)`: a signal
   computed from the close of day t is applied to the t->t+1 return. No signal
   ever touches the return it predicts.
3. *Unit-tested.* `test_no_lookahead_shift` corrupts the final day's return and
   asserts every prior day's P&L is unchanged -- a direct, mechanical proof of
   no look-ahead.

## 4. Signal construction -- every choice explained

### 4.1 Short-horizon reversion
- **Raw:** k-day return (default k=5).
- **Volatility normalization** (divide by 20-day vol): a 2% move in a calm name
  is a bigger *surprise* than 2% in a volatile name; normalizing puts names on a
  comparable risk scale. *Alternative rejected:* raw returns -- dominated by the
  most volatile names, which then drive turnover and drawdowns.
- **Negation:** long losers, short winners (that is what "reversion" means).
- **Winsorization at 3 sigma:** fat-tailed cross-sections let a couple of names
  hijack a z-score; clipping keeps the signal robust without dropping names.
  *Alternative rejected:* hard removal of outliers -- throws away real names and
  creates a survivorship-like bias intraday.
- **Cross-sectional z-score:** puts the signal on a standard scale each day so it
  blends cleanly with momentum.

### 4.2 Intermediate-horizon momentum
- **6-1 style:** cumulative return over ~126 days, **skipping the most recent 21
  days** (`momentum_gap`). *Why skip:* the last month is dominated by short-term
  *reversal*; including it makes momentum fight the reversion signal. This "12-1"
  (here 6-1) gap is standard in the literature for exactly this reason.
- **Risk-adjusted:** divide by vol so momentum is per-unit-risk.
- Winsorize + z-score as above.

### 4.3 Blend
- `w * reversion + (1-w) * momentum`, then re-standardized. One interpretable
  knob (`combo_weight_reversion`). Default 0.35 leans toward momentum because in
  testing momentum carried the significant IC and had far lower turnover.

### 4.4 Signal -> weights (portfolio construction)
- **Dollar neutral:** subtract the cross-sectional mean so long notional = short
  notional -> no net market bet (market-neutral). *This is the core of what
  Trexquant does: market-neutral equity books.*
- **Gross normalization:** scale so sum|w| = 1 each day (constant risk budget).
- **Position cap** (4% per name): controls single-name concentration so no one
  stock can dominate P&L. *Alternative rejected:* uncapped -- a few extreme
  signals become huge bets.
- **Weight smoothing (turnover control):** average target weights over 5 days.
  Trades a little signal freshness for a large cut in turnover and therefore
  cost. This directly encodes my WorldQuant lesson: turnover reduction improves
  real-world viability. With smoothing, combined turnover fell from ~105x to
  ~37x per year.
- **Iterative projection:** dollar-neutral + gross-target + cap conflict slightly,
  so I alternate the three projections ~12 times and finish on the cap. Result:
  cap is strictly satisfied, neutrality exact, gross within a few bps (documented
  as a soft constraint and unit-tested).

## 5. Backtest and cost model

- Vectorized: gross P&L = sum over names of `held_weight * forward_return`.
- **Turnover** = sum|w_t - w_{t-1}| (traded notional between rebalances).
- **Cost** = turnover x (5 bps per side + 1 bp slippage). This is the number
  that kills naive daily signals, so it is front and centre; the headline metric
  is always **net of cost**.
- *Alternative rejected:* event-driven per-order simulator -- more realistic
  microstructure but far more code and failure surface for a signal-research
  project; a vectorized cost-on-turnover model captures the first-order economics
  that decide whether a signal is viable.

## 6. Evaluation -- metrics and why each one

- **Information Coefficient (rank IC):** daily Spearman(signal, forward return).
  The cleanest measure of raw predictive power, independent of sizing and cost.
  Reported with **mean, IR, and t-stat** so significance is explicit. Computed
  vectorized (rank + row-wise correlation) -- identical to scipy, ~100x faster.
- **Sharpe / Sortino:** risk-adjusted return, net of cost.
- **Max drawdown:** worst peak-to-trough -- the pain metric.
- **Hit rate, turnover:** behavior and cost drivers.
- **In-sample vs out-of-sample:** the single most important honesty check.
- **Parameter sensitivity sweep:** 27 combos of lookbacks and blend weight. A
  real signal shows a broad plateau; a curve-fit shows a lone spike.

## 7. Results and interpretation (honest)

On real data 2015-2024: momentum and the blend have **significant IC**
(t-stat ~2.6-2.7), but the edge is **thin and cost-sensitive** (gross combined
Sharpe +0.27 -> net -0.04) and **decays out-of-sample** (IS Sharpe +0.37, OS
-0.65). Reversion adds no significant IC on this liquid universe and its ~87x
turnover makes it uneconomic. The sweep confirms this is not a parameter
accident: IS Sharpe is positive across the whole grid, OS is negative across the
whole grid.

**This is a genuine, publishable-style finding:** classic, well-known factors on
crowded, liquid large caps have largely been arbitraged away, and costs finish
the job. It is exactly the kind of result a systematic researcher must be able to
diagnose. See RESEARCH_REPORT.md for the tear sheet and failure analysis.

## 8. What I would do next (capacity & extension)

- Widen to the full S&P 500 / 1500 with a point-in-time, survivorship-free
  universe (CRSP/Compustat) -- more names = stronger cross-section and more
  capacity for reversion.
- Sector/beta neutralization (regress signal on sector dummies + beta) to strip
  residual risk-factor exposure.
- Add more weakly-correlated alphas and combine them (this is Project 2/3 in the
  portfolio) -- the blend logic here is the seed of that.
- Formal alpha-decay study by holding period (Project on IC decay).

## 9. Limitations (stated plainly)

- **Survivorship bias:** fixed modern universe over-represents survivors; likely
  inflates momentum slightly and understates crash risk.
- **Capacity:** ~100 names is small; reversion in particular needs breadth.
- **Costs are a model,** not fills; real slippage varies with size and regime.
- **No borrow costs / shorting constraints** modelled.
- **Single market, single asset class, daily bars.**

None of these are hidden; each is a concrete next step.
