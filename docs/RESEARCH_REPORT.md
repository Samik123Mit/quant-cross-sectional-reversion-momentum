# Research Report / Tear Sheet -- Cross-Sectional Reversion + Momentum

**Author:** Samiksha Mitra  |  **Universe:** ~97 liquid US large caps  |
**Period:** 2015-01-01 to 2024-12-31 (daily, adjusted close)  |
**IS/OS split:** 2020-12-31  |  **Costs:** 5 bps/side + 1 bp slippage on turnover.

> Personal research project. Hypothetical, backtested, not investment advice.

---

## 1. Hypothesis

Two classic cross-sectional equity signals earn a positive market-neutral return:
short-horizon **mean reversion** (long recent losers / short recent winners) and
intermediate-horizon **momentum** (long 6-1 winners), and a **blend** of the two
diversifies the alpha. All books are **dollar-neutral** long/short.

## 2. Method (one paragraph)

Volatility-normalized, winsorized, cross-sectionally z-scored signals are mapped
to a dollar-neutral, gross-normalized, position-capped book (4%/name) with 5-day
weight smoothing as a turnover control. The backtest trades on the **lagged**
signal (`shift(1)`) so there is no look-ahead, and charges cost on daily
turnover. Headline metrics are **net of cost**; gross is shown for diagnosis.

## 3. Headline results (net of cost)

| Signal | IC mean | IC t-stat | IC hit | Sharpe (full) | Sharpe IS | Sharpe OS | Ann. turnover | Max DD |
|--------|--------:|----------:|-------:|--------------:|----------:|----------:|--------------:|-------:|
| Reversion (5d) | 0.003 | 0.80 | 0.49 | -0.56 | -0.11 | -1.19 | 86.5x | -0.35 |
| Momentum (6-1) | 0.013 | 2.57 | 0.53 | +0.05 | +0.31 | -0.34 | 15.8x | -0.20 |
| Combined       | 0.013 | 2.68 | 0.53 | -0.04 | +0.37 | -0.65 | 36.7x | -0.23 |

Gross (pre-cost) full-sample Sharpe: reversion +0.20, momentum +0.18, combined
+0.27. **Costs turn the combined signal from +0.27 gross to -0.04 net.**

## 4. Figures (in results/)

- `equity_curves.png` -- net equity for all three legs.
- `cost_impact.png` -- combined gross vs net (the cost wedge).
- `rolling_ic.png` -- 63-day rolling IC of the combined signal.
- `sensitivity_heatmap.png` -- OOS Sharpe across the parameter grid.

## 5. What the results mean (honest interpretation)

1. **There is real but weak signal.** Momentum and the blend have IC t-stats
   above 2 -- statistically significant predictive power over ~2,600 days.
2. **Costs dominate.** The gross edge (+0.27 Sharpe combined) is small enough
   that realistic 5-bps/side costs erase it. Short-horizon reversion is the worst
   offender: ~87x annual turnover with no significant IC.
3. **The edge decayed out-of-sample.** Every leg is positive in-sample and
   negative out-of-sample (post-2020). This is consistent with well-documented
   factor crowding/decay in liquid large caps.
4. **It is not a parameter accident.** The 27-point sweep shows IS Sharpe
   positive everywhere and OS Sharpe negative everywhere -- a consistent story,
   not a fragile spike. Higher reversion weight monotonically raises turnover and
   worsens OOS, which is economically coherent.

## 6. Positive control (does the machinery work?)

Run on synthetic data with a **known injected reversion signal**, the pipeline
recovers a positive reversion IC (~0.017) and a positive OOS Sharpe on the
combined leg. This confirms that the flat real-data result is a statement about
*the market*, not a bug in the code.

## 7. Failure cases / when this does not work

- **Trending regimes** hurt reversion (losers keep losing); **choppy regimes**
  hurt momentum. Neither leg has an explicit regime filter -- a known gap.
- **High-turnover configs** (reversion-heavy) are uneconomic after costs.
- **Small universe** (~100 names) limits reversion, which needs breadth.

## 8. Limitations

Survivorship bias (fixed modern universe), modelled (not executed) costs, no
borrow/short constraints, single asset class, daily bars. Each is a concrete
next step, detailed in DESIGN_DOC.md section 8-9.

## 9. Conclusion

Classic cross-sectional reversion and momentum on liquid US large caps show
**statistically significant but economically thin** predictive power that is
**erased by costs and has decayed out-of-sample**. The value of this project is
the honest, reproducible framework that demonstrates *why* -- with leakage
controls, a cost model, IS/OS discipline, IC significance testing, and a
parameter sweep -- which is exactly the toolkit needed to evaluate the next,
better signal.
