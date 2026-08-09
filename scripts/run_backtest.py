"""Run the full research pipeline and write results + figures to results/.

Usage:
    python scripts/run_backtest.py            # real data if available, else synthetic
    python scripts/run_backtest.py --synthetic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsec.config import Config, RESULTS_DIR
from xsec.data import load_prices
from xsec.pipeline import run_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="force synthetic data")
    args = ap.parse_args()

    cfg = Config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_prices(cfg, force_synthetic=args.synthetic)
    res = run_all(cfg, data)

    # ---- collect summary table ----
    rows = []
    for name, leg in res["legs"].items():
        for horizon in ["full", "is", "os"]:
            d = dict(leg[horizon]); d["leg"] = name; d["horizon"] = horizon
            d.update({k: leg["ic_stats"][k] for k in ("ic_mean", "ic_ir", "ic_tstat")})
            rows.append(d)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump({"is_synthetic": res["is_synthetic"], "source": res["source"],
                   "config": res["config"],
                   "table": summary.to_dict(orient="records")}, f, indent=2, default=float)

    # ---- equity curves (net) ----
    plt.figure(figsize=(10, 6))
    for name, leg in res["legs"].items():
        eq = leg["backtest"].net_equity
        plt.plot(eq.index, eq.values, label=f"{name} (net)")
    plt.title(f"Net equity curves  [{'SYNTHETIC' if res['is_synthetic'] else res['source']}]")
    plt.ylabel("Growth of 1"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "equity_curves.png", dpi=130)
    plt.close()

    # ---- combined: gross vs net (cost impact) ----
    bt = res["legs"]["combined"]["backtest"]
    plt.figure(figsize=(10, 6))
    plt.plot(bt.gross_equity.index, bt.gross_equity.values, label="combined (gross)")
    plt.plot(bt.net_equity.index, bt.net_equity.values, label="combined (net of cost)")
    plt.title("Cost impact on the combined signal")
    plt.ylabel("Growth of 1"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "cost_impact.png", dpi=130)
    plt.close()

    # ---- rolling IC (combined) ----
    ic = res["legs"]["combined"]["ic"]
    if len(ic):
        plt.figure(figsize=(10, 4))
        ic.rolling(63).mean().plot()
        plt.axhline(0, color="k", lw=0.8)
        plt.title("Combined signal: 63-day rolling mean IC")
        plt.tight_layout(); plt.savefig(RESULTS_DIR / "rolling_ic.png", dpi=130)
        plt.close()

    # ---- console report ----
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("Data source:", res["source"], "| synthetic:", res["is_synthetic"])
    cols = ["leg", "horizon", "ann_return", "ann_vol", "sharpe", "sortino",
            "max_drawdown", "hit_rate", "ann_turnover", "ic_mean", "ic_ir"]
    print(summary[cols].round(3).to_string(index=False))
    print("\nSaved: summary.csv, summary.json, equity_curves.png, cost_impact.png, rolling_ic.png in", RESULTS_DIR)


if __name__ == "__main__":
    main()
