"""Parameter sensitivity sweep -> results/sensitivity.csv + heatmap.

Purpose: demonstrate the strategy is not a single lucky parameter set. We report
out-of-sample Sharpe across a grid of reversion/momentum lookbacks and blend
weights. A robust signal shows a broad plateau of positive OOS Sharpe, not a
lone spike.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsec.config import Config, RESULTS_DIR
from xsec.data import load_prices
from xsec.pipeline import parameter_sensitivity


def main() -> None:
    cfg = Config()
    data = load_prices(cfg)
    df = parameter_sensitivity(cfg, data, leg="combined")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "sensitivity.csv", index=False)

    piv = df.pivot_table(index="reversion_lookback", columns="momentum_lookback",
                         values="sharpe_OS", aggfunc="mean")
    plt.figure(figsize=(7, 5))
    im = plt.imshow(piv.values, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="OOS Sharpe (net)")
    plt.xticks(range(len(piv.columns)), piv.columns)
    plt.yticks(range(len(piv.index)), piv.index)
    plt.xlabel("momentum_lookback"); plt.ylabel("reversion_lookback")
    plt.title("OOS Sharpe across parameters (avg over blend weight)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            plt.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", color="w")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "sensitivity_heatmap.png", dpi=130)
    plt.close()

    print(df.round(3).to_string(index=False))
    print("\nSaved sensitivity.csv and sensitivity_heatmap.png in", RESULTS_DIR)


if __name__ == "__main__":
    main()
