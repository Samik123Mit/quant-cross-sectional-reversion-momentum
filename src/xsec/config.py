"""Central configuration for the research package.

All tunable choices live here so the design doc, code, and tear sheet stay in
sync. Values are deliberately conservative and documented in docs/DESIGN_DOC.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# A liquid, recognizable US universe (~100 large/mid caps across sectors). A
# fixed list keeps the demo reproducible and survivorship-aware (see design doc);
# the pipeline works unchanged on any wider universe (e.g. the S&P 500).
DEFAULT_UNIVERSE = [
    # mega-cap tech / comm
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","NFLX","ADBE","CRM",
    "ORCL","CSCO","INTC","AMD","QCOM","TXN","AVGO","IBM","NOW","INTU",
    # financials
    "JPM","BAC","WFC","C","GS","MS","AXP","BLK","SCHW","USB","PNC","COF",
    # health care
    "JNJ","PFE","MRK","ABBV","LLY","UNH","CVS","AMGN","GILD","BMY","MDT","TMO",
    # consumer
    "WMT","COST","HD","LOW","TGT","MCD","SBUX","NKE","PG","KO","PEP","CL","MDLZ",
    # industrials / energy / materials
    "XOM","CVX","COP","SLB","BA","CAT","GE","MMM","HON","UPS","FDX","LMT","RTX",
    "DE","EMR","DUK","SO","NEE","D",
    # misc large caps
    "DIS","V","MA","PYPL","T","VZ","CMCSA","ABT","DHR","LIN","APD","NEM",
    "F","GM","DAL","MAR","BKNG","ADP","ISRG","SPGI","MMC","PLD","AMT",
]

# de-dup while preserving order
_seen = set()
DEFAULT_UNIVERSE = [t for t in DEFAULT_UNIVERSE if not (t in _seen or _seen.add(t))]


@dataclass
class Config:
    # --- data ---
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    use_synthetic_if_needed: bool = True
    synthetic_seed: int = 7

    # --- signal ---
    reversion_lookback: int = 5      # short-horizon reversal window (days)
    momentum_lookback: int = 126     # ~6 months
    momentum_gap: int = 21           # skip most recent month (avoid 1M reversal)
    vol_lookback: int = 20           # rolling vol for normalization
    winsor_z: float = 3.0            # cross-sectional winsorization (z units)
    combo_weight_reversion: float = 0.35  # weight on reversion in the blend
    signal_smoothing: int = 5        # rolling-mean smoothing of weights (turnover control)

    # --- portfolio ---
    holding_period: int = 1          # rebalance daily (smoothing controls turnover)
    max_weight: float = 0.04         # position cap (per name)
    gross_exposure: float = 1.0      # gross book = 1.0 (long 0.5 / short 0.5)

    # --- costs ---
    cost_bps_per_side: float = 5.0   # 5 bps per side transaction cost
    slippage_bps: float = 1.0        # extra slippage on traded notional

    # --- evaluation ---
    train_end: str = "2020-12-31"    # in-sample / out-of-sample split date
    ann_factor: int = 252

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = Config()
