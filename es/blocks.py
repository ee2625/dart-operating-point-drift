"""Feature blocks, identical to v2/scripts/regime_ablation.py:21-35. Every model carries
the G6 calendar block first, in g6_features() order, then the price-history block."""
from __future__ import annotations

from v2.panel_prep import G6_COLUMNS, g1_feature_names

G6 = sorted(G6_COLUMNS)
G1 = g1_feature_names()
LEV6 = [f for f in G1 if f.startswith(("da_lmp_lag", "rt_lmp_lag"))]
DART18 = [f for f in G1 if f not in LEV6]
DART3 = [f for f in DART18 if f.startswith("dart_spread_lag")]
ROLL15 = [f for f in DART18 if "roll" in f]
ROLL_DISP9 = [f for f in ROLL15 if any(k in f for k in ("std", "maximum", "minimum"))]
ROLL_LOC6 = [f for f in ROLL15 if "mean" in f]
MID3 = [f"mid_lag_{h}h" for h in (48, 72, 168)]

BLOCKS = {"G6": [], "LEV6": LEV6, "MID3": MID3, "DART3": DART3, "MIDDART6": MID3 + DART3,
          "ROLL15": ROLL15, "ROLL_DISP9": ROLL_DISP9, "ROLL_LOC6": ROLL_LOC6,
          "DART18": DART18, "G1_24": G1}

#: Names used in the manuscript.
PAPER_BLOCK_NAMES = {"G6": "calendar", "LEV6": "absolute level", "MID3": "midpoint",
                     "DART3": "plain DART lags", "MIDDART6": "midpoint + difference",
                     "ROLL15": "all rolling", "ROLL_DISP9": "rolling dispersion",
                     "ROLL_LOC6": "rolling location", "DART18": "DART-relative",
                     "G1_24": "combined"}


def features(block: str) -> list[str]:
    return G6 + list(BLOCKS[block])
