"""Standing 1 MW virtual positions and fees (spec v3 §4.2;
docs/virtual_bidding_rules_and_costs.md). d = +1 is an INC (earns DART); d = -1 a DEC.

The functions that learn from training rows (direction, spike exposure, stress threshold) refuse any row whose
``split`` is not "train" (invariant I4). Nothing is dropped or skipped silently: a call that leaves something
out emits one consolidated UserWarning naming it, as in es/policies.py, and says nothing when nothing is affected.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

KEY = ["market", "node", "settle_hour"]
NODE = ["market", "node"]
DAY = ["market", "node", "settle_date"]

#: Calibrated, side-specific $/MWh scenario (1 = INC, -1 = DEC; Rulings 78 and 90). Every level is the latest value
#: published on or before 2026-06-30, rounded to 0.05, as derived in docs/fee_rederivation_2026-10-23.md. MISO omits
#: the Schedule 17 and 24 administrative adders, because no posting dated on or before June 2026 could be retrieved.
#: The uniform-fee grid covers higher fee levels.
CALIBRATED_FEES = {"NYISO": {1: 0.20, -1: 0.15}, "ISONE": {1: 0.70, -1: 0.70},
                   "MISO": {1: 0.15, -1: 0.05}, "CAISO": {1: 0.55, -1: 0.15}}
UNIFORM_FEES = (0.0, 0.10, 0.25, 0.50, 1.00)


def _key_name(key) -> str:
    return "/".join(map(str, key))


def _names(frame, cols) -> str:
    """The distinct rows of ``frame[cols]``, sorted, as 'market/node/hour' names."""
    keys = frame[cols].drop_duplicates().sort_values(cols).itertuples(index=False, name=None)
    return ", ".join(map(_key_name, keys))


def _require_training_rows(rows, caller) -> None:
    """Invariant I4: raise unless every row is a training row. A frame with no ``split`` column passes."""
    if "split" not in rows.columns:
        return
    other = rows["split"][rows["split"] != "train"]
    if len(other):
        found = ", ".join(f"{s!r} ({n} row(s))" for s, n in other.astype(str).value_counts().sort_index().items())
        raise ValueError(f"{caller}: only training rows may be used (invariant I4), but the split column also "
                         f"holds {found}; select split == 'train' first")


def premium_direction(train_rows, col="dart_spread"):
    """d per (market, node, hour): +1 (INC) where the training mean of ``col`` is >= 0, else -1 (DEC).

    A node-hour whose mean is NaN has no usable training value. Its d is NaN, meaning no position (the rows
    ``attach_pnl`` finds there are dropped), and one UserWarning names all such node-hours.
    """
    _require_training_rows(train_rows, "premium_direction")
    mean = train_rows.groupby(KEY)[col].mean()
    undefined = mean.isna()
    d = pd.Series(np.where(mean >= 0, 1, -1), index=mean.index, name="d").where(~undefined)
    if undefined.any():
        warnings.warn(f"premium_direction: {int(undefined.sum())} node-hour(s) get no direction because the "
                      f"training mean of {col!r} is NaN, so no position is taken there: "
                      f"{', '.join(map(_key_name, mean.index[undefined.to_numpy()]))}",
                      UserWarning, stacklevel=2)
    return d


def spike_exposure(train_rows, direction, col="dart_spread"):
    """"short_spikes" where the training skew of d * ``col`` is negative (steady gains, rare large losses:
    sitting out should help), else "long_spikes" (sitting out should hurt)."""
    _require_training_rows(train_rows, "spike_exposure")
    t = train_rows.join(direction, on=KEY)
    skew = (t["d"] * t[col]).groupby([t[k] for k in KEY]).skew()
    return pd.Series(np.where(skew < 0, "short_spikes", "long_spikes"), index=skew.index, name="exposure")


def stress_threshold(train_rows):
    """Per node, the 90th percentile of the unscaled abs_mean_168h over the training rows."""
    _require_training_rows(train_rows, "stress_threshold")
    return train_rows.groupby(NODE)["abs_mean_168h"].quantile(0.9).rename("stress_thr")


def _stressed_days(rows, stress):
    """The (market, node, settle_date) days that are stressed (spec §4.2: "fees double on days when ...").

    A day is stressed when abs_mean_168h at its first hour, the row with the earliest timestamp_utc, is at or
    above the node's threshold. That value is lagged 48 h, so it is known before the day-ahead bid. A node
    with no threshold (absent from ``stress``, or NaN) is never stressed.
    """
    first = (rows[DAY + ["timestamp_utc", "abs_mean_168h"]]
             .sort_values("timestamp_utc", kind="stable").drop_duplicates(DAY).join(stress, on=NODE))
    hot = first["abs_mean_168h"].to_numpy(dtype=float) >= first["stress_thr"].to_numpy(dtype=float)
    return first.loc[hot, DAY]


def attach_pnl(rows, position, direction=None, fees="calibrated", stress=None):
    """Add d, fee and pnl_open = d * dart_settle - fee, the hourly P&L while the position is open.

    ``position`` is "premium" (d from ``direction``), "inc" or "dec". ``fees`` is "calibrated" (side-specific,
    CALIBRATED_FEES) or one uniform $/MWh. With ``stress`` (from ``stress_threshold``) the fee doubles in every
    hour of a stressed day (see ``_stressed_days``; rows then need settle_date, timestamp_utc and
    abs_mean_168h). A day's stress is judged on all the rows given, whichever of them the premium position keeps.

    Rows the premium position has no direction for are dropped, and nodes with no stress threshold are never
    stressed. One UserWarning per call names both, and there is none when nothing is affected.
    """
    if stress is not None:
        absent = [c for c in ("settle_date", "timestamp_utc", "abs_mean_168h") if c not in rows.columns]
        if absent:
            raise ValueError("attach_pnl: stress is judged per node-day, so rows need settle_date, "
                             f"timestamp_utc and abs_mean_168h (missing: {', '.join(absent)})")
    r = rows.copy()
    notes = []
    if position == "premium":
        r = r.join(direction, on=KEY)
        has_d = r["d"].notna().to_numpy()
        if not has_d.all():
            notes.append(f"{int((~has_d).sum())} row(s) dropped because their (market, node, hour) has no "
                         f"direction: {_names(r[~has_d], KEY)}")
            r = r[has_d].copy()
        r["d"] = r["d"].astype(int)
    elif position == "inc":
        r["d"] = 1
    elif position == "dec":
        r["d"] = -1
    else:
        raise ValueError(position)
    if fees == "calibrated":
        r["fee"] = [CALIBRATED_FEES[m][d] for m, d in zip(r["market"], r["d"])]
    else:
        r["fee"] = float(fees)
    r["fee"] = r["fee"].astype(float)
    if stress is not None:
        stressed = _stressed_days(rows, stress)
        hot = r[DAY].merge(stressed, on=DAY, how="left", indicator=True)["_merge"].eq("both").to_numpy()
        r.loc[hot, "fee"] = r.loc[hot, "fee"] * 2.0
        nodes = r[NODE].drop_duplicates().join(stress, on=NODE)
        no_thr = nodes[nodes["stress_thr"].isna()]
        if len(no_thr):
            notes.append(f"no stress threshold for {len(no_thr)} node(s), so their fees are never doubled: "
                         f"{_names(no_thr, NODE)}")
    r["pnl_open"] = r["d"] * r["dart_settle"].astype(float) - r["fee"]
    if notes:
        warnings.warn(f"attach_pnl: {' | '.join(notes)}", UserWarning, stacklevel=2)
    return r
