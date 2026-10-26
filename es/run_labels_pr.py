"""Precision/recall at the deployed cutoff (R2), against the training-q label and an
evaluation-only top-alpha label; plus the 2x2 label/season control (spec v3 §5.3-5.4).
Run: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.run_labels_pr

Outputs go to ``outputs/es/labels_pr/``:

* ``pr_all.csv``: per node, for every legacy and analog analysis, block and estimator, under P1 at alpha
  0.05 with seed 42;
* ``label_control_2x2.csv``: failures, node count and median R of the four label-control analyses, xgb, four
  blocks. ``n_nodes`` is the number of nodes in the cell's activity table: a node that was not scored or has
  no calibration or deployment rows is absent, so cells with different node counts must not be compared on
  ``failures`` alone. Plan 2B reads this file by column name;
* ``label_control_pr.csv``: the per-node precision and recall of ``pr_all.csv`` for those control cells
  (without its ``estimator`` column: the control is xgb only), because spec v3 §5.4 says that item 3,
  precision and recall against both labels, also applies to the control (controller Ruling 43). It reuses
  the control cells' alerts, so it needs no further model fits.

``precision_recall`` takes the deployment rows of one analysis under one policy (the frame ``es.policies``
returns, with ``market``, ``node``, ``alert``, ``label`` and ``dart_spread``) and reports, per node:

* ``precision_q``, ``recall_q``: against the training-q ``label``;
* ``precision_eval``, ``recall_eval``: against the evaluation-only label, the node's top ``alpha`` share
  of |DART| within its own deployment rows (``es.labels.label_eval_top_alpha``);
* ``prevalence_eval``: the share of the node's rows that the evaluation label marks. It is close to
  ``alpha`` but not equal to it (ties, and the interpolated quantile on a finite sample);
* ``n_nan_outcome`` (controller Ruling 39): how many of the node's rows have a NaN ``dart_spread`` or a NaN
  ``label`` (each row counted once), i.e. the rows left out of at least one metric above. It is 0 when
  nothing is NaN;
* ``precision_lift`` = ``precision_eval / prevalence_eval`` (controller Ruling 12; spec v3 §5.3 carries this
  as an amendment, its original text divided by alpha, which has that small prevalence bias).

A ratio with nothing to divide by is NaN: precision for a node that never alerts, recall for a node with no
positive label, the lift of a node with no evaluation positive. A row whose outcome is unknown (NaN DART,
hence NaN ``label``) is neither a hit nor a miss: it leaves the denominators of precision, recall and
prevalence whether or not it alerts. It leaves a metric only if it lacks that metric's own label, the
training-q ``label`` for the ``_q`` metrics and ``dart_spread`` for the ``_eval`` ones (controller Ruling 39).
``n_nan_outcome`` counts such rows, so none is dropped silently.
"""
from __future__ import annotations

import pandas as pd

from .analyses import ANALOGS, LABEL_CONTROL, LEGACY
from .blocks import BLOCKS
from .config import OUT_DIR, ensure_dirs
from .data import build_panel, load_dev_raw
from .labels import label_eval_top_alpha
from .policies import activity, p1_fixed
from .scores import cached_scores

PR_DIR = OUT_DIR / "labels_pr"
PR_COLUMNS = ["market", "node", "precision_q", "recall_q", "precision_eval", "recall_eval",
              "prevalence_eval", "n_nan_outcome", "precision_lift"]

#: Cutoff alpha (also the evaluation label's alpha), the model seed and the estimators of the PR grid.
CUTOFF_ALPHA = 0.05
SEED = 42
ESTIMATORS = ("xgb", "logit")
#: Blocks of the 2x2 label control.
CONTROL_BLOCKS = ("LEV6", "DART18", "G1_24", "DART3")


def _share(flag, rows):
    """Mean of the boolean array ``flag`` over the ``rows`` mask; NaN if the mask selects nothing."""
    return float(flag[rows].mean()) if rows.any() else float("nan")


def precision_recall(alerts, alpha):
    """Per-node precision, recall, evaluation prevalence and precision lift of ``alerts``.

    One row per (market, node), columns ``PR_COLUMNS``. See the module docstring for the definitions.
    """
    a = alerts.assign(eval_label=label_eval_top_alpha(alerts, alpha))
    rows = []
    for (market, node), g in a.groupby(["market", "node"]):
        alert = g["alert"].to_numpy(bool)
        known_q, known_e = g["label"].notna().to_numpy(), g["dart_spread"].notna().to_numpy()
        pos_q = known_q & (g["label"].to_numpy(float) == 1)
        pos_e = known_e & (g["eval_label"].to_numpy(int) == 1)
        prevalence = _share(pos_e, known_e)
        precision_e = _share(pos_e, alert & known_e)
        rows.append({"market": market, "node": node,
                     "precision_q": _share(pos_q, alert & known_q), "recall_q": _share(alert, pos_q),
                     "precision_eval": precision_e, "recall_eval": _share(alert, pos_e),
                     "prevalence_eval": prevalence, "n_nan_outcome": int((~(known_q & known_e)).sum()),
                     "precision_lift": precision_e / prevalence if prevalence > 0 else float("nan")})
    return pd.DataFrame(rows, columns=PR_COLUMNS)


def main():
    ensure_dirs()
    PR_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_panel(load_dev_raw())
    out = []
    for a in list(LEGACY.values()) + list(ANALOGS.values()):
        for block in BLOCKS:
            for est in ESTIMATORS:
                alerts = p1_fixed(cached_scores(panel, a, block, est, SEED), CUTOFF_ALPHA)
                out.append(precision_recall(alerts, CUTOFF_ALPHA).assign(analysis=a.id, block=block, estimator=est))
    pd.concat(out, ignore_index=True).to_csv(PR_DIR / "pr_all.csv", index=False)
    ctl, ctl_pr = [], []
    for a in LABEL_CONTROL.values():
        for block in CONTROL_BLOCKS:
            alerts = p1_fixed(cached_scores(panel, a, block, "xgb", SEED), CUTOFF_ALPHA)
            act = activity(alerts, CUTOFF_ALPHA)
            ctl.append({"analysis": a.id, "block": block, "failures": int(act["fail_0.5_2"].sum()),
                        "n_nodes": len(act), "median_R": float(act["R"].median())})
            ctl_pr.append(precision_recall(alerts, CUTOFF_ALPHA).assign(analysis=a.id, block=block))
    pd.DataFrame(ctl).to_csv(PR_DIR / "label_control_2x2.csv", index=False)
    pd.concat(ctl_pr, ignore_index=True).to_csv(PR_DIR / "label_control_pr.csv", index=False)


if __name__ == "__main__":
    main()
