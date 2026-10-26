"""Every analysis the revision runs (spec v2 §4.6, §5.4, §6.1-6.2; FORENSICS D1)."""
from __future__ import annotations

from dataclasses import dataclass

from .windows import DateWindow as D
from .windows import UtcWindow as U


@dataclass(frozen=True)
class Analysis:
    id: str
    train: object
    calib: object
    deploy: object
    label_train: object
    maintenance: str     # "rolling" or "frozen"
    status: str          # "legacy", "post-open", "analog", "sealed"


Q2023 = U("2023-01-01", "2024-01-01")
LEGACY = {
    "24H1-25H1": Analysis("24H1-25H1", U("2023-01-01", "2024-01-01"), U("2024-01-01", "2024-07-01"),
                          U("2025-01-01", "2025-07-01"), Q2023, "rolling", "legacy"),
    # D1: one fit, training on all rows before calibration (2023-01..2024-07). Table 2 is corrected.
    "24H2-25H2": Analysis("24H2-25H2", U("2023-01-01", "2024-07-01"), U("2024-07-01", "2025-01-01"),
                          U("2025-07-01", "2026-01-01"), Q2023, "rolling", "legacy"),
    "24H2-25H2-train2023": Analysis("24H2-25H2-train2023", U("2023-01-01", "2024-01-01"),
                                    U("2024-07-01", "2025-01-01"), U("2025-07-01", "2026-01-01"),
                                    Q2023, "rolling", "legacy"),
    "2024-2025": Analysis("2024-2025", U("2023-01-01", "2024-01-01"), U("2024-01-01", "2025-01-01"),
                          U("2025-01-01", "2026-01-01"), Q2023, "rolling", "legacy"),
    "25H1-26H1": Analysis("25H1-26H1", U("2023-01-01", "2025-01-01"), U("2025-01-01", "2025-07-01"),
                          U("2026-01-01", "2026-07-01"), Q2023, "rolling", "post-open"),
    "sealed26H1": Analysis("sealed26H1", U("2023-01-01", "2025-01-01"), U("2025-01-01", "2026-01-01"),
                           U("2026-01-01", "2026-07-01"), U("2023-01-01", "2025-01-01"),
                           "rolling", "post-open"),
    "frozen-26H1": Analysis("frozen-26H1", U("2023-01-01", "2024-01-01"), U("2024-01-01", "2024-07-01"),
                            U("2026-01-01", "2026-07-01"), Q2023, "frozen", "post-open"),
}
# frozen-origin at 2025-H1 has the same windows as 24H1-25H1 (ablation_auprc.py:60-78).

LABEL_CONTROL = {  # M3: 2x2 {full-year, H1 calibration} x {2023 q, 2023-24 q}, train 2023-24, deploy 2026-H1
    f"LC-{cal_name}-{q_name}": Analysis(
        f"LC-{cal_name}-{q_name}", U("2023-01-01", "2025-01-01"), cal, U("2026-01-01", "2026-07-01"),
        q, "rolling", "post-open")
    for cal_name, cal in (("full", U("2025-01-01", "2026-01-01")), ("h1", U("2025-01-01", "2025-07-01")))
    for q_name, q in (("q2023", Q2023), ("q2324", U("2023-01-01", "2025-01-01")))
}

ANALOGS = {
    "AQ3": Analysis("AQ3", D("2023-01-01", "2024-06-30"), D("2024-07-01", "2024-09-30"),
                    D("2025-07-01", "2025-09-30"), D("2023-01-01", "2024-06-30"), "rolling", "analog"),
    "AQ4": Analysis("AQ4", D("2023-01-01", "2024-09-30"), D("2024-10-01", "2024-12-31"),
                    D("2025-10-01", "2025-12-31"), D("2023-01-01", "2024-09-30"), "rolling", "analog"),
    "AQ1": Analysis("AQ1", D("2023-01-01", "2024-12-31"), D("2025-01-01", "2025-03-31"),
                    D("2026-01-01", "2026-03-31"), D("2023-01-01", "2024-12-31"), "rolling", "analog"),
    "AQ2": Analysis("AQ2", D("2023-01-01", "2025-03-31"), D("2025-04-01", "2025-06-30"),
                    D("2026-04-01", "2026-06-30"), D("2023-01-01", "2025-03-31"), "rolling", "analog"),
    "AQ3F": Analysis("AQ3F", D("2023-01-01", "2023-12-31"), D("2024-07-01", "2024-09-30"),
                     D("2025-07-01", "2025-09-30"), D("2023-01-01", "2023-12-31"), "frozen", "analog"),
}

SEALED = {  # definitions only; running them requires the sealed entry point (plan 2)
    "SA": Analysis("SA", D("2023-01-01", "2025-06-30"), D("2025-07-01", "2025-09-30"),
                   D("2026-07-01", "2026-09-30"), D("2023-01-01", "2025-06-30"), "rolling", "sealed"),
    "SAF": Analysis("SAF", D("2023-01-01", "2023-12-31"), D("2024-07-01", "2024-09-30"),
                    D("2026-07-01", "2026-09-30"), D("2023-01-01", "2023-12-31"), "frozen", "sealed"),
    "SB": Analysis("SB", D("2023-01-01", "2025-09-30"), D("2025-10-01", "2025-12-31"),
                   D("2026-10-01", "2026-12-31"), D("2023-01-01", "2025-09-30"), "rolling", "sealed"),
    "SBF": Analysis("SBF", D("2023-01-01", "2023-12-31"), D("2024-10-01", "2024-12-31"),
                    D("2026-10-01", "2026-12-31"), D("2023-01-01", "2023-12-31"), "frozen", "sealed"),
}
