"""Analysis windows. New analyses use inclusive local settlement dates; the July
pipeline's analyses keep half-open UTC intervals so published values stay comparable."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class DateWindow:
    start: str
    end: str

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        d = frame["settle_date"]
        return (d >= date.fromisoformat(self.start)) & (d <= date.fromisoformat(self.end))

    @property
    def months(self) -> set[int]:
        return set(pd.date_range(self.start, self.end, freq="D").month)


@dataclass(frozen=True)
class UtcWindow:
    start: str
    end: str

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        ts = pd.to_datetime(frame["timestamp_utc"], utc=True)
        return (ts >= pd.Timestamp(self.start, tz="UTC")) & (ts < pd.Timestamp(self.end, tz="UTC"))

    @property
    def months(self) -> set[int]:
        last = pd.Timestamp(self.end) - pd.Timedelta(days=1)
        return set(pd.date_range(self.start, last, freq="D").month)
