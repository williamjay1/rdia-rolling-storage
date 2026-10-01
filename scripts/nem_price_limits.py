"""Official NEM regional reference price limits used by the study.

Values are AEMC market price caps (MPC), in nominal AUD/MWh. They are keyed
to delivery-interval timestamps, not forecast-origin timestamps. The latest
published schedule available for the study data gives $23,200/MWh from
2026-07-01; the historical replay ends before the next scheduled change.
"""

from __future__ import annotations

from datetime import timedelta
from bisect import bisect_right
from collections.abc import Iterable

import numpy as np
import pandas as pd


_EFFECTIVE = tuple(pd.Timestamp(x) for x in (
    "2015-01-01",
    "2015-07-01",
    "2016-07-01",
    "2017-07-01",
    "2018-07-01",
    "2019-07-01",
    "2020-07-01",
    "2021-07-01",
    "2022-07-01",
    "2023-07-01",
    "2024-07-01",
    "2025-07-01",
    "2026-07-01",
))
_VALUES = (13500.0, 13800.0, 14000.0, 14200.0, 14500.0, 14700.0, 15000.0,
           15100.0, 15500.0, 16600.0, 17500.0, 20300.0, 23200.0)


def market_price_cap_at(timestamp: object) -> float:
    """Return the AEMC MPC applicable at a delivery-interval timestamp."""
    ts = pd.Timestamp(timestamp)
    index = bisect_right(_EFFECTIVE, ts) - 1
    if index < 0:
        raise ValueError(f"No MPC schedule configured before {_EFFECTIVE[0].date()}: {ts}")
    return _VALUES[index]


def market_price_cap_paths(
    targets: Iterable[object], horizon_steps: int, step_minutes: int = 30
) -> np.ndarray:
    """Build an (origin, delivery-step) cap array for contiguous forecasts."""
    if horizon_steps < 1 or step_minutes < 1:
        raise ValueError("horizon_steps and step_minutes must be positive")
    starts = pd.to_datetime(list(targets))
    offsets = [timedelta(minutes=step_minutes * j) for j in range(horizon_steps)]
    return np.asarray(
        [[market_price_cap_at(start + offset) for offset in offsets] for start in starts],
        dtype=np.float32,
    )
