"""Chronological, mean-price benchmarks for the AOOR major revision.

Owned by the forecast revision task. All derived artifacts are on D:.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from nem_price_limits import _EFFECTIVE, _VALUES

H = 12
LEVEL_COLS = [f"fcst_{j:02d}" for j in range(H)] + ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "weekday", "decision_lead_min"]
REV_COLS = [f"{v}_{j:02d}" for v in ("rev_net", "rev_abs", "rev_sd", "rev_reversals", "rev_30m", "rev_60m", "rev_120m") for j in range(H)]
FC = [f"fcst_{j:02d}" for j in range(H)]
Y = [f"actual_{j:02d}" for j in range(H)]
GRID = {"ridge": [10.0, 1000.0, 100000.0], "hgb": [7, 15, 31]}
FIXED = {"ridge": 1000.0, "hgb": 15}


def cap_paths(targets, horizon: int = H) -> np.ndarray:
    """Vectorized lookup using the shared official delivery-date cap schedule."""
    t = pd.to_datetime(targets).to_numpy(dtype="datetime64[ns]")
    delivery = t[:, None] + (np.arange(horizon) * np.timedelta64(30, "m"))[None, :]
    index = np.searchsorted(np.asarray(_EFFECTIVE,dtype="datetime64[ns]"), delivery, side="right") - 1
    if np.any(index < 0):
        raise ValueError("Delivery predates configured cap schedule")
    return np.asarray(_VALUES,dtype=np.float32)[index]


def bound_paths(prices: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    return np.maximum(-1000.0, np.minimum(prices, cap_paths(frame["target"], H))).astype(np.float32)


def design(frame: pd.DataFrame, context: str) -> np.ndarray:
    cols = LEVEL_COLS if context == "level" else LEVEL_COLS + REV_COLS
    return frame[cols].to_numpy(dtype=np.float32)


def long_design(x: np.ndarray) -> np.ndarray:
    n = len(x)
    return np.column_stack((np.repeat(x, H, axis=0), np.tile(np.arange(H, dtype=np.float32) / (H - 1), n))).astype(np.float32)


class ResidualMeanPredictor:
    """Ridge uses 12 direct outputs; HGB shares a lead-indexed squared-loss fit."""
    def __init__(self, family: str, context: str, parameter: float):
        self.family, self.context, self.parameter = family, context, parameter

    def fit(self, train: pd.DataFrame):
        error = train[Y].to_numpy(dtype=np.float32) - train[FC].to_numpy(dtype=np.float32)
        self.train_rows = len(train)
        if self.family == "basic":
            self.bias = error.mean(axis=0)
        elif self.family == "ridge":
            self.model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                       StandardScaler(), Ridge(alpha=self.parameter))
            with threadpool_limits(limits=4):
                self.model.fit(design(train, self.context), error)
        elif self.family == "hgb":
            self.model = HistGradientBoostingRegressor(loss="squared_error", learning_rate=.08,
                max_iter=80, max_leaf_nodes=int(self.parameter), min_samples_leaf=100,
                l2_regularization=10.0, max_bins=127, early_stopping=False,
                random_state=20260930)
            with threadpool_limits(limits=4):
                self.model.fit(long_design(design(train, self.context)), error.ravel())
        else:
            raise ValueError(self.family)
        return self

    def predict(self, query: pd.DataFrame, batch: int = 4000) -> np.ndarray:
        correction = np.empty((len(query), H), dtype=np.float32)
        for lo in range(0, len(query), batch):
            sub = query.iloc[lo:lo+batch]
            if self.family == "basic":
                correction[lo:lo+len(sub)] = self.bias
            elif self.family == "ridge":
                with threadpool_limits(limits=4):
                    correction[lo:lo+len(sub)] = self.model.predict(design(sub, self.context))
            else:
                with threadpool_limits(limits=4):
                    correction[lo:lo+len(sub)] = self.model.predict(long_design(design(sub, self.context))).reshape(len(sub), H)
        return bound_paths(query[FC].to_numpy(dtype=np.float32) + correction, query)


def spread_matrix(prices: np.ndarray, eta: float = .91, cost: float = 5.0) -> np.ndarray:
    """Ordered 4h spreads per internal MWh, inclusive of conversion and wear.

    Buying internal MWh needs 1/eta MWh at grid; selling returns eta MWh.
    This diagnostic evaluates opportunities; it does not assume all are feasible.
    """
    return np.column_stack([eta * prices[:, j] - prices[:, i] / eta - cost * (eta + 1 / eta)
                            for i in range(8) for j in range(i+1, 8)])


def episode_metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    diff = pred - actual
    asp, psp = spread_matrix(actual), spread_matrix(pred)
    ds = psp - asp
    aop, pop = np.maximum(asp.max(axis=1), 0), np.maximum(psp.max(axis=1), 0)
    return {"price_mae": np.abs(diff).mean(axis=1), "price_mse": (diff**2).mean(axis=1),
        "price_bias": diff.mean(axis=1), "spread_mae": np.abs(ds).mean(axis=1),
        "spread_mse": (ds**2).mean(axis=1), "spread_bias": ds.mean(axis=1),
        "spread_sign_error": ((asp > 0) != (psp > 0)).mean(axis=1),
        "best_spread_mae": np.abs(pop - aop), "best_spread_bias": pop - aop,
        "first_price_mae": np.abs(diff[:, 0])}


def summarise(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    out = {k: float(v.mean()) for k, v in metrics.items()}
    out["price_rmse"] = float(np.sqrt(out["price_mse"]))
    out["spread_rmse"] = float(np.sqrt(out["spread_mse"]))
    return out
