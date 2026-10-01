"""v6 evidence extension for the AOR revision.

Adds, without altering any frozen v5 report:
  * il-mixed     : the four-replay trading/continuation decomposition for the
                   inverse delivery-lead rule (two new replays per region).
  * current-eval : every current-curve processor candidate replayed over the
                   full evaluation period, so the 2023 selection can be scored
                   out of sample.
  * sensitivity  : one-at-a-time round-trip efficiency and wear-cost switches.

Each task is additive and idempotent: an existing audited report with an
identical policy definition is reused rather than recomputed.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

import revision_v5_control_replays as R
from revision_v5_control_solver import ROOT, OUT

REGIONS = R.REGIONS
V6 = ROOT / "results" / "revision_v6"
CURRENT_CANDIDATES = ["raw", "current_shrink_a025", "current_shrink_a050",
                      "current_smooth_a025", "current_smooth_a050"]
SENSITIVITY = [("eta", .85), ("eta", .95), ("kappa", 2.), ("kappa", 10.)]
SENSITIVITY_POLICIES = ["raw", "sparse_equal", "inverse_lead"]


def scenario_name(kind, value):
    return f"{kind}{value:g}"


P = [f"p_{j:02d}" for j in range(1, 13)]


def aligned_pair(region, policy, kind):
    """Load one policy and the frozen latest baseline on a common chronology."""
    a = R.load_inputs(region, kind="baseline", cutoff=60, policies=["raw"]).sort_values("target").reset_index(drop=True)
    b = R.load_inputs(region, kind=kind, cutoff=60, policies=[policy]).sort_values("target").reset_index(drop=True)
    assert a.target.equals(b.target), f"chronology mismatch {region}/{policy}/{kind}"
    assert np.array_equal(a.actual_00.to_numpy(), b.actual_00.to_numpy()), f"price mismatch {region}/{policy}/{kind}"
    return a, b


def il_mixed(region):
    """Two new chronological replays: inverse trading / latest tail and back.

    The latest trading block is the frozen baseline raw array, so the trading
    channel and the already completed sparse decomposition share one anchor.
    """
    a, b = aligned_pair(region, "inverse_lead", "full")
    done = []
    for head, tail in [("raw", "inverse_lead"), ("inverse_lead", "raw")]:
        name = f"trade_{head}__terminal_{tail}"
        out = a.copy()
        # Trading block is the first eight horizons only; the continuation block
        # is the final four. Substituting all twelve would silently reproduce the
        # pure alternative and destroy the allocation identity.
        if head != "raw":
            out[P[:8]] = b[P[:8]]
        if tail != "raw":
            out[P[8:]] = b[P[8:]]
        # A correct mixed array must differ from both pure arrays whenever the
        # two programs differ; an identical array would silently reproduce one
        # of them and void the channel allocation.
        assert not np.array_equal(out[P].to_numpy(float), a[P].to_numpy(float)) or \
            np.array_equal(a[P].to_numpy(float), b[P].to_numpy(float)), \
            f"mixed array equals the latest array for {name}"
        assert not np.array_equal(out[P].to_numpy(float), b[P].to_numpy(float)) or \
            np.array_equal(a[P].to_numpy(float), b[P].to_numpy(float)), \
            f"mixed array equals the history array for {name}"
        rep = R.replay(region, out, name, scenario="input_2x2")
        done.append({"policy": name, "net_value_aud": rep["net_value_aud"],
                     "forecast_array_sha256": rep["forecast_array_sha256"]})
    return {"region": region, "task": "il-mixed", "replays": done}


def current_eval(region, only_policy=None):
    """Full evaluation replay of every 2023 current-processor candidate."""
    wanted = [only_policy] if only_policy else CURRENT_CANDIDATES
    f = R.load_inputs(region, policies=wanted)
    base = R.load_inputs(region, policies=["raw"]).sort_values("target")
    done = []
    for p, g in f.groupby("policy", sort=False):
        g = g.sort_values("target")
        assert np.array_equal(g.target.to_numpy(), base.target.to_numpy())
        assert np.array_equal(g.actual_00.to_numpy(), base.actual_00.to_numpy())
        rep = R.replay(region, g, p)
        done.append({"policy": p, "net_value_aud": rep["net_value_aud"],
                     "aud_per_nominal_day": rep["value_aud_per_day"]})
    return {"region": region, "task": "current-eval", "replays": done}


def sensitivity(region, kind, value, only_policy=None):
    """Full evaluation replay under one switched efficiency or wear parameter."""
    eta, kappa = .91, 5.
    if kind == "eta":
        eta = value
    else:
        kappa = value
    scenario = scenario_name(kind, value)
    done = []
    policies = [only_policy] if only_policy else SENSITIVITY_POLICIES
    for p in policies:
        source = "baseline" if p in ("raw", "sparse_equal") else "full"
        a, g = aligned_pair(region, p, source)
        rep = R.replay(region, g, p, scenario=scenario, eta=eta, kappa=kappa)
        done.append({"policy": p, "net_value_aud": rep["net_value_aud"],
                     "definition_eta": rep["policy_definition"]["eta_c"],
                     "definition_kappa": rep["policy_definition"]["kappa_aud_per_grid_mwh"]})
    return {"region": region, "task": "sensitivity", "kind": kind, "value": value,
            "eta": eta, "kappa": kappa, "scenario": scenario, "replays": done}


def run(region, task, setting=None, policy=None):
    start = time.time()
    if task == "il-mixed":
        rep = il_mixed(region)
    elif task == "current-eval":
        rep = current_eval(region, only_policy=policy)
    elif task == "sensitivity":
        if setting:
            kind, value = next((k, v) for k, v in SENSITIVITY if scenario_name(k, v) == setting)
            rep = sensitivity(region, kind, value, only_policy=policy)
        else:
            rep = {"region": region, "task": "sensitivity", "settings": []}
            for kind, value in SENSITIVITY:
                rep["settings"].append(sensitivity(region, kind, value))
    else:
        raise ValueError(task)
    rep["seconds"] = time.time() - start
    dest = V6 / "compute"
    dest.mkdir(parents=True, exist_ok=True)
    parts = [x for x in (setting, policy) if x]
    suffix = ("_" + "_".join(parts)) if parts else ""
    out = dest / f"{region}_{task}{suffix}.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({"region": region, "task": task, "setting": setting, "policy": policy,
                      "seconds": rep["seconds"], "report": str(out)}), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    ap.add_argument("--task", choices=["il-mixed", "current-eval", "sensitivity"], required=True)
    ap.add_argument("--setting", help="single sensitivity scenario, e.g. eta0.85")
    ap.add_argument("--policy", help="single policy for the sensitivity task")
    a = ap.parse_args()
    for region in a.regions:
        run(region, a.task, a.setting, a.policy)


if __name__ == "__main__":
    main()
