"""Small controlled RDIA checks, not a synthetic electricity-market study.

Freeze the prespecified inputs with --plan-only, then execute the unchanged
design. A separate scalar closed form audits every one-step LP replay. The
additional perturbation checks use 1/2/3-step compact convex storage programs.
No v6 source or observed-market result is read or modified.
"""
from __future__ import annotations

import os
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/revision_v7/synthetic"
SEED = 20261003
PRIMARY_TIE_ALLOWANCE = 1e-10
CHECK_TOLERANCE = 1e-6
EPSILON = .01
ACTION_MARGIN = .1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False), encoding="utf-8")


def csv_write(path, records):
    fields = list(records[0]) if records else []
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def case_designs():
    base = {"s0": 1., "actual": [4., 3., 2.], "endpoint_mark": 1.,
            "p_R": [0., 0., 0.], "p_H": [0., 0., 0.],
            "theta_R": [1., 1., 1.], "theta_H": [1., 1., 1.],
            "alternate_B_selector": "min"}

    def case(name, label, condition, **updates):
        return {**base, **updates, "case": name, "prespecified_label": label,
                "input_ground_truth_condition": condition}

    return [
        case("trade_only", "trading input only",
             "Terminal arrays identical; p_R-theta<0 and p_H-theta>0 at full initial inventory.",
             p_H=[3., 3., 3.]),
        case("terminal_only", "terminal input only",
             "Trading arrays identical; p-theta_R>0 and p-theta_H<0 at full initial inventory.",
             p_R=[1., 1., 1.], p_H=[1., 1., 1.], theta_R=[0., 0., 0.],
             theta_H=[3., 3., 3.]),
        case("interaction", "crossed threshold interaction",
             "Trading change crosses the sale threshold under theta_R, but theta_H reverses that ranking.",
             p_H=[2., 2., 2.], theta_H=[3., 3., 3.]),
        case("near_tie", "epsilon ambiguous action response",
             "Opposite objective slopes; each full feasible action range loses no more than epsilon=.01.",
             s0=.5, p_R=[.995, .995, .995], p_H=[1.005, 1.005, 1.005]),
        case("update_reversal", "delay loses value under subsequent fixed updates",
             "theta_H declines 20->3->0; it induces hold, hold, sell while R sells initially; realised prices are 10,2,6.",
             actual=[10., 2., 6.], endpoint_mark=0.,
             p_R=[10., 2., 6.], p_H=[10., 2., 6.], theta_R=[0., 0., 0.],
             theta_H=[20., 3., 0.]),
        case("null", "changed forecasts, unchanged strict action ranking",
             "All p-theta slopes remain strictly positive; both policies sell the full initial stock at the first origin.",
             p_R=[3., 4., 5.], p_H=[4., 6., 7.], theta_R=[.5, .5, .5],
             theta_H=[1., 1., 1.]),
        case("exact_tie_control", "secondary selector only",
             "The two primary objectives are identical and flat. B deliberately compares two valid primary-optimal selectors; A/C keep the same selector.",
             s0=.5, p_R=[1., 1., 1.], p_H=[1., 1., 1.],
             alternate_B_selector="max"),
    ]


def validate_input_labels(cases):
    """Labels are verified from inputs alone, before any solver is called."""
    facts = {}
    for c in cases:
        pr, ph = np.asarray(c["p_R"]), np.asarray(c["p_H"])
        tr, th = np.asarray(c["theta_R"]), np.asarray(c["theta_H"])
        name = c["case"]
        if name == "trade_only":
            valid = np.array_equal(tr, th) and np.all(pr-tr < 0) and np.all(ph-tr > 0)
        elif name == "terminal_only":
            valid = np.array_equal(pr, ph) and np.all(pr-tr > 0) and np.all(pr-th < 0)
        elif name == "interaction":
            valid = all((pr[0]-tr[0] < 0, ph[0]-tr[0] > 0,
                         pr[0]-th[0] < 0, ph[0]-th[0] < 0))
        elif name == "near_tie":
            valid = pr[0]-tr[0] < 0 < ph[0]-th[0] and \
                max(abs(pr[0]-tr[0]), abs(ph[0]-th[0])) <= EPSILON
        elif name == "update_reversal":
            valid = all((th[0] > ph[0], th[1] > ph[1], th[2] < ph[2],
                         c["actual"][0] > c["actual"][2] > c["actual"][1],
                         np.all(pr-tr > 0)))
        elif name == "null":
            valid = np.all(pr-tr > 0) and np.all(ph-th > 0) and \
                not np.array_equal(pr, ph)
        elif name == "exact_tie_control":
            valid = np.array_equal(pr, ph) and np.array_equal(tr, th) and \
                np.all(pr-tr == 0) and c["alternate_B_selector"] == "max"
        else:
            valid = False
        facts[name] = {"input_condition_verified": bool(valid),
                       "condition": c["input_ground_truth_condition"]}
        if not valid:
            raise AssertionError(f"Invalid prespecified input label: {name}")
    return facts


def lp_data(prices, theta, s0):
    """Compact ideal storage LP: net discharge u, s'=s-u, no hidden modes."""
    prices = np.asarray(prices, dtype=float)
    h = len(prices)
    reward = np.r_[prices, np.zeros(h-1), float(theta)]
    eq = np.zeros((h, 2*h))
    for j in range(h):
        eq[j, j] = 1.
        eq[j, h+j] = 1.
        if j:
            eq[j, h+j-1] = -1.
    rhs = np.r_[float(s0), np.zeros(h-1)]
    return {"reward": reward, "eq": eq, "rhs": rhs,
            "bounds": [(-1., 1.)]*h + [(0., 1.)]*h, "h": h}


def lp_run(data, cost, extra_rows=(), extra_rhs=()):
    result = linprog(cost, A_ub=np.asarray(extra_rows) if len(extra_rows) else None,
                     b_ub=np.asarray(extra_rhs) if len(extra_rhs) else None,
                     A_eq=data["eq"], b_eq=data["rhs"], bounds=data["bounds"],
                     method="highs", options={"presolve": True,
                         "primal_feasibility_tolerance": 1e-9,
                         "dual_feasibility_tolerance": 1e-9})
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(f"LP failure {result.status}: {result.message}")
    return result


def solve(prices, theta, s0, selector="min", epsilon=EPSILON):
    data = lp_data(prices, theta, s0)
    reward, h = data["reward"], data["h"]
    primary = lp_run(data, -reward)
    if primary is None:
        raise RuntimeError("Primary storage program infeasible")
    value = float(reward @ primary.x)
    first = np.zeros(2*h)
    first[0] = 1.
    selected = lp_run(data, first if selector == "min" else -first,
                      [-reward], [-value + PRIMARY_TIE_ALLOWANCE])
    low = lp_run(data, first, [-reward], [-value + epsilon])
    high = lp_run(data, -first, [-reward], [-value + epsilon])
    if any(x is None for x in (selected, low, high)):
        raise RuntimeError("Near-optimal action set unexpectedly infeasible")
    return {"value": value, "x": selected.x, "primary_x": primary.x,
            "reward": reward, "first_action": float(selected.x[0]),
            "next_state": float(selected.x[h]),
            "range_low": float(low.x[0]), "range_high": float(high.x[0]),
            "primary_loss": float(value-reward @ selected.x),
            "max_state_balance_error": float(np.max(abs(data["eq"] @ selected.x-data["rhs"]))),
            "data": data}


def closed_form(p, theta, s0, selector="min", epsilon=EPSILON):
    """Independent scalar oracle: no LP call or attribution result used."""
    lower, upper = max(-1., s0-1.), min(1., s0)
    slope = float(p-theta)
    action = upper if slope > 0 else lower if slope < 0 else \
        (lower if selector == "min" else upper)
    if slope > 0:
        rlo, rhi = max(lower, upper-epsilon/slope), upper
    elif slope < 0:
        rlo, rhi = lower, min(upper, lower+epsilon/abs(slope))
    else:
        rlo, rhi = lower, upper
    return {"first_action": action, "next_state": s0-action,
            "value": theta*s0+slope*action, "range_low": rlo,
            "range_high": rhi}


def replay(case, trading, terminal, first_only=False, analytic=False):
    s = float(case["s0"])
    rows = []
    for t, actual in enumerate(case["actual"]):
        head = "R" if first_only and t else trading
        tail = "R" if first_only and t else terminal
        p, theta = case["p_"+head][t], case["theta_"+tail][t]
        sol = closed_form(p, theta, s) if analytic else solve([p], theta, s)
        u, after = sol["first_action"], sol["next_state"]
        rows.append({"origin": t, "state_before": s, "action": u,
                     "state_after": after, "issued_p": p, "issued_theta": theta,
                     "actual_price": actual, "net_cash": actual*u})
        s = after
    cash = sum(row["net_cash"] for row in rows)
    opening_mark = case["actual"][0]*case["s0"]
    closing_mark = case["endpoint_mark"]*s
    value = cash+closing_mark-opening_mark
    return {"value": float(value), "cash": float(cash), "state_end": float(s),
            "opening_mark": opening_mark, "closing_mark": closing_mark,
            "trajectory": rows}


def allocation(values):
    j00, j10, j01, j11 = [values[k] for k in ("RR", "HR", "RH", "HH")]
    return {"T": .5*((j10-j00)+(j11-j01)),
            "C": .5*((j01-j00)+(j11-j10)),
            "I": j11-j10-j01+j00, "D": j11-j00}


def run_case(case):
    name = case["case"]
    paths, oracle_paths = {}, {}
    for head, tail in (("R", "R"), ("H", "R"), ("R", "H"), ("H", "H")):
        paths[head+tail] = replay(case, head, tail)
        oracle_paths[head+tail] = replay(case, head, tail, analytic=True)
    values = {key: value["value"] for key, value in paths.items()}
    reference = {key: value["value"] for key, value in oracle_paths.items()}
    components, expected = allocation(values), allocation(reference)
    ranges, range_rows = {}, []
    range_error = 0.
    for head, tail in (("R", "R"), ("H", "R"), ("R", "H"), ("H", "H")):
        selector = case["alternate_B_selector"] if head+tail == "HH" else "min"
        sol = solve([case["p_"+head][0]], case["theta_"+tail][0], case["s0"], selector)
        ref = closed_form(case["p_"+head][0], case["theta_"+tail][0], case["s0"], selector)
        for field in ("first_action", "value", "range_low", "range_high"):
            range_error = max(range_error, abs(sol[field]-ref[field]))
        ranges[head+tail] = {key: float(sol[key]) for key in
                           ("first_action", "range_low", "range_high", "value", "primary_loss")}
        range_rows.append({"case": name, "program": head+tail, "state": case["s0"],
            "selector": selector, "epsilon": EPSILON, **ranges[head+tail]})
    r, h = ranges["RR"], ranges["HH"]
    interval_separation = max(0., h["range_low"]-r["range_high"],
                              r["range_low"]-h["range_high"])
    change = abs(h["first_action"]-r["first_action"])
    if name == "exact_tie_control":
        b_class = "tie only: identical primary programs, different valid secondary selectors"
    elif change <= ACTION_MARGIN:
        b_class = "no material joint first-action change"
    elif interval_separation > ACTION_MARGIN:
        b_class = "separated epsilon-optimal action ranges"
    else:
        b_class = "epsilon ambiguous; overlap alone does not prove tie only"
    branch = {"raw": paths["RR"], "first_only": replay(case, "H", "H", first_only=True),
              "continuous": paths["HH"]}
    ref_branch = {"raw": oracle_paths["RR"],
                  "first_only": replay(case, "H", "H", first_only=True, analytic=True),
                  "continuous": oracle_paths["HH"]}
    path_error = max(abs(values[k]-reference[k]) for k in values)
    state_error, cash_error, action_error = 0., 0., 0.
    for key in paths:
        for row, ref in zip(paths[key]["trajectory"], oracle_paths[key]["trajectory"]):
            action_error = max(action_error, abs(row["action"]-ref["action"]))
            state_error = max(state_error, abs(row["state_after"]-ref["state_after"]))
            cash_error = max(cash_error, abs(row["net_cash"]-ref["net_cash"]))
    branch_error = max(abs(branch[k]["value"]-ref_branch[k]["value"]) for k in branch)
    recovered = path_error <= CHECK_TOLERANCE and range_error <= CHECK_TOLERANCE \
        and branch_error <= CHECK_TOLERANCE
    summary = {"case": name, "prespecified_label": case["prespecified_label"],
        **{"J_"+key: value for key, value in values.items()}, **components,
        "A_identity_error": abs(components["T"]+components["C"]-components["D"]),
        "B_raw_action": r["first_action"], "B_history_action": h["first_action"],
        "B_raw_low": r["range_low"], "B_raw_high": r["range_high"],
        "B_history_low": h["range_low"], "B_history_high": h["range_high"],
        "B_range_separation": interval_separation, "B_classification": b_class,
        "C_first_only_minus_raw": branch["first_only"]["value"]-branch["raw"]["value"],
        "C_continuous_minus_raw": branch["continuous"]["value"]-branch["raw"]["value"],
        "max_analytic_value_error": path_error, "max_analytic_range_error": range_error,
        "max_analytic_branch_error": branch_error, "recovery_PASS": recovered}
    return summary, {"design": case,
        "fixed_exogenous_stream_sha256": {key: array_sha(case[key]) for key in
            ("actual", "p_R", "p_H", "theta_R", "theta_H")}, "A_paths": paths,
        "A_scalar_reference_paths": oracle_paths, "A_components": components,
        "A_scalar_reference_components": expected, "B_ranges": ranges,
        "C_branches": branch, "C_scalar_reference_branches": ref_branch,
        "numeric_errors": {"value": path_error, "range_or_action": range_error,
                           "branch": branch_error, "state": state_error,
                           "cash": cash_error, "action": action_error}}, range_rows


def excluded_action_gap(data, optimum, first_action, radius=ACTION_MARGIN):
    """Maximize reference reward outside an action band; union of two LPs."""
    candidates = []
    first = np.zeros(len(data["reward"]))
    first[0] = 1.
    for row, bound in ((first, first_action-radius), (-first, -first_action-radius)):
        q = lp_run(data, -data["reward"], [row], [bound])
        if q is not None:
            candidates.append(float(data["reward"] @ q.x))
    return float(optimum-max(candidates)) if candidates else None


def perturbation_checks(n_trials):
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(n_trials):
        horizon = 1+i % 3
        s0 = float(rng.uniform(.05, .95))
        p_a = rng.uniform(-2, 3, size=horizon)
        p_b = p_a+rng.normal(0, .75, size=horizon)
        theta_ref = float(rng.uniform(-1, 2))
        delta = float(rng.uniform(-.8, .8))
        delta_b = float(rng.uniform(-.8, .8))
        ref_a, hat_a = solve(p_a, theta_ref, s0), solve(p_a, theta_ref+delta, s0)
        ref_b, hat_b = solve(p_b, theta_ref, s0), solve(p_b, theta_ref+delta, s0)
        different_hat_b = solve(p_b, theta_ref+delta_b, s0)
        # Every terminal SOC in [0,1] is reachable at these parameters, so E=M=1.
        E, absolute_S_bound = 1., 1.
        solve_epsilon = max(0., hat_a["primary_loss"])
        regret = max(0., ref_a["value"]-float(ref_a["reward"] @ hat_a["x"]))
        regret_bound = abs(delta)*E+solve_epsilon
        distortion = abs((hat_a["value"]-hat_b["value"])-
                         (ref_a["value"]-ref_b["value"]))
        different_distortion = abs((hat_a["value"]-different_hat_b["value"])-
                                   (ref_a["value"]-ref_b["value"]))
        different_bound = (abs(delta)+abs(delta_b))*absolute_S_bound
        gap = excluded_action_gap(ref_a["data"], ref_a["value"],
                                  float(ref_a["primary_x"][0]))
        certificate = gap is not None and gap > regret_bound+CHECK_TOLERANCE
        action_error = abs(hat_a["first_action"]-float(ref_a["primary_x"][0]))
        expanded = solve(p_a, theta_ref, s0, epsilon=EPSILON+abs(delta)*E)
        inclusion_violation = max(0., expanded["range_low"]-hat_a["range_low"],
                                  hat_a["range_high"]-expanded["range_high"])
        analytic_error = 0.
        if horizon == 1:
            scalar = closed_form(float(p_a[0]), theta_ref, s0)
            analytic_error = abs(ref_a["value"]-scalar["value"])
        rows.append({"trial": i, "horizon": horizon, "s0": s0,
            "theta_reference_proxy": theta_ref, "delta": delta, "delta_B": delta_b,
            "terminal_state_amplitude_E": E, "absolute_terminal_state_bound_M": absolute_S_bound,
            "regret": regret, "regret_bound_with_solve_epsilon": regret_bound,
            "common_delta_value_contrast_distortion": distortion,
            "common_delta_bound": abs(delta)*E,
            "different_delta_value_contrast_distortion": different_distortion,
            "different_delta_absolute_S_bound": different_bound,
            "outside_action_gap": gap, "action_stability_certificate": bool(certificate),
            "action_distance": action_error, "epsilon_range_inclusion_violation": inclusion_violation,
            "scalar_value_error_if_horizon_1": analytic_error,
            "state_balance_error": max(ref_a["max_state_balance_error"],
                hat_a["max_state_balance_error"], ref_b["max_state_balance_error"],
                hat_b["max_state_balance_error"]),
            "regret_bound_violation": max(0., regret-regret_bound),
            "common_contrast_bound_violation": max(0., distortion-abs(delta)*E),
            "different_contrast_bound_violation": max(0., different_distortion-different_bound),
            "certified_action_violation": max(0., action_error-ACTION_MARGIN) if certificate else 0.})
    # E, rather than 2E, is the sharp generic constant: this approaches the bound.
    tight_ref = solve([1.999], 1., 1.)
    tight_hat = solve([1.999], 2., 1.)
    tight_regret = tight_ref["value"]-float(tight_ref["reward"] @ tight_hat["x"])
    summary = {"trials": n_trials, "seed": SEED, "horizons": [1, 2, 3],
        "gap_certified_trials": sum(x["action_stability_certificate"] for x in rows),
        "tight_example": {"theta_reference_proxy": 1., "theta_approximate": 2.,
                          "price": 1.999, "E": 1., "bound": 1.,
                          "observed_reference_objective_regret": tight_regret,
                          "regret_to_bound_ratio": tight_regret},
        "maxima": {key: max(row[key] for row in rows) for key in
            ("regret_bound_violation", "common_contrast_bound_violation",
             "different_contrast_bound_violation", "certified_action_violation",
             "epsilon_range_inclusion_violation", "scalar_value_error_if_horizon_1",
             "state_balance_error")}}
    summary["PASS"] = all(value <= CHECK_TOLERANCE for value in summary["maxima"].values())
    return summary, rows


def freeze_plan(out, n_trials):
    design = {"status": "frozen before execution", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_sha256": sha(__file__), "synthetic_not_observed_market": True,
        "scientific_scope": "verification of controlled effects and elementary finite-window perturbation bounds, not market generalization or a new value-of-information estimator",
        "model": {"capacity": 1., "net_power": 1., "duration": 1., "efficiency": 1.,
                  "wear": 0., "state_bounds": [0., 1.], "trading_horizon": 1,
                  "replay_origins_per_case": 3, "objective": "issued_p*u + issued_theta*s_end",
                  "evaluation": "sum actual_price*u + endpoint_mark*s_end - actual_price[0]*s0"},
        "epsilon": EPSILON, "material_action_margin": ACTION_MARGIN,
        "primary_selector_allowance": PRIMARY_TIE_ALLOWANCE,
        "fixed_seed": SEED, "perturbation_trials": n_trials, "comparison_tolerance": CHECK_TOLERANCE,
        "cases": case_designs()}
    design["input_conditions"] = validate_input_labels(design["cases"])
    path = out / "prespecified_design.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["script_sha256"] != design["script_sha256"] or old["cases"] != design["cases"] \
                or old["perturbation_trials"] != n_trials:
            raise RuntimeError("Existing design differs: preserve it and choose another fresh --output directory")
        return old
    save_json(path, design)
    return design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--perturbation-trials", type=int, default=200)
    args = parser.parse_args()
    out = args.output.resolve()
    if os.name == "nt" and out.drive.upper() != "D:":
        parser.error("All computation outputs must remain on D:")
    if args.perturbation_trials < 3:
        parser.error("At least three perturbation trials are required")
    out.mkdir(parents=True, exist_ok=True)
    if args.plan_only:
        design = freeze_plan(out, args.perturbation_trials)
        print(json.dumps({"status": "design frozen; no solver executed", "cases": len(design["cases"]),
                          "design_sha256": sha(out / "prespecified_design.json"), "output": str(out)}))
        return
    design_path = out / "prespecified_design.json"
    if not design_path.is_file():
        parser.error("First run --plan-only to preserve the design before solver execution")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if design["script_sha256"] != sha(__file__) or design["cases"] != case_designs() \
            or design["perturbation_trials"] != args.perturbation_trials:
        raise RuntimeError("Script/design changed after freeze; choose a fresh output and refreeze")
    if (out / "execution_receipt.json").exists():
        raise RuntimeError("Execution receipt already exists; preserve it and use a fresh --output")
    validate_input_labels(design["cases"])
    start = time.perf_counter()
    summaries, details, ranges, failures = [], {}, [], []
    for case in design["cases"]:
        try:
            summary, detail, range_rows = run_case(case)
            summaries.append(summary)
            details[case["case"]] = detail
            ranges.extend(range_rows)
            if not summary["recovery_PASS"]:
                failures.append({"case": case["case"], "error": "scalar reference mismatch"})
        except Exception as exc:
            failures.append({"case": case["case"], "error": repr(exc)})
            details[case["case"]] = {"design": case, "execution_error": repr(exc)}
    try:
        bound_summary, bound_rows = perturbation_checks(args.perturbation_trials)
        if not bound_summary["PASS"]:
            failures.append({"scope": "perturbation checks", "error": bound_summary["maxima"]})
    except Exception as exc:
        bound_summary, bound_rows = {"PASS": False, "execution_error": repr(exc)}, []
        failures.append({"scope": "perturbation checks", "error": repr(exc)})
    save_json(out / "case_details.json", details)
    csv_write(out / "case_summary.csv", summaries)
    csv_write(out / "common_state_ranges.csv", ranges)
    save_json(out / "perturbation_checks.json", bound_summary)
    csv_write(out / "perturbation_trials.csv", bound_rows)
    maxima = {key: max((v["numeric_errors"][key] for v in details.values() if "numeric_errors" in v), default=None)
              for key in ("value", "range_or_action", "branch", "state", "cash", "action")}
    identity_max = max((x["A_identity_error"] for x in summaries), default=None)
    receipt = {"status": "PASS" if not failures else "FAILED; all failed cases retained",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter()-start, "python": sys.executable,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "solver": "scipy.optimize.linprog, HiGHS global LP",
        "script_sha256": sha(__file__), "design_sha256": sha(design_path),
        "synthetic_not_observed_market": True, "cases_planned": len(design["cases"]),
        "cases_completed": len(summaries), "cases_scalar_reference_PASS": sum(x["recovery_PASS"] for x in summaries),
        "A_rolling_paths": 4*len(summaries), "A_origins_each_path": 3,
        "B_common_state_programs": 4*len(summaries), "B_epsilon_range_extreme_LPs": 8*len(summaries),
        "C_branches": 3*len(summaries), "C_origins_each_branch": 3,
        "A_max_identity_error": identity_max, "max_scalar_reference_errors": maxima,
        "bound_checks": bound_summary, "failures": failures,
        "claims_not_supported": ["market effect prevalence", "out-of-sample market generalization",
                                 "unconditional nonconvex optimization guarantees",
                                 "unknown exact dynamic continuation values",
                                 "rolling cumulative cash regret bounded by one-window delta*E"],
        "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                  for p in sorted(out.iterdir()) if p.is_file()}}
    save_json(out / "execution_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "cases_PASS": receipt["cases_scalar_reference_PASS"],
                      "A_max_identity_error": identity_max, "max_scalar_reference_errors": maxima,
                      "bound_check_maxima": bound_summary.get("maxima"),
                      "elapsed_seconds": receipt["elapsed_seconds"], "receipt": str(out / "execution_receipt.json")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
