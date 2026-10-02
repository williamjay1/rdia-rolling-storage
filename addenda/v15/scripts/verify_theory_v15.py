"""Finite synthetic algebra checks; no market or statistical validation."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SEED = 20261002
rng = np.random.default_rng(SEED)
checks = []

def record(name, cases, maximum_error=0.0, details=None):
    assert np.isfinite(maximum_error) and maximum_error < 1e-8, (name, maximum_error)
    checks.append(dict(name=name, cases=cases, maximum_error=float(maximum_error), status="PASS", details=details))

def actions(s):
    return tuple(v for v in (-1, 0, 1) if 0 <= s + v <= 4)

def objective(prices, lam, s, increments):
    return sum(-p*v for p, v in zip(prices, increments)) + lam * (s + sum(increments))

def windows(prices, lam):
    q = {}
    w = np.empty(5)
    mu = np.empty(5, dtype=int)
    for s in range(5):
        for a in actions(s):
            q[s, a] = max(objective(prices, lam, s, (a, b)) for b in actions(s + a))
        mu[s] = max(actions(s), key=lambda a: (q[s, a], -abs(a), -a))
        w[s] = q[s, int(mu[s])]
    return q, w, mu

identity_error = oracle_error = bridge_violation = bound_violation = 0.0
different_stock_trials = 0
for trial in range(400):
    T = 6
    actual = rng.normal(0, 15, T)
    forecasts = rng.normal(0, 20, (T, 2))
    lambdas = rng.normal(0, 10, T)
    q_w_mu = [windows(forecasts[t], lambdas[t]) for t in range(T)]
    psi = rng.normal(0, 30, (T + 1, 5))
    g = rng.normal(0, 10) * np.arange(5)
    psi[-1] = g
    s_mu = s_pi = int(rng.integers(0, 5))
    j_mu = j_pi = rhs = 0.0
    bound = 0.0
    visited = []
    for t in range(T):
        q, w, mu = q_w_mu[t]
        bfun = lambda s, a: -actual[t]*a + psi[t+1, s+a]
        delta = np.array([psi[t,s] - bfun(s, int(mu[s])) for s in range(5)])
        e = {(s,a): bfun(s,a) - q[s,a] for s in range(5) for a in actions(s)}
        q_allow = max(e[s,int(mu[s])] - e[s,a] for s in range(5) for a in actions(s))
        a_pi = int(rng.choice(actions(s_pi)))
        a_mu = int(mu[s_mu])
        same_stock_mu = int(mu[s_pi])
        d = bfun(s_pi, same_stock_mu) - bfun(s_pi, a_pi)
        rho = w[s_pi] - q[s_pi,a_pi]
        bridge_violation = max(bridge_violation, d-rho-q_allow)
        rhs += d + delta[s_pi] - delta[s_mu]
        bound += rho + q_allow + np.ptp(delta)
        j_pi += -actual[t]*a_pi
        j_mu += -actual[t]*a_mu
        visited.append((s_pi,a_pi))
        s_pi += a_pi
        s_mu += a_mu
        different_stock_trials += int(s_pi != s_mu)
    j_pi += g[s_pi]
    j_mu += g[s_mu]
    identity_error = max(identity_error, abs((j_mu-j_pi)-rhs))
    bound_violation = max(bound_violation, j_mu-j_pi-bound)
    # Independent reference-value recursion; future-stream audit only.
    V = np.zeros((T+1,5))
    V[-1] = g
    for t in reversed(range(T)):
        mu = q_w_mu[t][2]
        for s in range(5):
            a = int(mu[s])
            V[t,s] = -actual[t]*a + V[t+1,s+a]
    oracle_rhs = 0.0
    for t,(s,a) in enumerate(visited):
        am = int(q_w_mu[t][2][s])
        oracle_rhs += (-actual[t]*am + V[t+1,s+am]) - (-actual[t]*a + V[t+1,s+a])
    oracle_error = max(oracle_error, abs(j_mu-j_pi-oracle_rhs))
record("separate-path residual identity",400,identity_error,dict(different_stock_origin_pairs=different_stock_trials))
record("reference-policy value oracle identity",400,oracle_error)
record("window-to-realized bridge inequality",2400,max(0.0,bridge_violation))
record("joint-cover cumulative upper bound",400,max(0.0,bound_violation))

clip_budget_error = reward_violation = power_violation = mode_violation = 0.0
for case in range(1500):
    H = int(rng.integers(2,15))
    lo, hi, P, dt = 0.2,1.8,1.0,0.5
    ec, ed = float(rng.uniform(.8,.99)),float(rng.uniform(.8,.99))
    s, sp = float(rng.uniform(lo,hi)),float(rng.uniform(lo,hi))
    s0, sp0 = s, sp
    v, vp = [],[]
    for j in range(H):
        vl,vu = max(-P*dt/ed,lo-s),min(ec*dt*P,hi-s)
        vv = float(rng.uniform(vl,vu))
        s += vv
        nsp = float(np.clip(sp+vv,lo,hi))
        vvp = nsp-sp
        sp=nsp
        v.append(vv);vp.append(vvp)
        power_violation = max(power_violation, vvp-ec*dt*P, -vvp-P*dt/ed)
        mode_violation=max(mode_violation,-vv*vvp)
    v=np.array(v);vp=np.array(vp)
    clip_budget_error=max(clip_budget_error,abs(np.sum(np.abs(v-vp))+abs(s-sp)-abs(s0-sp0)))
    prices=rng.normal(0,100,H); kappa=float(rng.uniform(0,10));lam=float(rng.normal(0,100))
    R=lambda inc: np.where(inc>=0,-(prices+kappa)*inc/ec,-(prices-kappa)*ed*inc)
    beta=np.maximum(np.abs(prices+kappa)/ec,ed*np.abs(prices-kappa))
    L=max(float(beta.max()),abs(lam))
    gap=abs(float(np.sum(R(v)-R(vp))+lam*(s-sp)))
    reward_violation=max(reward_violation,gap-L*abs(s0-sp0))
record("asymmetric-efficiency clipping budget",1500,clip_budget_error)
record("clipped plan reward bound including negative prices",1500,max(0.0,reward_violation))
record("mode-preserving clipped feasibility",1500,max(0.0,power_violation,mode_violation))

# Exact finite-grid counterpart of W Lipschitz and the 2Lh anchor bound.
lip_violation=anchor_violation=0.0
for case in range(120):
    prices=rng.normal(0,20,3);lam=float(rng.normal(0,15));L=max(max(abs(prices)),abs(lam))
    plans={s:[seq for seq in itertools.product((-1,0,1),repeat=3) if all(0<=s+sum(seq[:j])<=4 for j in range(1,4))] for s in range(5)}
    W={s:max(objective(prices,lam,s,x) for x in plans[s]) for s in range(5)}
    for si in range(5):
        x=max(plans[si],key=lambda seq:objective(prices,lam,si,seq))
        for s in range(5):
            lip_violation=max(lip_violation,abs(W[s]-W[si])-L*abs(s-si))
            st=s;xp=[]
            for a in x:
                ns=min(4,max(0,st+a));xp.append(ns-st);st=ns
            a=xp[0]
            Q=max(objective(prices,lam,s,y) for y in plans[s] if y[0]==a)
            anchor_violation=max(anchor_violation,W[s]-Q-2*L*abs(s-si))
record("enumerated optimal-value continuity",3000,max(0.0,lip_violation))
record("enumerated clipped first-action 2Lh bound",3000,max(0.0,anchor_violation))

# Explicit clock/action successor quantifier, including fallback.
nodes=("on_time","delayed")
cover={(t,node):set(range(5)) for t in range(4) for node in nodes}
for t in range(3):
    for node in nodes:
        for s in cover[t,node]:
            for a in actions(s):
                for nxt in nodes:
                    assert s+a in cover[t+1,nxt]
negative_cover={1}
assert 1+0 in negative_cover and 1-1 not in negative_cover
record("successor closure includes all clock successors and fallback",3*2*13*2,0,dict(negative_case="Cover {1} closes under idle but fails under admitted discharge fallback."))

# Zero window regret cannot imply zero realized cash regret.
q,w,mu=windows((0.0,0.0),0.0)
s=1; a_reference=int(mu[s]); a_candidate=1
q_reduced,w_reduced,mu_reduced=windows((-1.0,0.0),0.0)
assert a_reference==0 and int(mu_reduced[s])==a_candidate
rho=float(w[s]-q[s,a_candidate]); actual_price=50.0
loss=-actual_price*a_reference-(-actual_price*a_candidate)
assert rho==0.0 and loss==50.0
record("zero local window regret has no unconditional cash guarantee",1,0,dict(window_regret=rho,realized_loss=loss,reference_curve=[0,0],candidate_curve=[-1,0],reference_increment=a_reference,candidate_increment=a_candidate,shared_tie_rule=True))

source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
out=dict(status="PASS",seed=SEED,scope="Executed finite synthetic algebra checks; no empirical guarantee or runtime q_t/b_t certificate.",checks=checks,script_sha256=source_sha)
target=ROOT/"executed_validation_v15.json"
target.write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(dict(status=out["status"],groups=len(checks),output=str(target),maximum_error=max(c["maximum_error"] for c in checks))))
