"""Executed synthetic controls for rolling-sufficiency conditions.

This is a diagnostic, not a trained compression algorithm or empirical market
experiment.  It imports the unchanged v13 exclusive-mode storage controller.
All outputs are confined to results/revision_v14/theory.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from revision_v5_control_enumerated_policy import EnumeratedSafeLexMPC as MPC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/revision_v14/theory"


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def solve_summary(solver, curve, soc):
    sol = solver.solve(np.asarray(curve, float), float(soc))
    assert sol is not None
    return {k: float(sol[k]) for k in ["c", "d", "u", "soc", "value",
        "primary_optimum_incumbent_aud", "policy_optimum_loss_upper_aud"]}


def reward(v, price, eta=.91, kappa=5.):
    """Stored-energy increment v in MWh -> energy-only net AUD."""
    if v >= 0:
        return -(price + kappa) * v / eta
    return -(price - kappa) * eta * v


def clip_plan(increments, initial, lo=.2, hi=1.8):
    states = [float(initial)]
    adjusted = []
    for v in increments:
        nxt = min(hi, max(lo, states[-1] + float(v)))
        adjusted.append(nxt-states[-1])
        states.append(nxt)
    return np.array(adjusted), np.array(states)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start=time.perf_counter()
    checks=[]
    def check(name, condition, details=None):
        if not bool(condition):
            raise AssertionError(name)
        checks.append({"name":name,"pass":True,"details":details})

    # The first opening signal is cheaper than its forecast sale opportunity.
    # Distinct continuation coordinates have the same mean, so the represented
    # initial objective is identical and the first action is EXACTLY identical.
    initial_full=np.r_[10.,np.full(7,100.),np.zeros(4)]
    initial_comp=np.r_[10.,np.full(7,100.),[-10.,0.,0.,10.]]
    next_full=np.r_[100.,np.zeros(7),np.zeros(4)]
    next_comp=np.r_[100.,np.full(7,400.),np.full(4,400.)]
    actual=np.array([10.,100.]);final_mark=0.
    branches={}
    for name, curves in {"full":[initial_full,next_full],
                         "compressed":[initial_comp,next_comp]}.items():
        solver=MPC();soc=1.;rows=[]
        for t, curve in enumerate(curves):
            s0=soc;sol=solve_summary(solver,curve,soc);soc=sol["soc"]
            cash=.5*((actual[t]-5)*sol["d"]-(actual[t]+5)*sol["c"])
            rows.append({"stage":t+1,"soc_start_mwh":s0,
                "curve_aud_per_mwh":curve.tolist(),"realized_price_aud_per_mwh":float(actual[t]),
                "net_cash_aud":float(cash),**sol})
        branches[name]={"stages":rows,"total_net_cash_aud":sum(r["net_cash_aud"] for r in rows),
                        "marked_value_aud":sum(r["net_cash_aud"] for r in rows)+final_mark*soc,
                        "final_soc_mwh":soc}
    full=branches["full"];comp=branches["compressed"]
    check("same_first_action_exact",full["stages"][0]["u"]==comp["stages"][0]["u"])
    check("same_state_after_same_first_action_exact",full["stages"][0]["soc"]==comp["stages"][0]["soc"])
    check("different_second_action",abs(full["stages"][1]["u"]-comp["stages"][1]["u"])>.5)
    check("different_state_only_after_second_action",abs(full["final_soc_mwh"]-comp["final_soc_mwh"])>.5)
    check("nonzero_realized_two_step_loss",full["marked_value_aud"]-comp["marked_value_aud"]>50.)

    # Stronger state-only control: each representation supplies the SAME curve
    # template at both origins. Initial action compatibility fails at the next
    # inventory even without a larger newly introduced coefficient error.
    fixed_full=np.r_[50.,np.zeros(11)]
    fixed_comp=np.r_[50.,100.,np.zeros(10)]
    state_only={}
    for name,curve in {"full":fixed_full,"compressed":fixed_comp}.items():
        model=MPC();stock=1.4;rows=[]
        for t in range(2):
            old=stock;sol=solve_summary(model,curve,stock);stock=sol['soc']
            cash=.5*((50.-5.)*sol['d']-(50.+5.)*sol['c'])
            rows.append({'stage':t+1,'soc_start_mwh':old,'net_cash_aud':cash,**sol})
        state_only[name]={'curve_aud_per_mwh':curve.tolist(),'stages':rows,
                         'marked_value_aud':sum(x['net_cash_aud'] for x in rows),'terminal_mark':0.}
    sf,sc=state_only['full'],state_only['compressed']
    ideal_s1=1.4-.5/.91
    ideal_comp_d2=2*.91*(ideal_s1-.2)-1.
    ideal_loss=.5*45.*(1.-ideal_comp_d2)
    state_only['ideal_exact_primary']={'first_discharge_mw':1.,'common_next_soc_mwh':ideal_s1,
        'second_reference_discharge_mw':1.,'second_compressed_discharge_mw':ideal_comp_d2,
        'full_minus_compressed_marked_aud':ideal_loss,
        'analytical_reason':'Both initial stocks can fund first full discharge plus one future full discharge; at next stock the higher forecast future price reserves inventory.'}
    cover_lower=.2+2*.5/.91
    state_only['action_preserving_cover']={'lower_soc_mwh':cover_lower,'upper_soc_mwh':1.8,
        'image_lower_soc_mwh':cover_lower-.5/.91,'image_upper_soc_mwh':1.8-.5/.91,
        'image_is_disjoint_from_cover':1.8-.5/.91<cover_lower,
        'compression_future_high_price_step':2,
        'zero_price_steps_after_peak_never_improve_reward_by_charging_when_terminal_zero':True}
    state_only['executed_first_action_difference_mw']=abs(sf['stages'][0]['u']-sc['stages'][0]['u'])
    state_only['executed_full_minus_compressed_marked_aud']=sf['marked_value_aud']-sc['marked_value_aud']
    check('state_only_initial_actions_agree_within_one_ulp',state_only['executed_first_action_difference_mw']<1e-14)
    check('state_only_second_actions_separate_materially',abs(sf['stages'][1]['u']-sc['stages'][1]['u'])>.8)
    check('state_only_ideal_loss_is_18_36',abs(ideal_loss-18.36)<1e-12)
    check('state_only_default_controller_matches_ideal_with_numerical_budget',
          abs(state_only['executed_full_minus_compressed_marked_aud']-ideal_loss)<3e-5)
    check('entire_high_stock_action_compatibility_cover_is_not_forward_closed',
          state_only['action_preserving_cover']['image_is_disjoint_from_cover'])

    # Direct plan coupling under the unchanged inventory bounds and efficiencies.
    # No interpolation or Lipschitz assumption on the selected policy is used.
    rng=np.random.default_rng(20261002)
    max_energy_identity_error=0.;max_bound_excess=-np.inf
    for i in range(10000):
        s,sp=rng.uniform(.2,1.8,2)
        intended=rng.uniform(-.5/.91,.91*.5,8)
        plan,states=clip_plan(intended,s)
        coupled,other_states=clip_plan(plan,sp)
        prices=rng.uniform(-100.,500.,8);lam=float(rng.uniform(-100.,500.))
        L=max(np.max(np.abs(prices+5.)/.91),np.max(.91*np.abs(prices-5.)),abs(lam))
        energy_error=abs(np.abs(plan-coupled).sum()+abs(states[-1]-other_states[-1])-abs(s-sp))
        obj=sum(reward(v,p) for v,p in zip(plan,prices))+lam*states[-1]
        objp=sum(reward(v,p) for v,p in zip(coupled,prices))+lam*other_states[-1]
        excess=abs(obj-objp)-L*abs(s-sp)
        max_energy_identity_error=max(max_energy_identity_error,energy_error)
        max_bound_excess=max(max_bound_excess,excess)
        assert energy_error<1e-10 and excess<1e-8
    check("ten_thousand_clipped_plan_couplings",True,
          {"max_energy_budget_identity_error_mwh":max_energy_identity_error,
           "max_reward_bound_excess_aud":max_bound_excess})

    # Verify the optimal WINDOW-value bound with the actual solver.  This does
    # not assert Lipschitz continuity of a recursively updated policy's value.
    model=MPC();max_excess=-np.inf;pairs=[]
    for i in range(100):
        p=rng.uniform(-100.,500.,12);s,sp=rng.uniform(.2,1.8,2)
        a=solve_summary(model,p,s);b=solve_summary(model,p,sp)
        lam=float(p[8:].mean())
        L=max(np.max(np.abs(p[:8]+5.)/.91),np.max(.91*np.abs(p[:8]-5.)),abs(lam))
        # Use the incumbent value, with its certified primary tolerance allowance.
        allowance=a["policy_optimum_loss_upper_aud"]+b["policy_optimum_loss_upper_aud"]
        excess=abs(a["value"]-b["value"])-L*abs(s-sp)-allowance
        max_excess=max(max_excess,excess)
        assert excess < 1e-5
        pairs.append({"initial_soc_1":float(s),"initial_soc_2":float(sp),"L_aud_per_mwh":float(L),
            "window_value_1":a["value"],"window_value_2":b["value"],
            "bound_aud":float(L*abs(s-sp)),"certified_numerical_allowance_aud":allowance})
    check("one_hundred_actual_controller_window_value_pairs",True,{"max_bound_excess_aud":max_excess})

    # Exact finite-state reference-performance-difference identity. Prices are
    # frozen known inputs; this is an ex-post verification, not a runtime oracle.
    prices=[10.,20.,100.];terminal=lambda s:0.
    states=[0.,.25,.5,.75,1.]
    actions=[-.25,0.,.25]  # Positive is discharge here; eta=1 synthetic control.
    feasible=lambda s:[a for a in actions if -1e-12<=s-a<=1.+1e-12]
    mu=lambda t,s:0. if t<2 else max(feasible(s))
    pi=lambda t,s:0. if t<1 else min(feasible(s))
    from functools import lru_cache
    @lru_cache(None)
    def V(t,s):
        if t==3:return terminal(s)
        a=mu(t,s)
        return prices[t]*a+V(t+1,round(s-a,10))
    def rollout(policy):
        s=.5;total=0.;rows=[]
        for t,p in enumerate(prices):
            a=policy(t,s);ns=round(s-a,10);total+=p*a
            advantage=V(t,s)-p*a-V(t+1,ns)
            rows.append({"t":t,"soc":s,"action_discharge_mwh":a,"next_soc":ns,"advantage_aud":advantage})
            s=ns
        return total+terminal(s),rows
    jm,rm=rollout(mu);jp,rp=rollout(pi)
    check("performance_difference_identity_on_candidate_states",abs(jm-jp-sum(r["advantage_aud"] for r in rp))<1e-10)
    psi_slopes=[3.,-10.,12.,0.]
    psi=lambda t,s:psi_slopes[t]*s
    residual=lambda t,s:psi(t,s)-prices[t]*mu(t,s)-psi(t+1,round(s-mu(t,s),10))
    dt=[]
    for row in rp:
        t=row['t'];s=row['soc'];a=row['action_discharge_mwh'];am=mu(t,s)
        dt.append(prices[t]*am+psi(t+1,round(s-am,10))-prices[t]*a-psi(t+1,row['next_soc']))
    candidate_residual=sum(residual(r['t'],r['soc']) for r in rp)
    reference_residual=sum(residual(r['t'],r['soc']) for r in rm)
    residual_rhs=sum(dt)+candidate_residual-reference_residual
    check('update_residual_difference_identity',abs(jm-jp-residual_rhs)<1e-10,
          {'actual_loss':jm-jp,'same_state_surrogate_advantages':dt,
           'candidate_residual_sum':candidate_residual,'reference_residual_sum':reference_residual})
    actual_coupling_bound=sum(abs(prices[t])*abs(rm[t]['action_discharge_mwh']-rp[t]['action_discharge_mwh']) for t in range(3))
    check('observable_ex_post_action_coupling_bound',abs(jm-jp)<=actual_coupling_bound+1e-10,
          {'actual_loss':jm-jp,'bound_aud':actual_coupling_bound,'scope':'eta=1 finite-state synthetic accounting control'})

    # Exact nonexpansive transition alone does not give action stability: f(s)=s
    # gives zero action, while a projection-type transition can have action slope1.
    # A concrete discontinuous reference policy invalidates naive V^mu Lipschitz.
    policy_value=lambda s:100.*s if s>=.5 else 0.
    left,right=.5-1e-9,.5+1e-9
    discontinuity={"left_soc":left,"right_soc":right,"value_left":policy_value(left),
                   "value_right":policy_value(right),"finite_difference_slope":(policy_value(right)-policy_value(left))/(right-left),
                   "scope":"arbitrary threshold reference policy; not asserted to equal Cstar"}
    check("window_value_lipschitz_does_not_imply_policy_value_lipschitz",discontinuity["finite_difference_slope"]>1e9)

    # A known fixed per-version weight supports exact additive statistics.  This
    # control mirrors target-aligned inverse nominal lead weighting, not overwrites.
    n=1000;nominal_leads=rng.uniform(30.,1440.,n);forecast=rng.normal(70.,100.,n)
    w=1./nominal_leads;A=B=0.
    for wi,xi in zip(w,forecast):A+=float(wi*xi);B+=float(wi)
    batch=float(np.dot(w,forecast)/w.sum());stream=A/B
    check("fixed_version_inverse_lead_exact_streaming",abs(batch-stream)<1e-10,
          {"vintages":n,"batch":batch,"stream":stream,"numerical_difference":abs(batch-stream),
           "persistent_floats_per_active_target":2,"excludes_version_id_dedup_buffer":True})
    old=float(np.dot(w[:500],forecast[:500])/w[:500].sum())
    missed=float(np.dot(w[500:],forecast[500:])/w[500:].sum())
    q=float(w[500:].sum()/w.sum())
    check('clock_omitted_weight_mean_shift_identity',abs(batch-old-q*(missed-old))<1e-10,
          {'omitted_weight_fraction':q,'all_minus_old':batch-old,'formula':q*(missed-old)})

    sources=[Path(__file__),ROOT/'scripts/revision_v5_control_solver.py',
        ROOT/'scripts/revision_v5_control_safe_policy.py',ROOT/'scripts/revision_v5_control_scaled_policy.py',
        ROOT/'scripts/revision_v5_control_enumerated_policy.py']
    result={"status":"PASS_EXECUTED_SYNTHETIC_CONTROLS","seed":20261002,
        "empirical_market_result":False,"trained_compressor":False,
        "runtime_profit_certificate":False,"controller_definition":MPC().definition(),
        "two_step_counterexample":branches,
        "state_only_sufficiency_counterexample":state_only,
        "full_minus_compressed_marked_aud":full['marked_value_aud']-comp['marked_value_aud'],
        "window_lipschitz_solver_checks":pairs,
        "performance_difference":{"reference_value":jm,"candidate_value":jp,"candidate_state_advantages":rp},
        "policy_value_discontinuity_control":discontinuity,
        "checks":checks,"check_count":len(checks),"elapsed_seconds":time.perf_counter()-start,
        "sources":[{"file":str(p.relative_to(ROOT)).replace('\\','/'),"sha256":digest(p)} for p in sources]}
    (OUT/'executed_toy_controls.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({"status":result['status'],"checks":len(checks),
        "two_step_loss_aud":result['full_minus_compressed_marked_aud'],
        "elapsed_seconds":result['elapsed_seconds']}))


if __name__=='__main__':main()
