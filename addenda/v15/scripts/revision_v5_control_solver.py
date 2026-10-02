"""Cold, explicitly tolerance-ordered rolling storage policy.

Three optimizations at every origin: minimize negative primary reward; minimize
grid throughput within A$1e-5 of that incumbent; minimize first net discharge
within 1e-7 MWh of minimum throughput. This is tolerance ordering, NOT exact
lexicographic optimality. No preceding basis/solution is reused at any stage.
All mode variables are binary, including the secondary optimizations.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize._highspy import _core as hc

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"/"revision_v5"/"control"

class LexMPC:
    def __init__(self,h=8,power=1.,eta=.91,kappa=5.,dt=.5,smin=.2,smax=1.8,
                 terminal_factor=1.,primary_tolerance_aud=1e-5,
                 throughput_tolerance_mwh=1e-7,relax_if_mode_feasible=True):
        self.h=int(h); self.power=float(power); self.eta=float(eta); self.kappa=float(kappa)
        self.dt=float(dt); self.smin=float(smin); self.smax=float(smax)
        self.terminal_factor=float(terminal_factor)
        self.primary_tolerance_aud=float(primary_tolerance_aud)
        self.throughput_tolerance_mwh=float(throughput_tolerance_mwh)
        self.relax_if_mode_feasible=bool(relax_if_mode_feasible)
        h=self.h;nv=4*h+1;c0=0;d0=h;z0=2*h;s0=3*h
        self.nv=nv;self.primary_row=3*h;self.throughput_row=3*h+1;self.branch_row=3*h+2
        rows=[];cols=[];values=[]
        for j in range(h):
            rows += [j]*4;cols += [s0+j+1,s0+j,c0+j,d0+j];values += [1.,-1.,-eta*dt,dt/eta]
            rows += [h+j]*2;cols += [c0+j,z0+j];values += [1.,-power]
            rows += [2*h+j]*2;cols += [d0+j,z0+j];values += [1.,power]
        for j in range(2*h):
            rows += [self.primary_row,self.throughput_row];cols += [j,j];values += [1.,dt]
        rows += [self.primary_row,self.branch_row,self.branch_row]
        cols += [4*h,c0,d0];values += [1.,-1.,1.]
        matrix=coo_matrix((values,(rows,cols)),shape=(3*h+3,nv)).tocsc()
        lp=hc.HighsLp();lp.num_col_=nv;lp.num_row_=3*h+3
        lp.col_cost_=np.zeros(nv)
        lp.col_lower_=np.concatenate([np.zeros(3*h),np.full(h+1,smin)])
        lp.col_upper_=np.concatenate([np.full(2*h,power),np.ones(h),np.full(h+1,smax)])
        lp.row_lower_=np.concatenate([np.zeros(h),np.full(2*h+3,-np.inf)])
        lp.row_upper_=np.concatenate([np.zeros(2*h),np.full(h,power),np.full(3,np.inf)])
        integrality=[hc.HighsVarType.kContinuous]*nv
        for j in range(z0,z0+h):integrality[j]=hc.HighsVarType.kInteger
        lp.integrality_=integrality
        lp.a_matrix_.format_=hc.MatrixFormat.kColwise
        lp.a_matrix_.start_=matrix.indptr.astype(np.int32)
        lp.a_matrix_.index_=matrix.indices.astype(np.int32);lp.a_matrix_.value_=matrix.data
        self.solver=hc._Highs()
        for name,value in [("output_flag",False),("threads",1),("mip_rel_gap",1e-12),
                           ("mip_abs_gap",1e-10),("primal_feasibility_tolerance",1e-9),
                           ("dual_feasibility_tolerance",1e-9),("mip_feasibility_tolerance",1e-9),
                           ("time_limit",1e9),("random_seed",0)]:
            self.solver.setOptionValue(name,value)
        self.solver.passModel(lp)
        self.z_indices=np.arange(2*h,3*h,dtype=np.int32)
        self.integer_modes=np.ones(h,dtype=np.uint8)
        self.continuous_modes=np.zeros(h,dtype=np.uint8)
        self.indices=np.arange(nv,dtype=np.int32)
        self.primary_indices=np.array(list(range(2*h))+[4*h],np.int32)
        self.throughput_cost=np.zeros(nv);self.throughput_cost[:2*h]=dt
        self.first_cost=np.zeros(nv);self.first_cost[0]=-1;self.first_cost[h]=1
    def _run(self,cost):
        self.solver.changeColsCost(self.nv,self.indices,cost)
        if self.relax_if_mode_feasible:
            self.solver.changeColsIntegrality(self.h,self.z_indices,self.continuous_modes)
        self.solver.clearSolver();self.solver.run()
        status=self.solver.getModelStatus();sol=self.solver.getSolution()
        if status==hc.HighsModelStatus.kInfeasible:return None
        if status!=hc.HighsModelStatus.kOptimal or not sol.value_valid:
            raise RuntimeError(f"LexMPC stage {status}")
        info=self.solver.getInfo();x=np.asarray(sol.col_value,float)
        value=float(self.solver.getObjectiveValue())
        integer_run=not self.relax_if_mode_feasible
        if self.relax_if_mode_feasible and np.any((x[:self.h]>1e-8)&(x[self.h:2*self.h]>1e-8)):
            self.solver.changeColsIntegrality(self.h,self.z_indices,self.integer_modes)
            self.solver.clearSolver();self.solver.run()
            status=self.solver.getModelStatus();sol=self.solver.getSolution()
            if status==hc.HighsModelStatus.kInfeasible:return None
            if status!=hc.HighsModelStatus.kOptimal or not sol.value_valid:
                raise RuntimeError(f"LexMPC integer stage {status}")
            info=self.solver.getInfo();x=np.asarray(sol.col_value,float)
            value=float(self.solver.getObjectiveValue());integer_run=True
        bound=float(info.mip_dual_bound) if integer_run else value
        return x,value,bound
    def solve(self,path,soc,branch=None,cold=True):
        path=np.asarray(path,float);h=self.h
        if len(path)<=h or not np.isfinite(path).all():raise ValueError("finite path with terminal steps required")
        if not self.smin-1e-9<=soc<=self.smax+1e-9:raise ValueError(f"SOC {soc} outside bounds")
        primary=np.zeros(self.nv)
        primary[:h]=(path[:h]+self.kappa)*self.dt
        primary[h:2*h]=(-path[:h]+self.kappa)*self.dt
        primary[4*h]=-self.terminal_factor*float(path[h:].mean())
        for j in self.primary_indices:self.solver.changeCoeff(self.primary_row,int(j),float(primary[j]))
        self.solver.changeColBounds(3*h,float(soc),float(soc))
        self.solver.changeRowBounds(self.primary_row,-np.inf,np.inf)
        self.solver.changeRowBounds(self.throughput_row,-np.inf,np.inf)
        lo,hi=(-np.inf,np.inf) if branch is None else branch
        self.solver.changeRowBounds(self.branch_row,float(lo),float(hi))
        stage1=self._run(primary)
        if stage1 is None:return None
        x1,p1,b1=stage1
        self.solver.changeRowBounds(self.primary_row,-np.inf,p1+self.primary_tolerance_aud)
        stage2=self._run(self.throughput_cost)
        if stage2 is None:raise RuntimeError("Secondary policy infeasible")
        x2,t2,bt2=stage2
        self.solver.changeRowBounds(self.throughput_row,-np.inf,t2+self.throughput_tolerance_mwh)
        stage3=self._run(self.first_cost)
        if stage3 is None:raise RuntimeError("First-action policy infeasible")
        x,p3,b3=stage3
        c=float(np.clip(x[0],0,self.power));d=float(np.clip(x[h],0,self.power))
        ns=float(soc+self.eta*c*self.dt-d*self.dt/self.eta)
        if self.smin-1e-8<=ns<self.smin:ns=self.smin
        if self.smax<ns<=self.smax+1e-8:ns=self.smax
        primary_actual=float(primary@x);throughput=float(self.throughput_cost@x)
        loss=float(primary_actual-p1)
        if loss>self.primary_tolerance_aud+2e-6:raise RuntimeError(f"Primary loss exceeds policy tolerance: {loss}")
        if throughput>t2+self.throughput_tolerance_mwh+2e-8:raise RuntimeError("Throughput bound exceeded")
        if np.any((x[:h]>1e-6)&(x[h:2*h]>1e-6)):raise RuntimeError("Simultaneous modes")
        return {"c":c,"d":d,"u":d-c,"soc":ns,"value":-primary_actual,
                "value_lower_bound":-p1,"value_upper_bound":-b1,
                "primary_optimum_incumbent_aud":-p1,"primary_objective_gap_aud":max(0.,p1-b1),
                "policy_primary_loss_aud":loss,"policy_optimum_loss_upper_aud":max(0.,primary_actual-b1),
                "minimum_throughput_mwh":t2,"throughput_mwh":throughput,
                "throughput_optimum_gap_mwh":max(0.,t2-bt2),
                "first_action_optimum_gap_mw":max(0.,p3-b3),"x":x,"status":str(hc.HighsModelStatus.kOptimal)}
    def definition(self):
        return {"h":self.h,"power_mw":self.power,"eta_c":self.eta,"eta_d":self.eta,
                "dt_hours":self.dt,"kappa_aud_per_grid_mwh":self.kappa,"smin_mwh":self.smin,
                "smax_mwh":self.smax,"terminal_factor":self.terminal_factor,
                "primary_tolerance_aud":self.primary_tolerance_aud,
                "throughput_tolerance_mwh":self.throughput_tolerance_mwh,
                "third_objective":"minimize first-period net discharge d0-c0",
                "numerical_state":"clearSolver before every one of three stages",
                "physical_modes":("each stage: LP accepted only when its optimum obeys modes, otherwise cold binary solve" if self.relax_if_mode_feasible else "binary charge/discharge variables in all three stages"),
                "claim":"tolerance-ordered policy, not exact lexicographic optimum"}

def benchmark(n=1000):
    OUT.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(20261001)
    paths=rng.normal(70,120,(n,12));soc=rng.uniform(.2,1.8,n)
    model=LexMPC();start=time.time();solutions=[model.solve(p,s) for p,s in zip(paths,soc)]
    elapsed=time.time()-start
    binary=LexMPC(relax_if_mode_feasible=False);binary_diffs=[]
    for i in range(min(n,1000)):
        a=binary.solve(paths[i],soc[i]);b=solutions[i]
        binary_diffs.append({key:abs(a[key]-b[key]) for key in ["u","soc","value","primary_optimum_incumbent_aud","minimum_throughput_mwh"]})
    chosen=rng.choice(n,min(n,100),replace=False);order=np.r_[chosen,chosen[::-1]]
    repeat=LexMPC();diff=[]
    for i in order:
        a=repeat.solve(paths[i],soc[i]);b=solutions[i]
        diff.append(max(abs(a["u"]-b["u"]),abs(a["soc"]-b["soc"]),abs(a["value"]-b["value"])))
    report={"status":"passed","n":n,"seconds":elapsed,"solves_per_second":n/elapsed,
            "estimated_full_46741_policy_seconds":elapsed/n*46741,
            "max_reordered_output_difference":max(diff),
            "all_binary_comparison_cases":len(binary_diffs),
            "all_binary_max_abs_difference":{key:max(d[key] for d in binary_diffs) for key in binary_diffs[0]},
            "max_primary_loss_aud":max(s["policy_primary_loss_aud"] for s in solutions),
            "max_optimum_loss_upper_aud":max(s["policy_optimum_loss_upper_aud"] for s in solutions),
            "max_first_action_stage_gap_mw":max(s["first_action_optimum_gap_mw"] for s in solutions),
            "policy":model.definition()}
    (OUT/"cold_lex_solver_benchmark.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)

def actual_extreme_audit():
    import pandas as pd
    filename=ROOT/"results"/"revision_v4"/"control"/"NSW1_knn_mean_paths_strict.parquet"
    names=[f"raw_{j:02d}" for j in range(12)]
    x=pd.read_parquet(filename,columns=names).to_numpy(float)
    idx=np.unique(np.r_[np.argmax(x.max(axis=1)),np.argmin(x.min(axis=1)),np.argmax(np.ptp(x,axis=1)),np.argsort(np.ptp(x,axis=1))[:10],np.linspace(0,len(x)-1,20,dtype=int)])
    paths=np.r_[x[idx],np.full((1,12),-1000),np.full((1,12),-53.),np.full((1,12),-5.),np.full((1,12),0.),np.full((1,12),50.),np.full((1,12),17000.)]
    cases=[(p,s) for p in paths for s in [.2,.2001,1.,1.7999,1.8]]
    fast=LexMPC();binary=LexMPC(relax_if_mode_feasible=False);rows=[];cache=[];start=time.time()
    for p,s in cases:
        a=fast.solve(p,s);b=binary.solve(p,s);cache.append(a)
        rows.append({"u_abs_diff":abs(a["u"]-b["u"]),"soc_abs_diff":abs(a["soc"]-b["soc"]),
                     "reward_abs_diff":abs(a["value"]-b["value"]),
                     "primary_abs_diff":abs(a["primary_optimum_incumbent_aud"]-b["primary_optimum_incumbent_aud"]),
                     "loss_upper":a["policy_optimum_loss_upper_aud"],
                     "maximum_simultaneous_loop_mw":float(np.max(np.minimum(a["x"][:8],a["x"][8:16])))})
    order=np.arange(len(cases))[::-1];reorder=0.
    for i in order:
        a=fast.solve(*cases[i]);b=cache[i]
        reorder=max(reorder,abs(a["u"]-b["u"]),abs(a["soc"]-b["soc"]),abs(a["value"]-b["value"]))
    report={"status":"passed","cases":len(cases),"seconds":time.time()-start,
            "actual_price_max":float(x.max()),"actual_price_min":float(x.min()),
            "equal_flat_prices":[-1000,-53,-5,0,50,17000],"soc_cases":[.2,.2001,1.,1.7999,1.8],
            "maxima":{k:max(r[k] for r in rows) for k in rows[0]},"reordered_max_difference":reorder}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"actual_extremes_policy_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--benchmark",type=int,default=1000);ap.add_argument("--actual-extremes",action="store_true")
    args=ap.parse_args()
    if args.actual_extremes:actual_extreme_audit()
    else:benchmark(args.benchmark)
