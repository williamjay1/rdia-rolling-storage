"""Final rare numerical retry: eliminate SOC columns and enumerate modes.

The 2**h fixed charge/discharge patterns exhaust the original feasible set.
Each fixed pattern is a continuous LP with cumulative SOC inequalities whose
right-hand sides contain the unchanged initial state. All three original
objectives and monetary/throughput margins are retained. Previously successful
immutable ScaledSafe solutions are delegated unchanged.
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
import json,time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize._highspy import _core as hc
from revision_v5_control_solver import OUT
from revision_v5_control_scaled_policy import ScaledSafeLexMPC

class EnumeratedReducedLP:
    def __init__(self,parent):
        self.parent=parent;self.h=parent.h;h=self.h
        if h>12:raise RuntimeError("Exhaustive reduced LP retry predeclared only for h<=12")
        self.nv=2*h;self.primary_row=h;self.throughput_row=h+1;self.branch_row=h+2
        self.soc_row_scale=1000.;self.primary_row_scale=1000.
        rows=[];cols=[];data=[]
        for j in range(h):
            for k in range(j+1):
                rows.extend([j,j]);cols.extend([k,h+k]);data.extend([1000*parent.eta*parent.dt,-1000*parent.dt/parent.eta])
        for j in range(2*h):
            rows.extend([h,h+1]);cols.extend([j,j]);data.extend([1.,parent.dt])
        rows.extend([h+2,h+2]);cols.extend([0,h]);data.extend([-1.,1.])
        a=coo_matrix((data,(rows,cols)),shape=(h+3,2*h)).tocsc()
        lp=hc.HighsLp();lp.num_col_=2*h;lp.num_row_=h+3
        lp.col_cost_=np.zeros(2*h);lp.col_lower_=np.zeros(2*h);lp.col_upper_=np.full(2*h,parent.power)
        lp.row_lower_=np.full(h+3,-np.inf);lp.row_upper_=np.full(h+3,np.inf)
        lp.a_matrix_.format_=hc.MatrixFormat.kColwise
        lp.a_matrix_.start_=a.indptr.astype(np.int32);lp.a_matrix_.index_=a.indices.astype(np.int32);lp.a_matrix_.value_=a.data
        self.solver=hc._Highs()
        for name,value in [("output_flag",False),("threads",1),("presolve","off"),
           ("primal_feasibility_tolerance",1e-10),("dual_feasibility_tolerance",1e-10),("random_seed",0)]:self.solver.setOptionValue(name,value)
        self.solver.passModel(lp);self.indices=np.arange(2*h,dtype=np.int32)
        self.first=np.zeros(2*h);self.first[0]=-1.;self.first[h]=1.
        self.throughput=np.full(2*h,parent.dt)
    def pattern(self,mask):
        bits=np.array([(mask>>j)&1 for j in range(self.h)])
        upper=np.r_[bits*self.parent.power,(1-bits)*self.parent.power]
        self.solver.changeColsBounds(self.nv,self.indices,np.zeros(self.nv),upper)
    def run(self,cost):
        self.solver.changeColsCost(self.nv,self.indices,cost)
        self.solver.clearSolver();self.solver.run()
        status=self.solver.getModelStatus();solution=self.solver.getSolution()
        if status==hc.HighsModelStatus.kInfeasible:return None
        if status!=hc.HighsModelStatus.kOptimal or not solution.value_valid:raise RuntimeError(f"Reduced LP status {status}")
        x=np.asarray(solution.col_value,float)
        value=float(np.asarray(cost,dtype=np.longdouble)@np.asarray(x,dtype=np.longdouble))
        return x,value
    def solve(self,path,soc,branch=None):
        p=self.parent;h=self.h;path=np.asarray(path,float);terminal=p.terminal_factor*float(path[h:].mean())
        primary=np.r_[(path[:h]+p.kappa)*p.dt-terminal*p.eta*p.dt,
                      (-path[:h]+p.kappa)*p.dt+terminal*p.dt/p.eta]
        constant=-terminal*soc
        for j in range(h):self.solver.changeRowBounds(j,1000*(p.smin-soc),1000*(p.smax-soc))
        for j,value in enumerate(primary):self.solver.changeCoeff(self.primary_row,j,1000*float(value))
        self.solver.changeRowBounds(self.primary_row,-np.inf,np.inf)
        self.solver.changeRowBounds(self.throughput_row,-np.inf,np.inf)
        lo,hi=(-np.inf,np.inf) if branch is None else branch
        self.solver.changeRowBounds(self.branch_row,float(lo),float(hi))
        start=time.time();primary_solutions={}
        for mask in range(1<<h):
            self.pattern(mask);result=self.run(primary)
            if result is not None:primary_solutions[mask]=result
        if not primary_solutions:raise RuntimeError("No physically feasible enumerated primary pattern")
        first_mask=min(primary_solutions,key=lambda mask:primary_solutions[mask][1]);x1,v1=primary_solutions[first_mask]
        p1=v1+constant
        self.solver.changeRowBounds(self.primary_row,-np.inf,1000*(v1+p.primary_tolerance_aud))
        secondary_solutions={}
        for mask in range(1<<h):
            self.pattern(mask);result=self.run(self.throughput)
            if result is not None:secondary_solutions[mask]=result
        if not secondary_solutions:raise RuntimeError("No enumerated secondary pattern")
        second_mask=min(secondary_solutions,key=lambda mask:secondary_solutions[mask][1]);x2,t2=secondary_solutions[second_mask]
        self.solver.changeRowBounds(self.throughput_row,-np.inf,t2+p.throughput_tolerance_mwh)
        third_solutions={}
        for mask in range(1<<h):
            self.pattern(mask);result=self.run(self.first)
            if result is not None:third_solutions[mask]=result
        if not third_solutions:raise RuntimeError("No enumerated third pattern")
        third_mask=min(third_solutions,key=lambda mask:third_solutions[mask][1]);flow,u=third_solutions[third_mask]
        c=flow[:h];d=flow[h:];states=np.r_[soc,soc+np.cumsum(p.eta*c*p.dt-d*p.dt/p.eta)]
        full=np.r_[flow,[(third_mask>>j)&1 for j in range(h)],states]
        original_cost=np.r_[(path[:h]+p.kappa)*p.dt,(-path[:h]+p.kappa)*p.dt]
        pa=float(np.asarray(original_cost,dtype=np.longdouble)@np.asarray(flow,dtype=np.longdouble)-np.longdouble(terminal)*np.longdouble(states[-1]))
        t=float(self.throughput@flow)
        audit={"maximum_balance_error_mwh":float(np.max(np.abs(np.diff(states)-p.eta*c*p.dt+d*p.dt/p.eta))),
           "fixed_initial_error_mwh":float(abs(states[0]-soc)),"primary_bound_excess_aud":pa-(p1+p.primary_tolerance_aud),
           "throughput_bound_excess_mwh":t-(t2+p.throughput_tolerance_mwh),"maximum_loop_mw":float(np.minimum(c,d).max()),
           "soc_bound_violation_mwh":float(max(0.,p.smin-states.min(),states.max()-p.smax)),
           "power_bound_violation_mw":float(max(0.,-flow.min(),flow.max()-p.power))}
        if (audit["maximum_balance_error_mwh"]>1e-12 or audit["fixed_initial_error_mwh"]>1e-12
            or audit["primary_bound_excess_aud"]>1e-9 or audit["throughput_bound_excess_mwh"]>1e-10
            or audit["maximum_loop_mw"]>1e-10 or audit["soc_bound_violation_mwh"]>1e-12
            or audit["power_bound_violation_mw"]>1e-10):raise RuntimeError(f"Reduced LP original-unit audit failed {audit}")
        record={"enumerated_modes":1<<h,"primary_feasible_modes":len(primary_solutions),
           "secondary_feasible_modes":len(secondary_solutions),"third_feasible_modes":len(third_solutions),
           "stage1_mask":first_mask,"stage2_mask":second_mask,"stage3_mask":third_mask,
           "stage1_flows":x1.tolist(),"stage2_flows":x2.tolist(),"stage3_flows":flow.tolist(),
           "primary_marginal_cost":primary.tolist(),"primary_constant_aud":constant,
           "primary_optimum_negative_aud":p1,"stage2_throughput_mwh":t2,"seconds":time.time()-start,
           "original_unit_audit":audit,
           "bound_status":"all fixed-mode LPs report optimal under feasibility1e-10; zero algorithmic gap is numerical, not exact arithmetic"}
        return {"c":float(c[0]),"d":float(d[0]),"u":float(d[0]-c[0]),"soc":float(states[1]),"value":-pa,
          "value_lower_bound":-p1,"value_upper_bound":-p1,"primary_optimum_incumbent_aud":-p1,
          "primary_objective_gap_aud":0.,"policy_primary_loss_aud":pa-p1,"policy_optimum_loss_upper_aud":max(0.,pa-p1),
          "minimum_throughput_mwh":t2,"throughput_mwh":t,"throughput_optimum_gap_mwh":0.,
          "first_action_optimum_gap_mw":0.,"x":full,"status":"all fixed modes numerically optimal LPs",
          "original_unit_audit":audit,"reduced_enum_certificate":record}

class EnumeratedSafeLexMPC(ScaledSafeLexMPC):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.enumerated_fallback_count=0;self.enumerated_fallback_records=[]
    def solve(self,path,soc,branch=None,cold=True):
        try:return super().solve(path,soc,branch,cold)
        except RuntimeError as error:
            reduced=EnumeratedReducedLP(self)
            try:result=reduced.solve(path,soc,branch)
            except RuntimeError as final_error:
                OUT.mkdir(parents=True,exist_ok=True)
                case={"path":np.asarray(path,float).tolist(),"soc":soc,"branch":branch,"error":str(final_error),"definition":self.definition()}
                (OUT/f"unresolved_enum_case_{time.time_ns()}.json").write_text(json.dumps(case,indent=2),encoding="utf-8")
                raise
            record={"trigger":str(error),"path":np.asarray(path,float).tolist(),"soc":soc,"branch":branch,
                    "certificate":result["reduced_enum_certificate"]}
            self.enumerated_fallback_count+=1;self.enumerated_fallback_records.append(record)
            print(json.dumps({"enumerated_soc_elimination_fallback":{k:v for k,v in record.items() if k!="certificate"},
                             "original_unit_audit":result["original_unit_audit"]}),flush=True)
            return result
    def definition(self):
        d=super().definition()
        d["soc_elimination_failure_fallback"]="delegate frozen ScaledSafe first; if all fail and h<=12 eliminate SOC columns into scaled cumulative inequalities and enumerate all2^h fixed modes independently at each of three stages; same scientific margins and original-unit audit"
        return d

LexMPC=EnumeratedSafeLexMPC
