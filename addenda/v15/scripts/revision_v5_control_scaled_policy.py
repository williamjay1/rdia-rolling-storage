"""Equivalent-unit cold retry after the immutable Safe policy fails.

SOC is expressed in kWh internally, including its fixed initial-column bounds.
This changes neither physical constraints nor the three objective tolerances.
Every successful Safe solve is delegated verbatim; scaled retries are audited
in the original MWh/AUD units and retain stage witnesses.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_name]="1"
import json,time
import numpy as np
from revision_v5_control_solver import LexMPC as FrozenLexMPC,OUT
from revision_v5_control_safe_policy import SafeLexMPC

class UnitScaledLexMPC(FrozenLexMPC):
    def __init__(self,*args,soc_scale=1000.,primary_row_scale=1.,**kwargs):
        kwargs["relax_if_mode_feasible"]=False
        super().__init__(*args,**kwargs)
        self.soc_scale=float(soc_scale);self.primary_row_scale=float(primary_row_scale)
        self.stage_records=[];h=self.h;q=self.soc_scale
        for j in range(h):
            self.solver.changeCoeff(j,j,-q*self.eta*self.dt)
            self.solver.changeCoeff(j,h+j,q*self.dt/self.eta)
        for j in range(3*h,4*h+1):self.solver.changeColBounds(j,q*self.smin,q*self.smax)
        self.solver.setOptionValue("presolve","off")
        for name in ("primal_feasibility_tolerance","dual_feasibility_tolerance","mip_feasibility_tolerance"):
            self.solver.setOptionValue(name,1e-10)
    def _run(self,cost):
        result=super()._run(cost)
        self.stage_records.append((np.asarray(cost).copy(),result))
        return result
    def solve(self,path,soc,branch=None,cold=True):
        self.stage_records=[];path=np.asarray(path,float);h=self.h;q=self.soc_scale;r=self.primary_row_scale
        primary=np.zeros(self.nv)
        primary[:h]=(path[:h]+self.kappa)*self.dt
        primary[h:2*h]=(-path[:h]+self.kappa)*self.dt
        primary[4*h]=-self.terminal_factor*float(path[h:].mean())/q
        for j in self.primary_indices:self.solver.changeCoeff(self.primary_row,int(j),float(r*primary[j]))
        self.solver.changeColBounds(3*h,q*float(soc),q*float(soc))
        self.solver.changeRowBounds(self.primary_row,-np.inf,np.inf)
        self.solver.changeRowBounds(self.throughput_row,-np.inf,np.inf)
        lo,hi=(-np.inf,np.inf) if branch is None else branch
        self.solver.changeRowBounds(self.branch_row,float(lo),float(hi))
        stage1=self._run(primary)
        if stage1 is None:raise RuntimeError("Scaled primary infeasible")
        x1,p1,b1=stage1
        self.solver.changeRowBounds(self.primary_row,-np.inf,r*(p1+self.primary_tolerance_aud))
        stage2=self._run(self.throughput_cost)
        if stage2 is None:raise RuntimeError("Scaled throughput infeasible")
        x2,t2,bt2=stage2
        self.solver.changeRowBounds(self.throughput_row,-np.inf,t2+self.throughput_tolerance_mwh)
        stage3=self._run(self.first_cost)
        if stage3 is None:raise RuntimeError("Scaled first action infeasible")
        xs,p3,b3=stage3;x=xs.copy();x[3*h:]/=q
        c=float(x[0]);d=float(x[h]);ns=float(soc+self.eta*c*self.dt-d*self.dt/self.eta)
        pa=float(np.asarray(primary,dtype=np.longdouble)@np.asarray(xs,dtype=np.longdouble))
        throughput=float(self.throughput_cost@xs)
        balance=x[3*h+1:]-x[3*h:4*h]-self.eta*x[:h]*self.dt+x[h:2*h]*self.dt/self.eta
        audit={"maximum_balance_error_mwh":float(np.abs(balance).max()),
               "fixed_initial_error_mwh":float(abs(x[3*h]-soc)),
               "primary_bound_excess_aud":float(pa-(p1+self.primary_tolerance_aud)),
               "throughput_bound_excess_mwh":float(throughput-(t2+self.throughput_tolerance_mwh)),
               "maximum_loop_mw":float(np.minimum(x[:h],x[h:2*h]).max()),
               "soc_bound_violation_mwh":float(max(0.,self.smin-x[3*h:].min(),x[3*h:].max()-self.smax)),
               "power_bound_violation_mw":float(max(0.,-x[:2*h].min(),x[:2*h].max()-self.power))}
        if (audit["maximum_balance_error_mwh"]>1e-12 or audit["fixed_initial_error_mwh"]>1e-12
            or audit["primary_bound_excess_aud"]>1e-9 or audit["throughput_bound_excess_mwh"]>1e-10
            or audit["maximum_loop_mw"]>1e-8 or audit["soc_bound_violation_mwh"]>1e-12
            or audit["power_bound_violation_mw"]>1e-10):
            raise RuntimeError(f"Scaled original-unit audit failed: {audit}")
        return {"c":c,"d":d,"u":d-c,"soc":ns,"value":-pa,
          "value_lower_bound":-p1,"value_upper_bound":-b1,
          "primary_optimum_incumbent_aud":-p1,"primary_objective_gap_aud":max(0.,p1-b1),
          "policy_primary_loss_aud":pa-p1,"policy_optimum_loss_upper_aud":max(0.,pa-b1),
          "minimum_throughput_mwh":t2,"throughput_mwh":throughput,
          "throughput_optimum_gap_mwh":max(0.,t2-bt2),"first_action_optimum_gap_mw":max(0.,p3-b3),
          "x":x,"status":"optimal equivalent-unit cold retry","original_unit_audit":audit}

class ScaledSafeLexMPC(SafeLexMPC):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.scaled_fallback_count=0;self.scaled_fallback_records=[]
    def solve(self,path,soc,branch=None,cold=True):
        try:return super().solve(path,soc,branch,cold)
        except RuntimeError as error:
            start=time.time();attempts=[]
            # Fixed algebraic retry order; no result-dependent policy selection.
            for primary_scale in (1.,1000.,.001):
                model=UnitScaledLexMPC(h=self.h,power=self.power,eta=self.eta,kappa=self.kappa,dt=self.dt,
                 smin=self.smin,smax=self.smax,terminal_factor=self.terminal_factor,
                 primary_tolerance_aud=self.primary_tolerance_aud,
                 throughput_tolerance_mwh=self.throughput_tolerance_mwh,primary_row_scale=primary_scale)
                try:result=model.solve(path,soc,branch,cold=True)
                except RuntimeError as retry_error:
                    attempts.append({"primary_row_scale":primary_scale,"error":str(retry_error),"stages":encode_stages(model.stage_records)})
                    continue
                record={"trigger":str(error),"path":np.asarray(path,float).tolist(),"soc":float(soc),"branch":branch,
                    "soc_variable_scale":1000.,"primary_row_scale":primary_scale,"seconds":time.time()-start,
                    "original_unit_audit":result["original_unit_audit"],"failed_scaled_attempts":attempts,
                    "successful_stages":encode_stages(model.stage_records)}
                self.scaled_fallback_count+=1;self.scaled_fallback_records.append(record)
                print(json.dumps({"equivalent_unit_fallback":{k:v for k,v in record.items() if k not in ("successful_stages","failed_scaled_attempts")}}),flush=True)
                return result
            OUT.mkdir(parents=True,exist_ok=True)
            case={"path":np.asarray(path,float).tolist(),"soc":soc,"branch":branch,"attempts":attempts,"definition":self.definition()}
            (OUT/f"unresolved_scaled_case_{time.time_ns()}.json").write_text(json.dumps(case,indent=2),encoding="utf-8")
            raise RuntimeError(f"All unchanged-policy numerical retries failed: {error}")
    def definition(self):
        d=super().definition()
        d["equivalent_unit_failure_fallback"]="delegate immutable Safe first; on failure cold allbinary/presolveoff/feas1e-10 SOC variables in kWh; primary row scales1,1000,0.001 in fixed order; original-unit audits; no changed policy margins"
        return d

def encode_stages(stages):
    rows=[]
    for cost,result in stages:
        row={"cost":cost.tolist()}
        if result is None:row["status"]="infeasible"
        else:
            x,v,b=result;row.update(x=x.tolist(),objective=v,bound=b)
        rows.append(row)
    return rows

LexMPC=ScaledSafeLexMPC
