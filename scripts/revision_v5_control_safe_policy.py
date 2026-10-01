"""Logged cold numerical fallback for frozen tolerance-ordered MPC.

The policy objectives and bounds stay fixed. A failed stage is retried from a
new all-binary model with presolve disabled. Nothing is skipped or silently
rounded. This wrapper leaves the audited base solver file unchanged.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_name]="1"
import json,time
import numpy as np
from revision_v5_control_solver import LexMPC as FrozenLexMPC,OUT

class SafeLexMPC(FrozenLexMPC):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.fallback_count=0;self.fallback_records=[]
        self.stage_records=[]
    def _run(self,cost):
        result=super()._run(cost)
        self.stage_records.append((np.asarray(cost,float).copy(),result))
        return result
    def solve(self,path,soc,branch=None,cold=True):
        self.stage_records=[]
        try:return super().solve(path,soc,branch,cold)
        except RuntimeError as error:
            start=time.time()
            lp=self.solver.getLp();stages=[]
            for index,(cost,result) in enumerate(self.stage_records):
                upper=list(lp.row_upper_)
                if index==0:upper[self.primary_row]=np.inf
                if index<2:upper[self.throughput_row]=np.inf
                stage={"cost":cost.tolist(),"row_lower":list(lp.row_lower_),"row_upper":upper}
                if result is None:stage["status"]="infeasible"
                else:
                    x,value,bound=result;stage.update(x=x.tolist(),objective=value,bound=bound)
                stages.append(stage)
            initial_record={"trigger":str(error),"path":np.asarray(path,float).tolist(),"soc":float(soc),"branch":branch,"stages":stages}
            print(json.dumps({"numerical_fallback_trigger":{k:v for k,v in initial_record.items() if k!="stages"}}),flush=True)
            retry=FrozenLexMPC(h=self.h,power=self.power,eta=self.eta,kappa=self.kappa,dt=self.dt,
                  smin=self.smin,smax=self.smax,terminal_factor=self.terminal_factor,
                  primary_tolerance_aud=self.primary_tolerance_aud,
                  throughput_tolerance_mwh=self.throughput_tolerance_mwh,relax_if_mode_feasible=False)
            retry.solver.setOptionValue("presolve","off")
            # Feasible stage-2 incumbents can sit at the 1e-9 state-balance
            # limit; tighten feasibility for all fallback stages rather than
            # enlarge either scientific objective tolerance.
            retry.solver.setOptionValue("primal_feasibility_tolerance",1e-10)
            retry.solver.setOptionValue("dual_feasibility_tolerance",1e-10)
            retry.solver.setOptionValue("mip_feasibility_tolerance",1e-10)
            try:sol=retry.solve(path,soc,branch,cold=True)
            except RuntimeError:
                OUT.mkdir(parents=True,exist_ok=True)
                stamp=f"{time.time_ns()}"
                (OUT/f"unresolved_numerical_case_{stamp}.json").write_text(json.dumps(initial_record,indent=2),encoding="utf-8")
                raise
            if sol is None:raise RuntimeError(f"Cold fallback also infeasible, first error: {error}")
            record={"trigger":str(error),"path":np.asarray(path,float).tolist(),"soc":float(soc),
                    "branch":branch,"fallback":"new all-binary model; presolve off; unchanged three objectives and tolerances",
                    "fallback_primal_dual_mip_feasibility_tolerance":1e-10,
                    "seconds":time.time()-start,"primary_loss_upper_aud":sol["policy_optimum_loss_upper_aud"],"original_failed_stages":stages}
            self.fallback_count+=1;self.fallback_records.append(record)
            print(json.dumps({"numerical_fallback":{k:v for k,v in record.items() if k!="original_failed_stages"}}),flush=True)
            return sol
    def definition(self):
        d=super().definition()
        d["numerical_failure_fallback"]="new cold all-binary model; presolve off; primal/dual/MIP feasibility1e-10; same objective bounds; every occurrence logged"
        return d

LexMPC=SafeLexMPC
