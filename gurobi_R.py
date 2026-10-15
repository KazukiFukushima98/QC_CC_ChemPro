"""Gurobi baseline for the R-reactor selection problem (standard bounds 0<=x<=U,
p_i = alpha_i * x_i * y_i).  Also enumerates every reactor configuration to
report the objective margin between the optimum and the runner-up structure.
Results are saved to agg/gurobi_R2to6.csv (R = 2..6, reactor parameters from sa_sweep_full.CASES)."""
import itertools, time, sys
import gurobipy as gp
from gurobipy import GRB

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_sweep_full import CASES      # same reactor parameters as SA / GA / D-Wave (reactors 1-6)
import pandas as pd
U, d = 5.0, 3.0

def solve_full(R, P, d=d):
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.NonConvex = 2
    x = m.addVars(R, lb=0, ub=U); y = m.addVars(R, vtype=GRB.BINARY)
    m.setObjective(gp.quicksum(P["c"][i]*x[i]*x[i] + P["f"][i]*y[i] for i in range(R)), GRB.MINIMIZE)
    m.addConstr(gp.quicksum(P["alpha"][i]*x[i]*y[i] for i in range(R)) >= d)
    t = time.time(); m.optimize(); dt = time.time()-t
    return m.ObjVal, [x[i].X for i in range(R)], [int(round(y[i].X)) for i in range(R)], dt

def solve_fixed(R, P, ysel, d=d):
    m = gp.Model(); m.Params.OutputFlag = 0
    x = m.addVars(R, lb=0, ub=U)
    m.setObjective(gp.quicksum(P["c"][i]*x[i]*x[i] for i in range(R)) + sum(P["f"][i]*ysel[i] for i in range(R)), GRB.MINIMIZE)
    m.addConstr(gp.quicksum(P["alpha"][i]*ysel[i]*x[i] for i in range(R)) >= d)
    m.optimize()
    return (m.ObjVal, [x[i].X for i in range(R)]) if m.Status == GRB.OPTIMAL else (float("inf"), None)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agg", "gurobi_R2to6.csv")
REPS = 20
rows_out = []
for dd in [3.0] + [float(v) for v in sys.argv[1:]]:
    print("" + chr(10) + f"########## demand d = {dd}")
    for R in sorted(CASES):
        P = CASES[R]
        res = [solve_full(R, P, dd) for _ in range(REPS)]          # repeat to report a stable wall time
        obj, x, y, _ = res[0]
        ts = [r[3] for r in res]
        rows = []
        for ysel in itertools.product([0, 1], repeat=R):
            o, xs = solve_fixed(R, P, ysel, dd)
            if o < float("inf"):
                rows.append((o, ysel, xs))
        rows.sort()
        print(f"R={R}: Gurobi opt = {obj:.6f}  y={y}  x={[round(v, 6) for v in x]}  time median={sorted(ts)[REPS // 2] * 1e3:.2f} ms (min {min(ts) * 1e3:.2f}, max {max(ts) * 1e3:.2f})")
        for o, ysel, xs in rows[:4]:
            print(f"      y={ysel} obj={o:.6f} (+{(o / rows[0][0] - 1) * 100:5.2f}%)  x={[round(v, 4) for v in xs]}")
        rows_out.append(dict(R=R, demand=dd, obj=obj, y="".join(map(str, y)), x=str([round(v, 6) for v in x]),
                             t_median_s=sorted(ts)[REPS // 2], t_min_s=min(ts), t_max_s=max(ts), reps=REPS,
                             runner_up_y="".join(map(str, rows[1][1])), runner_up_obj=rows[1][0],
                             runner_up_gap_pct=(rows[1][0] / rows[0][0] - 1) * 100))
pd.DataFrame(rows_out).to_csv(OUT, index=False)
print("saved", OUT)
