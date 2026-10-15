"""D-Wave Leap CQM hybrid (via Amplify) with exactly the HOBO variables: y (R bits), x_i = w * sum 2^k q_ik (n bits each),
slack s = w * sum 2^k s_k (n bits, 0..5), w = 5/(2^n - 1). Objective sum c x^2 + f y (same as HOBO).
CQM_FORM=ineq (default; the form used in the manuscript): the demand sum alpha x y >= 3 is passed as an inequality CONSTRAINT
(no slack, no penalty; R + R n variables). CQM_FORM=eq (tested once, not used in the manuscript): the HOBO equation
sum alpha x y - s = 3 with the HOBO slack bits is passed as an equality constraint (same variables as the HOBO).
usage: hybrid_cqm_bin.py check R N
       hybrid_cqm_bin.py run R N RUNS [MAX_TOTAL_S]      (env DWAVE_TOKEN_FILE, BUDGET_S as in hybrid_run.py)
Rows are appended to hybrid_CQM_bin_results.csv (column order fixed by the header)."""
import sys, os, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_sweep_full import CASES
from amplify import VariableGenerator, Model, LeapHybridCQMSamplerClient, solve, equal_to, greater_equal
FORM = os.environ.get("CQM_FORM", "ineq")   # eq: sum a x y - s = 3 with HOBO slack bits; ineq: sum a x y >= 3, no slack

ROOT = os.environ.get("QCCC_RESULTS", os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/hybrid_CQM_bin_results.csv"
OPT = 19.9625
TOKEN_FILE = os.environ.get("DWAVE_TOKEN_FILE", "~/.dwave_token")
TOKEN_LABEL = os.path.basename(TOKEN_FILE)
BUDGET_S = float(os.environ.get("BUDGET_S", "0"))


def token():
    return open(os.path.expanduser(TOKEN_FILE)).read().strip()


def used_s():
    tot = 0.0
    for f in (OUT, f"{ROOT}/hybrid_NB_results.csv"):          # same token may be used by BQM runs too
        if os.path.exists(f):
            d = pd.read_csv(f)
            if "token_label" in d.columns:
                m = (d.token_label == TOKEN_LABEL) & d["date"].astype(str).str.startswith(time.strftime("%Y-%m"))   # this month only
                tot += float(d.loc[m, "charge_time_s"].sum())
    return tot


def build(R, n):
    P = CASES[R]
    c, f, a = (np.array(P[k]) for k in ("c", "f", "alpha"))
    w = 5.0 / (2 ** n - 1)
    gen = VariableGenerator()
    y = gen.array("Binary", R)
    q = gen.array("Binary", (R, n))
    sb = gen.array("Binary", n)
    x = [w * sum((2 ** k) * q[i, k] for k in range(n)) for i in range(R)]
    s = w * sum((2 ** k) * sb[k] for k in range(n))
    obj = sum(float(c[i]) * x[i] * x[i] + float(f[i]) * y[i] for i in range(R))
    if FORM == "ineq":
        dem = greater_equal(sum(float(a[i]) * x[i] * y[i] for i in range(R)), 3.0)
    else:
        dem = equal_to(sum(float(a[i]) * x[i] * y[i] for i in range(R)) - s, 3.0)
    return Model(obj, dem), y, q, sb, (c, f, a, w)


if __name__ == "__main__":
    cmd, R, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    model, y, q, sb, (c, f, a, w) = build(R, n)
    cl = LeapHybridCQMSamplerClient()
    cl.token = token()
    tmin = float(cl.properties["minimum_time_limit_s"])
    if cmd == "check":
        im, _ = model.to_intermediate_model(cl.acceptable_degrees)
        types = {}
        for v in im.variables:
            t = str(v).split("type: ")[1].rstrip("}")
            types[t] = types.get(t, 0) + 1
        print(f"R={R} n={n}: variables sent {len(im.variables)} {types}; constraints {len(im.constraints)}; "
              f"objective terms {len(im.objective.as_dict())}; minimum time {tmin} s; token {TOKEN_LABEL}, used so far {used_s():.1f} s")
        for cst in list(im.constraints)[:1]:
            print("  constraint:", str(cst)[:160], "...")
    elif cmd == "run":
        runs = int(sys.argv[4])
        max_total = float(sys.argv[5]) if len(sys.argv) > 5 else 15.0
        if tmin * runs > max_total:
            sys.exit(f"abort: {runs} x {tmin} s exceeds MAX_TOTAL_S = {max_total}")
        for r in range(runs):
            if BUDGET_S > 0 and used_s() + tmin > BUDGET_S:
                sys.exit(f"budget stop: used {used_s():.1f} s + {tmin} s > {BUDGET_S} s for {TOKEN_LABEL}")
            t0 = time.time()
            res = solve(model, cl, filter_solution=False)
            wall = time.time() - t0
            info = getattr(res.client_result, "info", {}) or {}
            best = res.best
            yv = np.array([int(round(best.values[v])) for v in y])
            qv = np.array([[int(round(best.values[q[i, k]])) for k in range(n)] for i in range(R)])
            sv = np.array([int(round(best.values[sb[k]])) if sb[k] in best.values else 0 for k in range(n)]) if FORM != "ineq" else np.zeros(n, int)
            x = w * (qv @ (2 ** np.arange(n)))
            sval = w * float(sv @ (2 ** np.arange(n)))
            objv = float(np.sum(c * x ** 2) + np.sum(f * yv))
            prod = float(np.sum(a * x * yv))
            row = dict(R=R, n=n, run=r, form=("binary_inequality_constraint" if FORM == "ineq" else "binary_equality_constraint"), tmin_s=tmin,
                       run_time_s=info.get("run_time", np.nan) * 1e-6, qpu_access_time_s=info.get("qpu_access_time", np.nan) * 1e-6,
                       charge_time_s=info.get("charge_time", np.nan) * 1e-6, total_time_s=res.total_time.total_seconds(), wall_s=wall,
                       feasible_amplify=bool(best.feasible), obj=objv, prod=prod, slack=sval, residual=(prod - 3.0 if FORM == "ineq" else prod - sval - 3.0),
                       y="".join(map(str, yv)), x=json.dumps([round(float(v), 6) for v in x]),
                       ok1=objv <= OPT * 1.01, met=prod >= 3 - 1e-9, n_solutions=len(res),
                       date=time.strftime("%Y-%m-%d %H:%M:%S"), token_label=TOKEN_LABEL)
            rdf = pd.DataFrame([row])
            if os.path.exists(OUT):
                rdf = rdf.reindex(columns=pd.read_csv(OUT, nrows=0).columns)
            rdf.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"CQM-bin R={R} n={n} run {r}: obj={objv:.4f} (ok1={row['ok1']}) prod={prod:.4f} met={row['met']} y={row['y']} "
                  f"x={row['x']} slack={sval:.4f} residual={row['residual']:.2e} feasible={row['feasible_amplify']} "
                  f"run_time={row['run_time_s']:.2f}s qpu={row['qpu_access_time_s']:.3f}s", flush=True)
