"""D-Wave Leap hybrid (via Amplify) for the reactor problem, formulation without the x<=5 penalty (drop_bounds).
usage:
  hybrid_run.py query R_LIST N_LIST            -> QUBO size and the solver's minimum time_limit (no solver time used)
  hybrid_run.py run R N LAM RUNS [MAX_TOTAL_S] -> RUNS independent calls with the default (= minimum) time_limit
The token is read from ~/.dwave_token and never printed.
Per call we record: solver run_time, qpu_access_time, charge_time, total wall time (incl. connection),
best-sample bits and the decoded objective / demand. Rows are appended to hybrid_NB_results.csv."""
import sys, os, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hobo_builder import ReactorHOBO
from sa_sweep_full import CASES
from amplify import VariableGenerator, Model, AcceptableDegrees, LeapHybridSamplerClient, solve

ROOT = os.environ.get("QCCC_RESULTS", os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/hybrid_NB_results.csv"
OPT = 19.9625
TOKEN_FILE = os.environ.get("DWAVE_TOKEN_FILE", "~/.dwave_token")
TOKEN_LABEL = os.path.basename(TOKEN_FILE)
BUDGET_S = float(os.environ.get("BUDGET_S", "0"))   # >0: stop before exceeding this charged total for TOKEN_LABEL


def token():
    return open(os.path.expanduser(TOKEN_FILE)).read().strip()


def used_s():
    if not os.path.exists(OUT):
        return 0.0
    d = pd.read_csv(OUT)
    lab = d["token_label"] if "token_label" in d.columns else pd.Series(".dwave_token", index=d.index)
    return float(d.loc[lab == TOKEN_LABEL, "charge_time_s"].sum())


def build(R, n, lam):
    P = CASES[R]
    m = ReactorHOBO(R, n, lam, P["c"], P["f"], P["alpha"], drop_bounds=True)
    gen = VariableGenerator()
    poly, q = m.to_amplify(gen)
    model = Model(poly)
    im, _ = model.to_intermediate_model(AcceptableDegrees(objective={"Binary": "Quadratic"}))
    nq = im.num_variables if hasattr(im, "num_variables") else len(im.variables)
    return m, q, model, nq


def min_time_curve():
    # solver properties through Amplify (the token is accepted only by the Amplify client)
    c = LeapHybridSamplerClient()
    c.token = token()
    pr = c.properties
    return c.solver, np.array(pr["minimum_time_limit"], float), pr.get("maximum_number_of_variables")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "query":
        name, pts, vmax = min_time_curve()
        print(f"solver {name}; max variables {vmax}; minimum_time_limit points (vars, s): {pts.tolist()}")
        for R in map(int, sys.argv[2].split(",")):
            for n in map(int, sys.argv[3].split(",")):
                _, _, _, nq = build(R, n, 1e3)
                print(f"R={R} n={n}: QUBO variables {nq}, minimum time_limit {np.interp(nq, pts[:, 0], pts[:, 1]):.2f} s")
    elif cmd == "run":
        R, n, lam, runs = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
        max_total = float(sys.argv[6]) if len(sys.argv) > 6 else 30.0
        m, q, model, nq = build(R, n, lam)
        name, pts, _ = min_time_curve()
        tmin = float(np.interp(nq, pts[:, 0], pts[:, 1]))
        tl = float(os.environ.get("TIME_LIMIT", "0"))   # seconds; 0 = solver minimum
        tuse = max(tmin, tl)
        if tuse * runs > max_total:
            sys.exit(f"abort: {runs} x {tuse:.2f} s = {tuse * runs:.1f} s exceeds MAX_TOTAL_S = {max_total}")
        client = LeapHybridSamplerClient()
        client.token = token()
        if tl > 0:
            from datetime import timedelta
            client.parameters.time_limit = tl
        if os.path.exists(OUT):
            old = pd.read_csv(OUT, dtype={"bits": str, "y": str})
            if "time_limit_s" not in old.columns:
                old.insert(old.columns.get_loc("tmin_s") + 1, "time_limit_s", old["tmin_s"])
                old.to_csv(OUT, index=False)
        if os.path.exists(OUT):
            old = pd.read_csv(OUT, dtype={"bits": str, "y": str})
            if "token_label" not in old.columns:
                old["token_label"] = ".dwave_token"
                old.to_csv(OUT, index=False)
        for k in range(runs):
            if BUDGET_S > 0 and used_s() + tuse > BUDGET_S:
                sys.exit(f"budget stop: used {used_s():.1f} s + {tuse:.1f} s > {BUDGET_S} s for {TOKEN_LABEL}")
            t0 = time.time()
            res = solve(model, client)
            wall = time.time() - t0
            info = getattr(res.client_result, "info", {}) or {}
            bits = np.asarray(q.evaluate(res.best.values), int)
            dec = m.decode(bits)
            row = dict(R=R, n=n, penalty=lam, run=k, qubo_vars=nq, solver=name, tmin_s=tmin, time_limit_s=tuse,
                       run_time_s=info.get("run_time", np.nan) * 1e-6, qpu_access_time_s=info.get("qpu_access_time", np.nan) * 1e-6,
                       charge_time_s=info.get("charge_time", np.nan) * 1e-6, total_time_s=res.total_time.total_seconds(), wall_s=wall,
                       energy=res.best.objective, obj=dec["obj"], prod=dec["prod"], y="".join(map(str, dec["y"])),
                       x=json.dumps([round(float(v), 6) for v in dec["x"]]), ok1=dec["obj"] <= OPT * 1.01,
                       bits="".join(map(str, bits)), date=time.strftime("%Y-%m-%d %H:%M:%S"), token_label=TOKEN_LABEL)
            rdf = pd.DataFrame([row])
            if os.path.exists(OUT):   # always append in the existing column order
                rdf = rdf.reindex(columns=pd.read_csv(OUT, nrows=0).columns)
            rdf.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"R={R} n={n} lam={lam:g} run {k}: obj={dec['obj']:.4f} (ok1={row['ok1']}) prod={dec['prod']:.4f} y={row['y']} "
                  f"run_time={row['run_time_s']:.2f}s qpu={row['qpu_access_time_s']:.3f}s total={row['total_time_s']:.2f}s", flush=True)
