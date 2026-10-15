"""Same solver (hobotan MIKASAmpler, GPU, T_num=50000) on the HOBO vs on the Amplify Ishikawa QUBO of the same problem.
R = 2, 4; n = 10; lambda = 1e3; shots 5..200 (640 samples), success = objective <= 1.01 x 19.9625."""
import time, types, numpy as np, pandas as pd
from hobo_builder import ReactorHOBO
from sa_sweep_full import CASES
from amplify import VariableGenerator, Model, AcceptableDegrees
from hobotan import sampler

OPT, SHOTS = 19.9625, [5, 10, 20, 30, 50, 75, 100, 150, 200]
solver = sampler.MIKASAmpler(mode="GPU", verbose=0)
rows = []
for R in (2, 4):
    P = CASES[R]
    m = ReactorHOBO(R, 10, 1e3, P["c"], P["f"], P["alpha"], drop_bounds=True)
    gen = VariableGenerator()
    poly, q = m.to_amplify(gen)
    im, _ = Model(poly).to_intermediate_model(AcceptableDegrees(objective={"Binary": "Quadratic"}))
    d = {tuple(sorted(k)): v for k, v in im.objective.as_dict().items() if v != 0}
    nq = len(im.variables)
    Q = types.SimpleNamespace(poly=d, names=[f"v{i}" for i in range(nq)], nvar=nq)
    for form, obj in (("QUBO (Amplify Ishikawa)", Q), ("HOBO", m)):
        hobo, off = ReactorHOBO.to_hobotan(obj, "cuda:0")
        names = obj.names
        for shots in SHOTS:
            t0 = time.time()
            res = solver.run(hobo, shots=shots, T_num=50000)
            dt = time.time() - t0
            for sol, en, cnt in res:
                bits = np.array([sol[nm] for nm in names[:m.nvar]], int)
                dec = m.decode(bits)
                rows.append(dict(R=R, form=form, nvar=obj.nvar, shots=shots, time=dt, count=cnt, obj=dec["obj"], prod=dec["prod"],
                                 ok1=dec["obj"] <= OPT * 1.01, met=dec["prod"] >= 3 - 1e-9, y="".join(map(str, dec["y"]))))
            time.sleep(2)
        g = pd.DataFrame([r for r in rows if r["R"] == R and r["form"] == form])
        w = g["count"]
        N = w.sum()
        print(f"R={R} {form:24s} vars={obj.nvar:5d}  samples={N}  p_ok1={(w * g.ok1).sum() / N:.3f}  "
              f"p_ok1&met={(w * (g.ok1 & g.met)).sum() / N:.3f}  best obj={g.obj.min():.3f}  "
              f"median obj={np.median(np.repeat(g.obj.values, w.values)):.2f}  mean time/run={g.drop_duplicates('shots').time.mean():.1f}s", flush=True)
import os
ROOT = os.environ.get("QCCC_RESULTS", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{ROOT}/agg", exist_ok=True)
pd.DataFrame(rows).to_csv(f"{ROOT}/agg/hobotan_qubo_vs_hobo.csv", index=False)
