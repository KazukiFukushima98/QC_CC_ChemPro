"""Numbers of variables and terms of the HOBO, of the QUBO obtained with Amplify (Ishikawa-KZFD), and of the CQM model,
for R = 2..6 and n = 4..30 (SI Table S1)."""
import pandas as pd
from hobo_builder import ReactorHOBO
from sa_sweep_full import CASES
from amplify import VariableGenerator, Model, AcceptableDegrees
ad = AcceptableDegrees(objective={"Binary": "Quadratic"})
rows = []
for R in range(2, 7):
    P = CASES[R]
    for n in range(4, 31):
        m = ReactorHOBO(R, n, 1e3, P["c"], P["f"], P["alpha"], drop_bounds=True)
        gen = VariableGenerator(); poly, q = m.to_amplify(gen)
        im, _ = Model(poly).to_intermediate_model(ad)
        rows.append(dict(R=R, n=n, n_hobo=m.nvar, n_terms_hobo=len(m.poly), n_qubo=len(im.variables), n_terms_qubo=len(im.objective.as_dict()), n_cqm=R + R * n))
        print(rows[-1], flush=True)
import os
ROOT = os.environ.get("QCCC_RESULTS", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{ROOT}/agg", exist_ok=True)
pd.DataFrame(rows).to_csv(f"{ROOT}/agg/vars_nb_full.csv", index=False)
