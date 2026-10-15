"""QAOA circuit size for the HOBO of the reactor problem (Section 3.3 of the manuscript).

The HOBO is expanded in Pauli-Z operators (q = (1 - Z)/2). Each m-body term is implemented with the CNOT ladder of
Eq. 8 (variables in index order), so that the first and last qubit of the term take part in 2 CNOTs and every other
qubit in 4. Since a qubit can take part in only one gate at a time, the largest number of CNOTs acting on one qubit
gives the number of CNOT layers of one QAOA layer (L = 1); terms with |coefficient| < 1e-12 are excluded.
Writes agg/qaoa_cnot.csv (R = 2..6, n = 10, 20, 30).
"""
import itertools, math
import numpy as np
from hobo_builder import ReactorHOBO

PAR = [(1.5, 6.1, 0.75), (1.0, 5.9, 0.80), (1.2, 6.0, 0.70), (1.1, 7.2, 0.80), (0.9, 7.5, 0.75), (0.8, 7.6, 0.70)]


def case(R):
    return dict(c=tuple(p[0] for p in PAR[:R]), f=tuple(p[1] for p in PAR[:R]), alpha=tuple(p[2] for p in PAR[:R]))


def process_polynomial(poly, n_vars):
    """Ising expansion q = (1 - Z)/2 of the HOBO; returns
    terms_list[m] = {tuple(indices): coefficient} for m-body terms (terms_list[0] = constant)."""
    ising = {}
    for mono, c in poly.items():
        k = len(mono); sc = c / 2**k
        for r in range(k + 1):
            for S in itertools.combinations(mono, r):
                key = tuple(sorted(S)); ising[key] = ising.get(key, 0.0) + sc * (-1)**r
    terms_list = [0.0] + [{} for _ in range(n_vars)]
    for key, v in ising.items():
        if abs(v) < 1e-12: continue
        if len(key) == 0: terms_list[0] += v
        else: terms_list[len(key)][key] = v
    return terms_list


def count_cnots_from_terms(terms_list, n_layers=1):
    """Total CNOT count: 2(m-1) CNOTs per m-body term."""
    per_layer = sum(2 * (len(idx) - 1) for m in range(2, len(terms_list)) for idx in terms_list[m])
    return per_layer * n_layers


def count_cnots_per_variable(terms_list, n_vars):
    """CNOTs acting on each variable in one cost layer (Eq. 8 ladder, index order): ends 2, others 4."""
    use = np.zeros(n_vars, dtype=np.int64)
    for m in range(2, len(terms_list)):
        for idx in terms_list[m]:
            for pos, q in enumerate(idx):
                use[q] += 2 if pos in (0, len(idx) - 1) else 4
    return use


def analyse(R, n):
    m = ReactorHOBO(R, n, 1e3, **case(R), drop_bounds=True)
    n_qubits = m.nvar
    terms_list = process_polynomial(m.poly, n_qubits)
    use = count_cnots_per_variable(terms_list, n_qubits)
    b = int(np.argmax(use))
    return dict(R=R, n=n, qubits=n_qubits,
                cnot_total_per_layer=count_cnots_from_terms(terms_list, 1),
                busiest_variable=m.names[b], cnot_busiest_per_layer=int(use[b]))


if __name__ == "__main__":
    import pandas as pd
    rows = [analyse(R, n) for R in range(2, 7) for n in (10, 20, 30)]
    import os
    ROOT = os.environ.get("QCCC_RESULTS", os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(f"{ROOT}/agg", exist_ok=True)
    D = pd.DataFrame(rows); D.to_csv(f"{ROOT}/agg/qaoa_cnot.csv", index=False); print(D.to_string(index=False))
