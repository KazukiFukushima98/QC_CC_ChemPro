"""GA (DEAP) for the R-reactor problem on the original MINLP (continuous x, binary y).
Fixed generation budget (no early stopping); success is evaluated afterwards
under the unified criteria.   usage: ga_R.py R_LIST LAM_LIST TRIALS"""
import sys, time, random, json
import numpy as np, pandas as pd
from deap import base, creator, tools

import os
# Same reactor parameters as sa_sweep_full.py (optimum = reactor 2 only, 19.9625, for all R).
_PAR = [(1.5, 6.1, 0.75), (1.0, 5.9, 0.80), (1.2, 6.0, 0.70), (1.1, 7.2, 0.80), (0.9, 7.5, 0.75), (0.8, 7.6, 0.70)]
CASES = {R: dict(c=tuple(p[0] for p in _PAR[:R]), f=tuple(p[1] for p in _PAR[:R]), alpha=tuple(p[2] for p in _PAR[:R])) for R in range(2, 7)}
OPT = {R: (19.9625, tuple(1 if i == 1 else 0 for i in range(R))) for R in range(2, 7)}
U, d = 5.0, 3.0
EPS = [0.005, 0.01, 0.02, 0.05, 0.10]
GENS, POP, CXPB, MUTPB = 300, 300, 0.7, 0.2

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def make_toolbox(R, P, lam):
    c, f, a = (np.array(P[k]) for k in ("c", "f", "alpha"))
    tb = base.Toolbox()
    tb.register("individual", lambda: creator.Individual(
        [random.uniform(0, U) for _ in range(R)] + [random.randint(0, 1) for _ in range(R)]))
    tb.register("population", tools.initRepeat, list, tb.individual)

    def evaluate(ind):
        x = np.clip(np.array(ind[:R], float), 0, U)
        y = np.array(ind[R:], float)
        obj = float(c @ x ** 2 + f @ y)
        viol = max(0.0, d - float((a * y) @ x))
        return obj + lam * viol ** 2,

    def mate(i1, i2):
        for k in range(R):
            if random.random() < 0.5:
                g = 2.0 * random.random() - 0.5
                i1[k], i2[k] = (1 - g) * i1[k] + g * i2[k], g * i1[k] + (1 - g) * i2[k]
        for k in range(R, 2 * R):
            if random.random() < 0.5:
                i1[k], i2[k] = i2[k], i1[k]
        return i1, i2

    def mutate(ind, indpb=0.2, sigma=2.0):
        for k in range(R):
            if random.random() < indpb:
                ind[k] = min(U, max(0.0, ind[k] + random.gauss(0, sigma)))
        for k in range(R, 2 * R):
            if random.random() < indpb:
                ind[k] = 1 - ind[k]
        return ind,

    tb.register("evaluate", evaluate)
    tb.register("mate", mate)
    tb.register("mutate", mutate)
    tb.register("select", tools.selTournament, tournsize=3)
    return tb, (c, f, a)


def run_ga(tb):
    pop = tb.population(n=POP)
    hof = tools.HallOfFame(1)
    for ind, fit in zip(pop, map(tb.evaluate, pop)):
        ind.fitness.values = fit
    hof.update(pop)
    for gen in range(1, GENS + 1):
        off = list(map(tb.clone, tb.select(pop, len(pop))))
        for c1, c2 in zip(off[::2], off[1::2]):
            if random.random() < CXPB:
                tb.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values
        for mu in off:
            if random.random() < MUTPB:
                tb.mutate(mu)
                del mu.fitness.values
        inv = [i for i in off if not i.fitness.valid]
        for ind, fit in zip(inv, map(tb.evaluate, inv)):
            ind.fitness.values = fit
        pop[:] = off
        hof.update(pop)
    return hof[0]


if __name__ == "__main__":
    R_list = [int(v) for v in sys.argv[1].split(",")]
    lam_list = [float(v) for v in sys.argv[2].split(",")]
    trials = int(sys.argv[3])
    out = os.path.join(os.environ.get("QCCC_RESULTS", os.path.dirname(os.path.abspath(__file__))), f"ga_R_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    rows = []
    for R in R_list:
        for lam in lam_list:
            tb, (c, f, a) = make_toolbox(R, CASES[R], lam)
            opt, ystar = OPT[R]
            for t in range(trials):
                t0 = time.time()
                best = run_ga(tb)
                dt = time.time() - t0
                x = np.clip(np.array(best[:R], float), 0, U)
                y = tuple(int(v) for v in best[R:])
                obj = float(c @ x ** 2 + f @ np.array(y))
                prod = float((a * np.array(y)) @ x)
                feas = prod >= d - 1e-9
                gap = obj / opt - 1
                row = dict(R=R, penalty=lam, trial=t, time=dt, obj=obj, gap=gap, feasible=feas,
                           y="".join(map(str, y)), struct=(y == ystar),
                           x=json.dumps([round(float(v), 6) for v in x]))
                for e in EPS:
                    row[f"fg{e*100:g}"] = feas and gap <= e
                    row[f"sg{e*100:g}"] = feas and gap <= e and y == ystar
                rows.append(row)
                pd.DataFrame(rows).to_csv(out, index=False)
            sub = pd.DataFrame([r for r in rows if r["R"] == R and r["penalty"] == lam])
            print(f"GA R={R} lam={lam:g}: mean t={sub.time.mean():.2f}s  p_feas={sub.feasible.mean():.2f} "
                  f"p_struct={sub.struct.mean():.2f} p_fg1={sub.fg1.mean():.2f} p_fg5={sub.fg5.mean():.2f} "
                  f"p_sg1={sub.sg1.mean():.2f}", flush=True)
    print("DONE", out)
