"""Full Parallel-SA (hobotan MIKASAmpler) sweep over R, n, penalty, shots.
Every sample is decoded in float64; per-run success probabilities under
unified criteria are written incrementally to CSV (resumable).
usage: sa_sweep_full.py MODE R_LIST N_LIST LAM_LIST SHOTS_LIST TAG"""
import sys, time, itertools, json, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hobo_builder import ReactorHOBO
from hobotan import sampler

# Reactor parameters (c, f, alpha). Reactors 1-2: original problem; reactors 3-6 were chosen so that the optimum stays
# "reactor 2 only, x2 = 3.75" (19.9625) for all R and the second-best combination is 2.5-4.5 % more expensive.
_PAR = [(1.5, 6.1, 0.75), (1.0, 5.9, 0.80), (1.2, 6.0, 0.70), (1.1, 7.2, 0.80), (0.9, 7.5, 0.75), (0.8, 7.6, 0.70)]
CASES = {R: dict(c=tuple(p[0] for p in _PAR[:R]), f=tuple(p[1] for p in _PAR[:R]), alpha=tuple(p[2] for p in _PAR[:R])) for R in range(2, 7)}
OPT = {R: (19.9625, tuple(1 if i == 1 else 0 for i in range(R))) for R in range(2, 7)}
EPS = [0.005, 0.01, 0.02, 0.05, 0.10]
T_NUM = int(os.environ.get("T_NUM", "50000"))
DROP_BOUNDS = bool(int(os.environ.get("DROP_BOUNDS", "1")))   # 1: no penalty for x <= 5 (as in the manuscript)


def crit(dec, R):
    opt, ystar = OPT[R]
    gap = dec["obj"] / opt - 1
    out = {"feasible": dec["feasible"], "struct": dec["y"] == ystar, "gap": gap}
    out["obj1"] = gap <= 0.01          # success criterion of the manuscript (objective only, tolerance 1 %)
    for e in EPS:
        out[f"fg{e*100:g}"] = dec["feasible"] and gap <= e
        out[f"sg{e*100:g}"] = dec["feasible"] and gap <= e and dec["y"] == ystar
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    R_list = [int(v) for v in sys.argv[2].split(",")]
    n_list = [int(v) for v in sys.argv[3].split(",")]
    lam_list = [float(v) for v in sys.argv[4].split(",")]
    shots_list = [int(v) for v in sys.argv[5].split(",")]
    tag = sys.argv[6]
    out = os.path.join(os.environ.get("QCCC_RESULTS", os.path.dirname(os.path.abspath(__file__))), f"sa_full_{tag}_{mode}")
    device = "cuda:0" if mode == "GPU" else "cpu"
    solver = sampler.MIKASAmpler(mode=mode, verbose=0)
    summ_path, samp_path = out + "_summary.csv", out + "_samples.csv"
    done = set()
    if os.path.exists(summ_path):
        d0 = pd.read_csv(summ_path)
        done = set(zip(d0.R, d0.n, d0.penalty, d0.shots))
    for R, n, lam in itertools.product(R_list, n_list, lam_list):
        P = CASES[R]
        t0 = time.time()
        m = ReactorHOBO(R, n, lam, P["c"], P["f"], P["alpha"], drop_bounds=DROP_BOUNDS)
        hobo, offset = m.to_hobotan(device)
        tq = time.time() - t0
        for shots in shots_list:
            if (R, n, lam, shots) in done:
                continue
            t0 = time.time()
            res = solver.run(hobo, shots=shots, T_num=T_NUM)
            tc = time.time() - t0
            samples, acc = [], {}
            for sol, en, cnt in res:
                bits = [sol[nm] for nm in m.names]
                dec = m.decode(bits)
                c = crit(dec, R)
                samples.append(dict(mode=mode, R=R, n=n, penalty=lam, shots=shots, count=cnt,
                                    hobo_energy=en + float(offset), energy=dec["energy"], obj=dec["obj"],
                                    prod=dec["prod"], y="".join(map(str, dec["y"])),
                                    x=json.dumps([round(float(v), 6) for v in dec["x"]]),
                                    bits="".join(str(int(b)) for b in bits),
                                    **{k: (float(v) if k == "gap" else bool(v)) for k, v in c.items()}))
                for k, v in c.items():
                    if k != "gap":
                        acc[k] = acc.get(k, 0) + cnt * bool(v)
            feas = [s["gap"] for s in samples if s["feasible"]]
            row = dict(mode=mode, formulation=("drop_bounds" if DROP_BOUNDS else "original"), R=R, n=n, nvar=m.nvar, nterms=len(m.poly), penalty=lam, shots=shots, T_num=T_NUM,
                       calctime=tc, qubotime=tq, best_feasible_gap=(min(feas) if feas else np.nan),
                       **{"p_" + k: v / shots for k, v in acc.items()})
            pd.DataFrame([row]).to_csv(summ_path, mode="a", header=not os.path.exists(summ_path), index=False)
            pd.DataFrame(samples).to_csv(samp_path, mode="a", header=not os.path.exists(samp_path), index=False)
            print(f"{mode} R={R} n={n:2d} lam={lam:g} shots={shots:3d} t={tc:6.1f}s "
                  f"p_fg1={row['p_fg1']:.3f} p_sg1={row['p_sg1']:.3f} p_feas={row['p_feasible']:.2f}", flush=True)
            time.sleep(2)
    print("DONE", out)
