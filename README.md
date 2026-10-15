# QC_CC_ChemPro

Code for the manuscript

> K. Fukushima, K. Yagihara, H. Ohno, Y. Fukushima, "Toward quantum computing-aided process design: Quantum-classical hybrid approaches in conceptual reactor optimization", AIChE Journal (under review).

The scripts build the Higher-Order Binary Optimization (HOBO) form of the reactor selection problem (R = 2–6 candidate reactors, n bits per continuous variable), solve it with parallel simulated annealing (hobotan, CPU/GPU), a genetic algorithm (DEAP), Gurobi, and the D-Wave hybrid solvers BQM and CQM (via Amplify), and count the variables of the formulations and the CNOT gates of the QAOA circuit.

The numerical results (all data plotted in the figures, Tables S1 and S2, and every run of the hybrid solvers) are provided as Data S1 in the Supporting Information of the article. This repository contains the code for the computations only.

## Environment

Python 3.12. Packages: see `requirements.txt` (the versions used for the manuscript).

- **hobotan** (parallel SA): installed from GitHub at the commit given in `requirements.txt`. Its sampler `MIKASAmpler` obtains its sampling routine from the developer's server when it is run, so an internet connection is required.
- **Gurobi**: a Gurobi license is required.
- **D-Wave hybrid solvers**: accessed through Amplify (`LeapHybridSamplerClient`, `LeapHybridCQMSamplerClient`) with a D-Wave Leap token, which is read from `~/.dwave_token` (or from the file given by the environment variable `DWAVE_TOKEN_FILE`). Running these scripts uses solver time on your Leap account.
- The computations in the manuscript used an Intel Core i9-13900KF CPU and an NVIDIA GeForce RTX 3080 GPU.

Results are written next to the scripts (CSV files and `agg/`), or to the directory given by `QCCC_RESULTS`.

## Files

| Script | Purpose | Output | Manuscript |
|---|---|---|---|
| `hobo_builder.py` | HOBO of the reactor problem (Eq. 14) as a monomial dictionary; export to hobotan and Amplify | – | Section 3.1 |
| `sa_sweep_full.py` | Parallel SA (hobotan `MIKASAmpler`, `T_num` = 50,000) over R, n, λ, and the number of shots; reactor parameters | `sa_full_<TAG>_<CPU/GPU>_summary.csv`, `..._samples.csv` | Sections 3.1–3.3, 4 |
| `ga_R.py` | GA on the original MINLP (continuous x) | `ga_R_<date>.csv` | Sections 3.2, 4 |
| `gurobi_R.py` | Gurobi on the original MINLP (Eq. 12); all reactor combinations for the second-best one | `agg/gurobi_R2to6.csv` | Sections 3.1, 4.2 |
| `hybrid_run.py` | D-Wave BQM (QUBO obtained from the HOBO) | `hybrid_NB_results.csv` | Sections 3.2, 4 |
| `hybrid_cqm_bin.py` | D-Wave CQM (quadratic objective, demand as an inequality constraint) | `hybrid_CQM_bin_results.csv` | Sections 3.2, 4 |
| `hobotan_qubo_check.py` | The same parallel SA on the QUBO obtained from the HOBO (R = 2, 4; n = 10) | `agg/hobotan_qubo_vs_hobo.csv` | Section 4.1, Fig. 3 |
| `count_vars_nb_full.py` | Numbers of variables and terms of the HOBO, the QUBO (Amplify, Ishikawa-KZFD), and the CQM model | `agg/vars_nb_full.csv` | Section 3.1, Table S1 |
| `qaoa_cnot_analysis.py` | CNOT layers of one QAOA layer (largest number of CNOTs acting on one qubit) | `agg/qaoa_cnot.csv` | Section 3.3 |

## Reproducing the results

In the manuscript, a solution is successful if its objective value is at most (1 + ε) times the optimum (19.9625), with ε = 1% (and 10%), for all methods; the success probability and the TTS were computed from the result files written by these scripts.

```bash
# parallel SA: R = 2..6, n = 4..30, lambda = 1e3 and 1e4, 5..200 shots (TAG = NB_R<R>); repeat with CPU
python sa_sweep_full.py GPU 2 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 1000,10000 5,10,20,30,50,75,100,150,200 NB_R2
#   lambda = 1e2 and 1e5 (Fig. S1) and 500 shots were run for a subset of conditions; see Data S1 (sheet SA_summary)

# GA: 100 runs per condition
python ga_R.py 2,3,4,5,6 100,1000,10000,100000 100

# Gurobi
python gurobi_R.py

# D-Wave BQM (QUBO, n = 10, lambda = 1e3, minimum time limit, 5 runs) and CQM (5 runs per condition)
python hybrid_run.py run 2 10 1000 5
python hybrid_run.py run 4 10 1000 5
python hybrid_cqm_bin.py run 2 10 5          # R = 2..6, n = 10, 20, 30

# QUBO comparison, numbers of variables, QAOA circuit
python hobotan_qubo_check.py
python count_vars_nb_full.py
python qaoa_cnot_analysis.py
```

The QUBO is obtained from the HOBO with the Ishikawa-KZFD algorithm of Amplify. The objective value of each solution of parallel SA is recomputed in double precision from the returned bit string.
