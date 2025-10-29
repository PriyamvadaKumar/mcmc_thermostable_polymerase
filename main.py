#!/usr/bin/env python
"""
MCMC vs. Greedy for Thermostable E. coli DNA Polymerase I
Author: Priyamvada Kumar
"""

import os, random, warnings, requests, io
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from Bio import SeqIO

warnings.filterwarnings("ignore")

# ====================== CONFIG ======================
UNIPROT_ID = "P00582"
FASTA_URL = f"https://www.uniprot.org/uniprot/{UNIPROT_ID}.fasta"
ACTIVE_SITE = range(699, 712)  # 0-indexed
N_MUTATIONS = 2
TOTAL_PROPOSALS = 10_000
N_REPLICATES = 30
SEED = 42
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

# ====================== FETCH SEQUENCE (FIXED) ======================
def fetch_seq():
    response = requests.get(FASTA_URL)
    response.raise_for_status()
    fasta_io = io.StringIO(response.text)  # Use StringIO
    record = next(SeqIO.parse(fasta_io, "fasta"))
    return str(record.seq)

WT_SEQ = fetch_seq()
L = len(WT_SEQ)
print(f"Loaded {UNIPROT_ID}: {L} aa")

# ====================== LOAD FULL P-INDEX (400 entries) ======================
P_DF = pd.read_csv("data/p_index_full.csv", index_col=0)
P_INDEX = P_DF.stack().to_dict()  # (aa1, aa2) → value

def ti(seq):
    if len(seq) < 2: return 0.0
    total = sum(P_INDEX.get(seq[i:i+2], 100.0) for i in range(len(seq)-1))
    return (100 / (len(seq)-1)) * (total - 9372) / 398

ti_wt = ti(WT_SEQ)
print(f"Wild-type TI: {ti_wt:.3f}")

# ====================== TOOLS ======================
AA = "ACDEFGHIKLMNPQRSTVWY"

def mutate(seq, n=2, protected=None):
    s = list(seq)
    pos = [i for i in range(len(s)) if protected is None or i not in protected]
    if len(pos) < n: return seq
    for p in random.sample(pos, n):
        cur = s[p]
        s[p] = random.choice([a for a in AA if a != cur])
    return "".join(s)

def identity(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a)

# ====================== MCMC (Single MH) ======================
def mcmc_run(wt, steps=10_000):
    cur = wt
    ti_hist = [ti(cur)]
    sim_hist = [1.0]
    for _ in range(steps):
        cand = mutate(cur, n=N_MUTATIONS, protected=ACTIVE_SITE)
        ti_old, ti_new = ti(cur), ti(cand)
        sim = identity(cand, wt)
        score = (ti_new / ti_old) * sim if ti_old > 0 else 0
        if score > 1 or random.random() < score:
            cur = cand
        ti_hist.append(ti(cur))
        sim_hist.append(identity(cur, wt))
    return cur, ti_hist, sim_hist

# ====================== GREEDY (100 gen × 100 child) ======================
def greedy_run(wt, gens=100, children=100):
    parent = wt
    ti_hist = [ti(parent)]
    sim_hist = [1.0]
    for _ in range(gens):
        offs = [mutate(parent, n=N_MUTATIONS, protected=ACTIVE_SITE) for _ in range(children)]
        ti_vals = [ti(c) for c in offs]
        sim_vals = [identity(c, wt) for c in offs]
        best_idx = np.argmax(ti_vals)
        if random.random() < sim_vals[best_idx]:
            parent = offs[best_idx]
        ti_hist.append(ti(parent))
        sim_hist.append(identity(parent, wt))
    return parent, ti_hist, sim_hist

# ====================== RUN 30 REPLICATES ======================
results = {"MCMC": [], "Greedy": []}
for alg in ["MCMC", "Greedy"]:
    func = mcmc_run if alg == "MCMC" else greedy_run
    kwargs = {"steps": TOTAL_PROPOSALS} if alg == "MCMC" else {"gens": 100, "children": 100}
    for rep in range(N_REPLICATES):
        print(f"{alg} rep {rep+1}/{N_REPLICATES}", end="\r")
        final_seq, ti_traj, sim_traj = func(WT_SEQ, **kwargs)
        results[alg].append({
            "final_ti": ti_traj[-1],
            "final_sim": sim_traj[-1],
            "ti_traj": ti_traj,
            "sim_traj": sim_traj
        })
    print()

# ====================== ANALYZE ======================
df = pd.DataFrame([
    {"alg": k.upper(), "rep": i, "TI": r["final_ti"], "Similarity": r["final_sim"]}
    for k, reps in results.items() for i, r in enumerate(reps)
])

summary = df.groupby("alg").agg(
    TI_mean=("TI","mean"), TI_std=("TI","std"),
    Sim_mean=("Similarity","mean"), Sim_std=("Similarity","std")
).round(4)

print("\n=== FINAL PERFORMANCE (30 replicates) ===")
print(summary)

mcmc_ti = df[df["alg"]=="MCMC"]["TI"]
greedy_ti = df[df["alg"]=="GREEDY"]["TI"]
_, p_ti = stats.ranksums(mcmc_ti, greedy_ti)

mcmc_sim = df[df["alg"]=="MCMC"]["Similarity"]
greedy_sim = df[df["alg"]=="GREEDY"]["Similarity"]
_, p_sim = stats.ranksums(mcmc_sim, greedy_sim)

print(f"Wilcoxon TI p-value: {p_ti:.2e}")
print(f"Wilcoxon Similarity p-value: {p_sim:.2e}")

# ====================== PLOT ======================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
for alg, col in zip(["MCMC","Greedy"], ["C0","C1"]):
    trajs = [r["ti_traj"] for r in results[alg]]
    mean = np.mean(trajs, axis=0); std = np.std(trajs, axis=0)
    x = np.arange(len(mean))
    plt.plot(x, mean, label=alg, color=col)
    plt.fill_between(x, mean-std, mean+std, color=col, alpha=0.2)
plt.title("TI Trajectory"); plt.xlabel("Step / Generation"); plt.ylabel("TI"); plt.legend()

plt.subplot(1,2,2)
for alg, col in zip(["MCMC","Greedy"], ["C0","C1"]):
    trajs = [r["sim_traj"] for r in results[alg]]
    mean = np.mean(trajs, axis=0); std = np.std(trajs, axis=0)
    x = np.arange(len(mean))
    plt.plot(x, mean, label=alg, color=col)
    plt.fill_between(x, mean-std, mean+std, color=col, alpha=0.2)
plt.title("Sequence Similarity"); plt.xlabel("Step / Generation"); plt.ylabel("Identity to WT"); plt.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/trajectories.png", dpi=300)
plt.show()

# ====================== SAVE ======================
df.to_csv(f"{OUTPUT_DIR}/supplement_final_metrics.csv", index=False)
summary.to_csv(f"{OUTPUT_DIR}/summary.csv")
print(f"\nResults saved to: {OUTPUT_DIR}/")
