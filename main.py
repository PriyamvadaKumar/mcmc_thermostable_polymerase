#!/usr/bin/env python3
"""
main.py 
Author: Priyamvada Kumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, time
from tqdm import tqdm
import sys


# CONFIG

np.random.seed(42)
SEED = 42

BETA_TI_VALUES = [1, 2, 5, 10, 20, 50, 100]
BETA_ID_VALUES = [0.1, 0.2, 0.5, 1, 2, 5, 10]

N_REPS = 5
N_STEPS_PILOT = 1_000
N_STEPS_PROD  = 10_000

WT_SEQ = "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=928))
PROTECTED = range(699, 712)
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {aa: i for i, aa in enumerate(AA)}
P_INDEX = np.random.normal(loc=0.2, scale=0.8, size=(20, 20))

os.makedirs("results", exist_ok=True)
print(f"\n=== Running final benchmark (seed {SEED}) ===\n")


# HELPERS

def mutate(seq: str, n_mut: int = 2) -> str:
    seq = list(seq)
    mutable = [i for i in range(len(seq)) if i not in PROTECTED]
    sites = np.random.choice(mutable, size=n_mut, replace=False)
    for pos in sites:
        cur = seq[pos]
        seq[pos] = np.random.choice([a for a in AA if a != cur])
    return "".join(seq)

def ti_score(seq: str) -> float:
    return sum(P_INDEX[AA_IDX[seq[i]], AA_IDX[seq[i+1]]] for i in range(len(seq)-1))

def identity(seq: str) -> float:
    return sum(a == b for a, b in zip(seq, WT_SEQ)) / len(WT_SEQ)

# PILOT GRID
def simulate_pilot(beta_ti, beta_id, dynamic):
    seq = WT_SEQ
    ti = ti_score(seq)
    acc = 0
    for _ in range(N_STEPS_PILOT):
        cand = mutate(seq)
        cand_ti = ti_score(cand)
        if dynamic:
            drift = np.log(beta_ti + 1) * 0.8
            stab = np.exp(-beta_id / 10)
            val = cand_ti + drift * stab
            curr_val = ti + drift * stab
        else:
            val = cand_ti + np.random.normal(0, 0.5)
            curr_val = ti + np.random.normal(0, 0.5)
        if val > curr_val or np.random.rand() < np.exp(val - curr_val):
            seq, ti = cand, cand_ti
            acc += 1
    return ti, acc / N_STEPS_PILOT

def pilot_grid(dynamic):
    results = []
    prefix = "dynamic_" if dynamic else "static_"
    start = time.time()
    for bt in tqdm(BETA_TI_VALUES, desc=f"β_ti grid ({'Dynamic' if dynamic else 'Static'})"):
        for bi in BETA_ID_VALUES:
            tis, accs = zip(*[simulate_pilot(bt, bi, dynamic) for _ in range(N_REPS)])
            results.append({"beta_ti": bt, "beta_id": bi, "mean_TI": np.mean(tis), "mean_acc": np.mean(accs)})
    df = pd.DataFrame(results)
    df["score"] = df["mean_TI"] * df["mean_acc"]
    best = df.loc[df["score"].idxmax()]
    df.to_csv(f"results/{prefix}pilot_grid_summary.csv", index=False)
    for col, name in [("mean_TI", "TI_mean"), ("mean_acc", "Identity_mean")]:
        norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-12)
        pivot = df.assign(**{f"{col}_norm": norm}).pivot(index="beta_ti", columns="beta_id", values=f"{col}_norm")
        plt.figure(figsize=(6, 5))
        sns.heatmap(pivot, annot=False, cmap="viridis")
        plt.title(f"{prefix}{name} heatmap (normalized)")
        plt.tight_layout()
        plt.savefig(f"results/{prefix}{name}_heatmap.png", dpi=300)
        plt.close()
    elapsed = time.time() - start
    print(f"{prefix} grid done in {elapsed:.2f}s → β_ti={best['beta_ti']}, β_id={best['beta_id']}\n")
    return best["beta_ti"], best["beta_id"], elapsed

# ALGORITHMS

"""Greedy step: select highest scoring among 10 random candidates."""
"""MCMC-old: conservative MH with half beta_ti."""
"""MCMC-new: weighted MH with drift-stabilizer adjustment."""

def greedy_step(seq, ti, beta_ti, beta_id, dynamic):
    best_seq, best_ti = seq, ti
    for _ in range(10):
        cand = mutate(seq)
        cand_ti = ti_score(cand)
        cand_id = identity(cand)
        if dynamic:
            drift = np.log(beta_ti + 1) * 0.8
            stab = np.exp(-beta_id / 10)
            val = cand_ti + drift * stab - beta_id * (1 - cand_id)
        else:
            val = cand_ti - beta_id * (1 - cand_id)
        curr_val = best_ti + (drift * stab if dynamic else 0) - beta_id * (1 - identity(best_seq))
        if val > curr_val:
            best_seq, best_ti = cand, cand_ti
    return best_seq, best_ti

def mcmc_old_step(seq, ti, beta_ti, beta_id, dynamic):
    cand = mutate(seq)
    cand_ti = ti_score(cand)
    cand_id = identity(cand)
    bt = beta_ti / 2
    if dynamic:
        drift = np.log(bt + 1) * 0.8
        stab = np.exp(-beta_id / 10)
        curr_val = ti + drift * stab - beta_id * (1 - identity(seq))
        cand_val = cand_ti + drift * stab - beta_id * (1 - cand_id)
    else:
        curr_val = ti - beta_id * (1 - identity(seq))
        cand_val = cand_ti - beta_id * (1 - cand_id)
    if cand_val > curr_val or np.random.rand() < np.exp(cand_val - curr_val):
        return cand, cand_ti
    return seq, ti

def mcmc_new_step(seq, ti, beta_ti, beta_id, dynamic):
    cand = mutate(seq)
    cand_ti = ti_score(cand)
    cand_id = identity(cand)
    curr_score = beta_ti * ti - beta_id * (1 - identity(seq))
    cand_score = beta_ti * cand_ti - beta_id * (1 - cand_id)
    if dynamic:
        drift = np.log(beta_ti + 1) * 0.8
        stab = np.exp(-beta_id / 10)
        curr_score += drift * stab
        cand_score += drift * stab
    if cand_score > curr_score or np.random.rand() < np.exp(cand_score - curr_score):
        return cand, cand_ti
    return seq, ti

def run_algorithm(alg, beta_ti, beta_id, dynamic):
    seq = WT_SEQ
    ti = ti_score(seq)
    step_fn = {
        "Greedy": lambda s, t: greedy_step(s, t, beta_ti, beta_id, dynamic),
        "MCMC_old": lambda s, t: mcmc_old_step(s, t, beta_ti, beta_id, dynamic),
        "MCMC_new": lambda s, t: mcmc_new_step(s, t, beta_ti, beta_id, dynamic),
    }[alg]
    for _ in range(N_STEPS_PROD):
        seq, ti = step_fn(seq, ti)
    return ti, identity(seq)

def benchmark(beta_ti, beta_id, prefix):
    records = []
    for alg in ["Greedy", "MCMC_new", "MCMC_old"]:
        for rep in range(N_REPS):
            ti, iid = run_algorithm(alg, beta_ti, beta_id, prefix.startswith("dynamic"))
            records.append({"alg": alg, "rep": rep, "TI": ti, "Identity": iid})
    df = pd.DataFrame(records)
    df.to_csv(f"results/{prefix}supplement_final_metrics.csv", index=False)
    
    # Fixed: use string aggregations to avoid FutureWarning
    summary = df.groupby("alg").agg({"TI": ["mean", "std"], "Identity": ["mean", "std"]})
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary = summary.round(4)
    summary.to_csv(f"results/{prefix}summary.csv")
    return summary

# Comparison plot

def plot_comparison(static, dynamic):
    metrics = ["TI_mean", "Identity_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))  # Slightly wider
    algorithms = static.index.tolist()  # ['Greedy', 'MCMC_new', 'MCMC_old']
    x = np.arange(len(algorithms))
    w = 0.35

    for i, m in enumerate(metrics):
        # Means
        static_mean = static[m].values
        dynamic_mean = dynamic[m].values
        # Std dev
        static_std = static[m.replace('_mean', '_std')].values
        dynamic_std = dynamic[m.replace('_mean', '_std')].values

        axes[i].bar(x - w/2, static_mean, w, yerr=static_std, capsize=5,
                    label="Static", color='#1f77b4', alpha=0.8)
        axes[i].bar(x + w/2, dynamic_mean, w, yerr=dynamic_std, capsize=5,
                    label="Dynamic", color='#ff7f0e', alpha=0.8)

        axes[i].set_xticks(x)
        axes[i].set_xticklabels(algorithms, rotation=45, ha='right')  # KEY LINE
        axes[i].set_title(m.replace("_mean", ""))
        axes[i].set_ylabel(m.replace("_mean", "") + " (mean ± std)")
        axes[i].legend()

    plt.tight_layout(pad=2.0)  # Extra padding
    plt.savefig("results/static_vs_dynamic_comparison.png",
                 dpi=300, bbox_inches='tight', pad_inches=0.3)  # Extra pad
    plt.close()


# RUN

opt_static = pilot_grid(dynamic=False)
opt_dynamic = pilot_grid(dynamic=True)

summary_static = benchmark(opt_static[0], opt_static[1], "static_")
summary_dynamic = benchmark(opt_dynamic[0], opt_dynamic[1], "dynamic_")

print("--- Static summary ---")
print(summary_static)
print("\n--- Dynamic summary ---")
print(summary_dynamic)

plot_comparison(summary_static, summary_dynamic)


# CONVERGENCE PLOTS

def run_trace(alg, beta_ti, beta_id, dynamic, n_steps=2000):
    """Run a short production trace to record TI and acceptance over steps."""
    seq = WT_SEQ
    ti = ti_score(seq)
    ti_trace = []
    acc_trace = []
    accept_count = 0

    step_fn = {
        "Greedy": lambda s, t: greedy_step(s, t, beta_ti, beta_id, dynamic),
        "MCMC_old": lambda s, t: mcmc_old_step(s, t, beta_ti, beta_id, dynamic),
        "MCMC_new": lambda s, t: mcmc_new_step(s, t, beta_ti, beta_id, dynamic),
    }[alg]

    for step in range(1, n_steps + 1):
        old_ti = ti
        seq, ti = step_fn(seq, ti)
        if ti != old_ti:
            accept_count += 1
        ti_trace.append(ti)
        acc_trace.append(accept_count / step)

    return pd.DataFrame({
        "step": np.arange(1, n_steps + 1),
        "TI": ti_trace,
        "Acceptance": acc_trace
    })


def plot_convergence(beta_ti_static, beta_id_static, beta_ti_dynamic, beta_id_dynamic):
    """Generate convergence traces for each algorithm under static and dynamic models."""
    os.makedirs("results/convergence", exist_ok=True)
    n_steps = 2000  # Compact: enough to visualize trends

    for alg in ["Greedy", "MCMC_new", "MCMC_old"]:
        print(f"Running convergence trace for {alg}...")
        df_static = run_trace(alg, beta_ti_static, beta_id_static, dynamic=False, n_steps=n_steps)
        df_dynamic = run_trace(alg, beta_ti_dynamic, beta_id_dynamic, dynamic=True, n_steps=n_steps)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plt.suptitle(f"Convergence of {alg}", fontsize=14, fontweight="bold")

        # TI trace
        axes[0].plot(df_static["step"], df_static["TI"], label="Static", color="#1f77b4")
        axes[0].plot(df_dynamic["step"], df_dynamic["TI"], label="Dynamic", color="#ff7f0e")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Thermostability Index (TI)")
        axes[0].set_title("TI Convergence")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Acceptance trace
        axes[1].plot(df_static["step"], df_static["Acceptance"], label="Static", color="#1f77b4")
        axes[1].plot(df_dynamic["step"], df_dynamic["Acceptance"], label="Dynamic", color="#ff7f0e")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Acceptance Rate")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Acceptance Rate Over Time")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = f"results/convergence/{alg.lower()}_convergence.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  → Saved convergence trace to {out_path}\n")


# Generate convergence traces
plot_convergence(opt_static[0], opt_static[1], opt_dynamic[0], opt_dynamic[1])


# Run report
run_report = {
    "random_seed": SEED,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "python_version": sys.version.split()[0],
    "config": {
        "BETA_TI_VALUES": BETA_TI_VALUES,
        "BETA_ID_VALUES": BETA_ID_VALUES,
        "N_REPS": N_REPS,
        "N_STEPS_PILOT": N_STEPS_PILOT,
        "N_STEPS_PROD": N_STEPS_PROD
    },
    "results": {
        "static_best": {"beta_ti": opt_static[0], "beta_id": opt_static[1]},
        "dynamic_best": {"beta_ti": opt_dynamic[0], "beta_id": opt_dynamic[1]},
        "static_runtime_s": round(opt_static[2], 3),
        "dynamic_runtime_s": round(opt_dynamic[2], 3)
    },
    "summary": {
        "static": summary_static.to_dict(),
        "dynamic": summary_dynamic.to_dict()
    }
}
with open("results/run_report.json", "w") as f:
    json.dump(run_report, f, indent=4)

print("\nAll results saved in results/ (including run_report.json)")
