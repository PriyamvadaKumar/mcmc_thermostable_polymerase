# MCMC vs. Greedy Evolution of Thermostable DNA Polymerase 

---

## Overview

Compares **single-chain Metropolis-Hastings MCMC** and **greedy iterative selection** for *in silico* directed evolution of thermostable *E. coli* DNA polymerase I (UniProt: **P00582**).

**Key result**: Both methods saturate **Ku et al.'s TI = 22.5854**, but **greedy retains 82.5% sequence identity**, while **MCMC drifts to 8.1%** (p = 2.87×10⁻¹¹).

---

## Methods Summary

- **10,000 double-mutation proposals** per run (both algorithms)
- **Active site protected**: Val700–Arg712
- **Fitness**: `score = (TI_new / TI_old) × identity`
- **30 independent replicates**
- **Statistical test**: Wilcoxon rank-sum

---

## Results

| Algorithm | Final TI | Identity to WT | p-value (vs Greedy) |
|---------|----------|----------------|---------------------|
| **Greedy** | 22.5854 ± 0.0 | **0.8245 ± 0.0068** | — |
| **MCMC**   | 22.5854 ± 0.0 | 0.0813 ± 0.0076 | 2.87e-11 |

> **MCMC reaches maximum TI but fails to preserve functional similarity due to neutral drift after TI saturation.**

---

## Run

```bash
pip install -r requirements.txt
python main.py


Reference
Ku, T., et al. (2009). "Prediction of protein thermostability using dipeptide composition." Protein Engineering, Design & Selection, 22(8), 515–521.
DOI: 10.1093/protein/gzp077

