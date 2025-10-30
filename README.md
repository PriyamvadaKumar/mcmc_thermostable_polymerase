# Benchmarking Greedy and MCMC Strategies for Synthetic Thermostability Optimization of *E. coli* DNA Polymerase I

**Priyamvada Kumar**  

---

## Abstract

Directed evolution through iterative mutation and selection is a central approach in protein engineering. This work presents a reproducible synthetic benchmark comparing greedy selection and Markov Chain Monte Carlo (MCMC) strategies for optimizing thermostability of *E. coli* DNA polymerase I (UniProt: P00582). A unified Python pipeline (`main.py`) performs pilot grid searches over thermal intensity and identity-weight parameters, runs algorithmic benchmarks under static and dynamic scoring models, and exports normalized heatmaps, comparative plots, and a comprehensive run report (`run_report.json`). The dynamic scoring model generates interpretable trade-offs between thermostability and sequence identity. **The static model preserves up to 3.3× more identity than dynamic for MCMC-new**, while dynamic enables tunable drift for exploratory sampling. The code and results are fully reproducible and suitable as a computational writing sample with straightforward extension to empirical datasets.

---

## Key Results

- **Greedy**: Maximizes thermostability (TI ≈ 1308) but sacrifices identity.
- **MCMC-new**: Balances TI and identity.
- **MCMC-old**: Most conservative — retains highest identity.
- **Dynamic model**: Enables **3.3× identity retention** under fine-tuned parameters.

---

## Results Visualization

### Pilot Grid Search Heatmaps (Normalized)

| Static TI | Static Identity |
|---------|---------|
<img width="1800" height="1500" alt="static_TI_mean_heatmap" src="https://github.com/user-attachments/assets/20014643-cc20-4761-a909-7273795af12b" /> | <img width="1800" height="1500" alt="static_Identity_mean_heatmap" src="https://github.com/user-attachments/assets/4874ca86-bc30-432b-8240-988122ce108c" />


| Dynamic TI | Dynamic Identity |
|----------|-----------|
<img width="1800" height="1500" alt="dynamic_TI_mean_heatmap" src="https://github.com/user-attachments/assets/6527b8c5-949d-4d9c-8f36-a89f55d21614" /> | <img width="1800" height="1500" alt="dynamic_Identity_mean_heatmap" src="https://github.com/user-attachments/assets/58fc115d-e00d-4c32-adf5-7a8d1cbbec02" />


### Final Benchmark: Static vs Dynamic

<img width="3913" height="1662" alt="static_vs_dynamic_comparison" src="https://github.com/user-attachments/assets/d4cf73f9-3d1c-49aa-ba4e-1173b985b9e7" />
*(Bar plot with error bars showing mean ± std across 5 replicates)*

---

## Installation

```bash
git clone https://github.com/yourusername/mcmc_thermostable_polymerase.git
cd mcmc_thermostable_polymerase
pip install numpy pandas matplotlib seaborn tqdm biopython
python main.py
```

## Future Work

- Multi-objective optimization: Identify Pareto-efficient variants using NSGA-II.

- Bayesian hyperparameter tuning: Adapt β_ti and β_id during sampling.

- Empirical calibration: Replace synthetic proxy with FoldX/Rosetta/DeepDDG.

- Real data integration: Load P00582 via UniProt + Biopython.

- Validation databases: FireProtDB, ProTherm.

- Wet-lab validation: Deep mutational scanning (DMS) or PCR assays.

### Three-phase roadmap:

- Synthetic tuning (this work)

- Semi-empirical scoring

- Empirical validation


## References

1. Bloom, J. D., et al. (2006). Protein stability promotes evolvability. PNAS, 103(15), 5869–5874.
https://doi.org/10.1073/pnas.0510098103

2. Goldenzweig, A., & Fleishman, S. J. (2018). Principles of protein stability and their application in computational design. Annu. Rev. Biochem., 87, 105–129.
https://doi.org/10.1146/annurev-biochem-062917-012102

3. Kuhlman, B., & Bradley, P. (2019). Advances in protein structure prediction and design. Nat. Rev. Mol. Cell Biol., 20, 681–697.
https://doi.org/10.1038/s41580-019-0163-x

4. Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Trans. Evol. Comput., 6(2), 182–197.
https://doi.org/10.1109/4235.996017

5. Snoek, J., et al. (2012). Practical Bayesian optimization of machine learning algorithms. NeurIPS, 25.
https://arxiv.org/abs/1206.2944

6. Musil, M., et al. (2021). FireProtDB: A database of manually curated protein stability data. Nucleic Acids Res., 49(D1), D319–D324.
https://doi.org/10.1093/nar/gkaa1083

7. Kumar, M. D. S., et al. (2006). ProTherm and ProNIT: Thermodynamic databases for proteins and protein–nucleic acid interactions. Nucleic Acids Res., 34(suppl_1), D204–D206.
https://doi.org/10.1093/nar/gkj103



