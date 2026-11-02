# Clamped-Biomass Dynamic FBA Simulation
Benchmarks giFBA against a clamped-biomass dynamic FBA implementation for the following purposes:

- Defining when giFBA is analogous to dynamic FBA
- Understanding what dFBA hyperparams best match giFBA
- Benchmarking efficiency differences in giFBA vs dFBA

The experiments, result fields, and notebooks are described in
[`docs/examples/dfba_benchmark.md`](../../../docs/examples/dfba_benchmark.md).

## Quick Use
Requires plotly, seaborn, tol_colors and matplotlib on top of `gifba` (the analysis was run in the `M3` conda env). Unzip the raw results in place first (`unzip data/raw/Benchmark_AH_BI.zip -d data/raw/`), then from this directory:

```bash
python process_results.py                  # every experiment
python process_results.py --config AH_BI   # one experiment
```

The scripts in `vacc_scripts/` ran on the UVM VACC and write to a relative `Benchmark_*` folder on the cluster, so their paths are not tied to this layout.
