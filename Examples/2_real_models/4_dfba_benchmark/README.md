# Clamped-Biomass Dynamic FBA Simulation
This directory aims to compare how giFBA compared to dFBA, as giFBA is a steady state method for emergent metabolic interactions from dynamic FBA. This directory benchmarks several factors for the following purposes:

- Defining when giFBA is analogous to dynamic FBA
- Understanding what dFBA hyperparams best match giFBA
- Benchmarking efficiency differences in giFBA vs dFBA

## Layout
```
4_dfba_benchmark
 ┣ 4_*.ipynb                  Analysis notebooks
 ┣ process_results.py         Raw JSON -> processed tables
 ┣ benchmark_analysis_utils.py
 ┣ vacc_scripts/              Cluster scripts that ran the simulations
 ┗ data/
   ┣ AH_BI/, N_orgs/          Processed tables (csv, pkl)
   ┗ raw/                     Zipped raw results
```

Requires plotly, seaborn, tol_colors and matplotlib on top of `gifba` (the analysis was run in the `M3` conda env).

## Experiments
This directory was used to simulate multiple configurations:
- 1 microbe only communities, which various y0 transformations are used
- AH BI only communities, with various y0 transformations. This community shows a curious interaction, and understanding the media transformations is curious and functional, so methods are maintained on these communities
- 2 microbe only communities, varying dfba params across the community
- N_microbes comparing communities with 2-10 models to benchmark efficiency and identify dfba params. 

## Results
The simulations were run on the UVM VACC with the scripts in `vacc_scripts/`, which write one JSON file per run. Those results are zipped in `data/raw/` (`Benchmark_AH_BI.zip`, `Benchmark_N_orgs.zip`); unzip them in place to reprocess (`unzip data/raw/Benchmark_AH_BI.zip -d data/raw/`). The unzipped folders are gitignored. The VACC scripts write to a relative `Benchmark_*` folder on the cluster, so their paths are not tied to this layout.

Each result stores the following information:
- Number of organisms
- Model Names of each GEM in the community
- Relative abundance of each GEM in the community 
- y0_factor: Since media is stored as a flux, this defines how much each media component's concentration is scaled up/down from the original flux dataset - this only affects metabolites in the media, components not present in the base media remain 0.
- y0_offset: This is a flat value used to offset the concentrations calculated. This amount is added to the media component's concentration only if it is present in the base media. 
- Sum_N_rxns: Sum of the length of all the reactions in the community (ignoring duplicates across organisms) as an analogue for the complexity of each community
- Sum_N_mets: Same as the reactions, but summed across the metabolites
- vmax and km: michaelis menten kinetics values. Each metabolite is assumed to have exactly these same identical constants. 
- solver_method: the chosen method for solving the dfba simulation, using scipy `solve_ivp`. May be either "BDF" or "RK45" to understand which solver may be faster
- early_stop_window and early_stop_tol: dfba is considered converged once the concentrations are linear over time. This is done by running a linear regression after each time step. The window defines the number of points used for the regressor. The stop tolerance defines the threshold for the residuals in order to be considered sufficiently linear. When a model is considered linear, the dfba simulation is halted.
- model_solver: generally listed as "glpk". This tracks what solver is used to solve each LP in the GEMs. "glpk" is slower, but used since gurobi implements limits on the number of parallel simulations allowed. 
- gifba_solve_time and dfba_solve_time: The wall clock time required to solve the respective simulation types. This is stored only after the simulation converges/completes. 
- gifba_pfba_calls and dfba_pfba_calls: The number of times pfba is solved (each technique simulates each organism at least 1 time per iteration/time step). This is a stronger display of efficiency compared to wall clock time, due to wall clock time's dependence on the computer
- dfba_status: logs the current number of pfba calls while dfba is simulating. This is kept in case the method times out on the UVM VACC, then the number of calls can still be recorded and used - the wall clock time will simply be replaced with the timeout length in analyses
- error: this is in case a simulation fails for development purposes to see why it failed - other lines may be added in this case, but primarily for development only
- status: this logs the status of the simulation for tracking where the simulations are at each time step. This will match dfba_status during dfba runs, but will state if params are loaded, models are loaded, if gifba was simulated, etc. Generally ignorable
- gifba_fluxes & dfba_fluxes: a dataframe turned to a dict (for json compatibility) for the flux of each reaction in each microbe for the respective method. This is used to compare accuracy 

## Benchmark Analysis
`process_results.py` loads the raw JSON once and writes the processed tables to `data/{AH_BI,N_orgs}/` (`hyperparams_*.csv`, `model_names_*.csv`, `fluxes_*.pkl`):

```bash
python process_results.py                  # every experiment
python process_results.py --config AH_BI   # one experiment
```

The notebooks read those tables and save figures to `Examples/figures/real_model_figs/dfba_benchmark/`:
- `4_1_compare_hyperparameters.ipynb`: wall clock and pFBA-call efficiency, and MAE/RPE against the dFBA hyperparameters
- `4_2_compare_fluxes.ipynb`: reaction-level giFBA vs. dFBA flux comparison
- `4_3_pca_comparison_AH_BI.ipynb`: PCA of giFBA and dFBA fluxes for the A. hallii / B. infantis community (needs a local AGORA2 install set in `agora_dir_base`, plus the `.mat` models in `../data/`)

Set `ACTIVE_CONFIG_NAME` at the top of `4_1`/`4_2` to switch between experiments.

Each notebook does (or will) include a random forest regressor to understand what community metadata and dfba hyper params (vmax, km, N_models, early_stop_window, early_stop_tol, solver_method, relative_abundance) may affect the accuracy. The aim is to be able to use this data to inform how dfba and gifba compare answering a form of the question: "giFBA is a steady state method for estimating a dfba solution, but what kind of dfba simulation is it efficiently solving?"