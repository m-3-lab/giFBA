import os
import copy
import json
import time
import traceback
from tqdm import tqdm

import numpy as np
import pandas as pd
import cobra as cb
from scipy.integrate import solve_ivp

import gifba


# Paths & constants
agora_dir_base = "/gpfs2/scratch/rdsiegel/agora2_shared"
# agora_dir_base = "/home/rseag/AGORA2_All_Models"
EURO_MEDIA_FILE = "/users/r/d/rdsiegel/giFBA/dFBA_runs/euro_diet.tsv"
# EURO_MEDIA_FILE = "/home/rseag/UVM/M3_Lab/giFBA/Examples/2_real_models/data/euro_diet.tsv"
output_dir = "/home/rseag/UVM/M3_Lab/giFBA/Examples/2_real_models/glpk_benchmark_results"
LOG_EVERY_N_CALLS = 500
OLD_BIOMASS_ID = "EX_biomass(e)"
COBRA_SOLVER = "glpk"
t_start, t_end = 0, 100
GIFBA_ITERATIONS = 100

# Monte Carlo sampling space for (N_models, Vmax, Km, ODE solver)
N_models_list = list(range(1, 11))
V_MAX_LIST = list(np.logspace(2, 3, 100))  # (mmol/(gi*hr)) max uptake rate
KM_LIST = list(np.logspace(-1, 0, 100))    # (mmol) Michaelis-Menten constant
SOLVER_LIST = ["RK45", "BDF"]
EARLY_STOP_WINDOW_LIST = list(range(4, 30)) # window size for linear regression in early stopping event
EARLY_STOP_TOL_LIST = list(np.logspace(-7, -4, 100))  # tolerances for residuals in early stopping event

array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
# array_job_id = np.random.randint(100000, 999999)          # For local testing, generate a random job ID
# task_id = np.random.randint(1, 100)                       # For local testing, generate a random task ID
job_id = f"{array_job_id}_{task_id}" if array_job_id else "local_run"
output_file = os.path.join(output_dir, f"{job_id}.json")

# experimental loc
experiment_logs = {}
n_dfba_pfba_calls = 0


def init_logs():
    global experiment_logs
    experiment_logs = {
        "N_models": None,
        "model_names": None,
        "sum_N_rxns": None,
        "sum_N_mets": None,
        "vmax": None,
        "km": None,
        "solver_method": None,
        "early_stop_window": None,
        "early_stop_tol": None,
        "model_solver": COBRA_SOLVER,
        "gifba_solve_time": None,
        "dfba_solve_time": None,
        "gifba_pfba_calls": None,
        "dfba_pfba_calls": None,
        "dfba_status": "Not Started",
        "error": None,
        "status": None,
        "gifba_fluxes": None,
        "dfba_fluxes": None,
    }


def save_checkpoint(status=None):
    """Saves current state of experiment_logs to JSON."""
    if status:
        experiment_logs["status"] = status
    if status.startswith("dfba"):
        experiment_logs["dfba_status"] = status
    experiment_logs["dfba_pfba_calls"] = int(n_dfba_pfba_calls)

    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(experiment_logs, f, indent=4)


# Model loading
def rename_community_biomass(models, old_biomass_id="EX_Bio(e)"):
    """
    Renames the biomass exchange reactions and their associated metabolites
    to track individual species separately.
    """
    # Deepcopy ensures we don't accidentally modify the original template models
    updated_models = copy.deepcopy(models)

    for idx, model in enumerate(updated_models):
        suffix = f"_{idx + 1}"  # e.g., _1, _2

        if old_biomass_id in model.reactions:
            # 1. Isolate the biomass reaction
            biomass_rxn = model.reactions.get_by_id(old_biomass_id)

            # 2. Rename the associated biomass metabolite(s) FIRST
            # Cast to list() to avoid modifying a dictionary while iterating over it
            for metabolite in list(biomass_rxn.metabolites):
                metabolite.id = f"{metabolite.id}{suffix}"
                metabolite.name = f"{metabolite.name} (Species {idx + 1})"

            # 3. Rename the reaction itself
            biomass_rxn.id = f"{old_biomass_id}{suffix}"
            biomass_rxn.name = f"{biomass_rxn.name} (Species {idx + 1})"

    return updated_models


def load_agora_models(N):
    selected_models = []
    selected_models_files = []

    # List of directory contents to randomly choose from
    model_files = [
        name for name in os.listdir(agora_dir_base)
        if os.path.isfile(os.path.join(agora_dir_base, name))
    ]

    for _ in range(N):
        model_file = np.random.choice(model_files)
        while model_file in selected_models_files:
            model_file = np.random.choice(model_files) # avoid duplicates - not necessary but nice for diversity
        model_path = os.path.join(agora_dir_base, model_file)
        model = cb.io.load_matlab_model(model_path)
        model.solver = COBRA_SOLVER

        selected_models.append(model)
        selected_models_files.append(model_file)

    return selected_models, selected_models_files


def load_medium():
    """Load the Euro-diet medium and key it by AGORA-style exchange IDs."""
    df = pd.read_csv(EURO_MEDIA_FILE, sep="\t", header=0, index_col=0)
    flux_dict = df.to_dict()["Flux Value"]
    return {ex.replace("[e]", "(e)"): -flux for ex, flux in flux_dict.items()}


def sample_parameters():
    """Randomly draw one Monte Carlo parameter set."""
    N = int(np.random.choice(N_models_list))
    V_MAX_ALL = float(np.random.choice(V_MAX_LIST))
    K_M_ALL = float(np.random.choice(KM_LIST))
    SOLVER = str(np.random.choice(SOLVER_LIST))
    EARLY_STOP_WINDOW_SIZE = int(np.random.choice(EARLY_STOP_WINDOW_LIST))
    EARLY_STOP_TOL = float(np.random.choice(EARLY_STOP_TOL_LIST))
    return N, V_MAX_ALL, K_M_ALL, SOLVER, EARLY_STOP_WINDOW_SIZE, EARLY_STOP_TOL


# clamped dFBA
def dsdt(t, S, models, vmax, km, media_keys, community, rel_abund, pbar=None):
    global n_dfba_pfba_calls

    dsdt_rates = np.zeros(len(S))

    # Quick lookup: {metabolite_id: current_concentration}
    conc_map = dict(zip(media_keys, S))

    for mdl_idx, model in enumerate(models):
        # Update lower bounds based on Michaelis-Menten kinetics
        for ex_id in media_keys:
            if ex_id in model.reactions and "bio" not in ex_id:
                v_max = vmax.loc[str(mdl_idx), ex_id]
                k_m = km.loc[str(mdl_idx), ex_id]
                c = conc_map[ex_id]

                if c < 1e-6:
                    c = 0.0  # avoid very small concentrations causing numerical errors

                uptake_limit = (v_max * c) / (k_m + c)  # (mmol/(gi*hr))
                model.reactions.get_by_id(ex_id).lower_bound = -uptake_limit

        # Run pFBA
        if pbar is not None:
            pbar.set_postfix_str(f"growth = {model.slim_optimize():.6f}")
        solution = cb.flux_analysis.parsimonious.pfba(model)
        n_dfba_pfba_calls += 1

        # Add fluxes to the net rate of change (weighted by clamped relative abundance)
        if solution.status == "optimal":
            for i, ex_id in enumerate(media_keys):
                flux_per_hr = solution.fluxes.get(ex_id, 0)
                dsdt_rates[i] += flux_per_hr * rel_abund[mdl_idx].item()

    # Add baseline media exchange fluxes (fixed, taken from the giFBA solution)
    for i, ex_id in enumerate(media_keys):
        media_flux = community.env_fluxes.loc[0, ex_id]
        dsdt_rates[i] += media_flux

    if n_dfba_pfba_calls % LOG_EVERY_N_CALLS == 0:
        save_checkpoint(f"dfba_in_progress_{n_dfba_pfba_calls}")

    return dsdt_rates


class dFBATracker:
    """Wraps dsdt() for solve_ivp and implements an early-stopping event
    based on the residual of a rolling-window linear fit (steady state)."""

    def __init__(self, t_start, t_end, window_size=5, tol=1e-5):
        self.highest_t = t_start
        self.tol = tol
        self.window_size = window_size

        # History for early stopping
        self.t_history = []
        self.s_history = []

        self.t_start = t_start
        self.t_end = t_end
        self.last_t = t_start
        self.pbar = tqdm(total=self.t_end, desc="dFBA Simulation Progress", unit="s")

    def ode_wrapper(self, t, S,*args):
        self.pbar.n = t
        self.pbar.refresh()

        
        return dsdt(t, S, *args)

    def check_stop_condition(self, t, S, *args):
        self.t_history.append(t)
        self.s_history.append(np.copy(S))

        if len(self.t_history) < self.window_size:
            return 1.0  # keep running

        t_recent = np.array(self.t_history[-self.window_size:])
        s_recent = np.array(self.s_history[-self.window_size:])

        _, residuals, _, _, _ = np.polyfit(t_recent, s_recent, deg=1, full=True)

        if residuals.size == 0:
            max_residual = 0.0
        else:
            rmse_per_variable = np.sqrt(residuals / self.window_size)
            max_residual = np.max(rmse_per_variable)

        if max_residual < self.tol:
            return 0.0
        
        return 1.0  # keep running


# run gifba algorithm
def benchmark_gifba(renamed_models, medium, rel_abund):
    community = gifba.gifbaObject(renamed_models, [medium, 0.1], rel_abund=rel_abund)

    gifba_start_time = time.time()
    env_flux, org_flux = community.run_gifba(n_iterations=GIFBA_ITERATIONS, method="pfba")
    gifba_end_time = time.time()

    gifba_solve_time = gifba_end_time - gifba_start_time
    gifba_pfba_calls = community.simulation_count

    return community, env_flux, org_flux, gifba_solve_time, gifba_pfba_calls

# run dfba algo.
def benchmark_dfba(renamed_models, vmax_vals, km_vals, m_keys, y0, community, rel_abund, solver, early_stop_window, early_stop_tol):
    tracker = dFBATracker(t_start=t_start, t_end=t_end, window_size=early_stop_window, tol=early_stop_tol)

    def steady_state_event(t, S, *args):
        return tracker.check_stop_condition(t, S, *args)

    steady_state_event.terminal = True
    steady_state_event.direction = -1  # trigger only dropping below/to zero

    # Deep-copy models so the dFBA phase never mutates the bounds giFBA left behind
    dfba_args = (copy.deepcopy(renamed_models), vmax_vals, km_vals, m_keys, community, rel_abund, tracker.pbar)

    dfba_time_start = time.time()
    sol = solve_ivp(
        tracker.ode_wrapper,
        t_span=(t_start, t_end),
        y0=y0,
        args=dfba_args,
        method=solver,
        events=[steady_state_event],
    )
    dfba_time_end = time.time()
    dfba_solve_time = dfba_time_end - dfba_time_start

    return sol, dfba_solve_time


def main():
    global n_dfba_pfba_calls
    n_dfba_pfba_calls = 0
    init_logs()

    try:
        N, V_MAX_ALL, K_M_ALL, SOLVER, EARLY_STOP_WINDOW_SIZE, EARLY_STOP_TOL = sample_parameters()
        experiment_logs.update(N_models=N, 
                               vmax=V_MAX_ALL, 
                               km=K_M_ALL, 
                               solver_method=SOLVER, 
                               early_stop_window=EARLY_STOP_WINDOW_SIZE, 
                               early_stop_tol=EARLY_STOP_TOL)
        save_checkpoint(status="params_sampled")

        # change dfba logger to log every LOG_EVERY_N_CALLS * N_models calls
        global LOG_EVERY_N_CALLS
        LOG_EVERY_N_CALLS = LOG_EVERY_N_CALLS * N

        # AGORA model selection and loading
        selected_models, selected_models_files = load_agora_models(N)
        renamed_models = rename_community_biomass(selected_models, old_biomass_id=OLD_BIOMASS_ID)

        N_rxns = sum(len(model.reactions) for model in renamed_models)
        N_mets = sum(len(model.metabolites) for model in renamed_models)

        experiment_logs.update(
            model_names=selected_models_files,
            sum_N_rxns=int(N_rxns),
            sum_N_mets=int(N_mets),
        )
        save_checkpoint(status="models_loaded")

        # giFBA
        rel_abund = np.ones(N) / N
        medium = load_medium()

        community, env_flux, org_flux, gifba_solve_time, gifba_pfba_calls = benchmark_gifba(
            renamed_models, medium, rel_abund
        )
        experiment_logs.update(
            gifba_solve_time=float(gifba_solve_time),
            gifba_pfba_calls=int(gifba_pfba_calls),
        )
        save_checkpoint(status="gifba_complete")

        # Clamped dFBA
        rel_abund = community.rel_abund.copy()

        substrate_0 = community.env_fluxes.loc[0, :].to_dict()
        m_keys = list(substrate_0.keys())
        m_vals = np.array([substrate_0[k] for k in m_keys])
        y0 = np.zeros(len(m_vals))

        vmax_vals = pd.DataFrame(
            np.full((len(renamed_models), len(m_keys)), V_MAX_ALL),
            columns=m_keys,
            index=[str(mdl_idx) for mdl_idx in range(len(renamed_models))],
        )
        km_vals = pd.DataFrame(
            np.full((len(renamed_models), len(m_keys)), K_M_ALL),
            columns=m_keys,
            index=[str(mdl_idx) for mdl_idx in range(len(renamed_models))],
        )

        sol, dfba_solve_time = benchmark_dfba(
            renamed_models, vmax_vals, km_vals, m_keys, y0, community, rel_abund, SOLVER, EARLY_STOP_WINDOW_SIZE, EARLY_STOP_TOL
        )
        
        # get final fluxes at last time
        dfba_final_fluxes = org_flux.copy(deep=True)
        dfba_final_fluxes.iloc[:] = 0.0 # reset to zero
        final_conc = sol.y[:, -1]
        conc_map = dict(zip(m_keys, final_conc))
        for mdl_idx, model in enumerate(renamed_models):
            # Update lower bounds based on Michaelis-Menten kinetics
            for ex_id in m_keys:
                if ex_id in model.reactions and "bio" not in ex_id:
                    v_max = vmax_vals.loc[str(mdl_idx), ex_id]
                    k_m = km_vals.loc[str(mdl_idx), ex_id]
                    c = conc_map[ex_id]

                    # if c < 1e-6:
                    #     c = 0.0  # avoid very small concentrations causing numerical errors

                    uptake_limit = (v_max * c) / (k_m + c)  # (mmol/(gi*hr))
                    model.reactions.get_by_id(ex_id).lower_bound = -uptake_limit

            solution = cb.flux_analysis.parsimonious.pfba(model)
    
            # Add fluxes to the net rate of change (weighted by clamped relative abundance)
            if solution.status == "optimal":
                dfba_final_fluxes.loc[mdl_idx, list(solution.fluxes.index)] = solution.fluxes * rel_abund[mdl_idx].item()

            print(f"Model {mdl_idx} final fluxes: {solution.fluxes.to_dict()}")

        experiment_logs.update(
            dfba_solve_time=float(dfba_solve_time),
            dfba_pfba_calls=int(n_dfba_pfba_calls),
            dfba_success=bool(sol.success),
            dfba_message=str(sol.message),
            dfba_termination_t=float(sol.t[-1]) if len(sol.t) > 0 else None,
            gifba_fluxes=org_flux.to_dict(),
            dfba_fluxes=dfba_final_fluxes.to_dict(),
        )
        save_checkpoint(status="dfba_complete")



    except Exception as exc:
        experiment_logs["error"] = f"{type(exc).__name__}: {exc}"
        experiment_logs["traceback"] = traceback.format_exc()
        save_checkpoint(status="failed")
        raise


if __name__ == "__main__":
    main()