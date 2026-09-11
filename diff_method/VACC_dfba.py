import gifba
from scipy.integrate import solve_ivp
import numpy as np
import cobra as cb
import pandas as pd
from IPython.display import clear_output
import copy
import matplotlib.pyplot as plt
from cobra import Model
import os
import time
from tqdm import tqdm
import json


array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
job_id = f"{array_job_id}_{task_id}" if array_job_id else "local_run"
output_dir = "benchmark_results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{job_id}.json")
experiment_logs = dict({})


agora_dir_base = "/gpfs2/scratch/rdsiegel/agora2_shared"
# agora_dirs = ['AGORA2_annotatedMat_A_C', 'AGORA2_annotatedMat_D_F', 'AGORA2_annotatedMat_G_P', 'AGORA2_annotatedMat_R_Y']
EURO_MEDIA_FILE = "/users/r/d/rdsiegel/giFBA/dFBA_runs/euro_diet.tsv"
N_TOT_AGORA = 7302
LOG_EVERY_N_CALLS = 1000
N_models_list = [2, 6, 10]
V_MAX_LIST = [100, 500, 1000]  # (mmol/(gi*hr)) Maximum uptake rate for all substrates
KM_LIST = [0.1, 0.5, 1.0]  # (mmol) Michaelis-Menten constant for all substrates
solver_list = ['RK45', 'BDF']

N = np.random.choice(N_models_list)  # Number of models to randomly select for the simulation
V_MAX_ALL = np.random.choice(V_MAX_LIST)  # (mmol/(gi*hr)) Maximum uptake rate for all substrates
K_M_ALL = np.random.choice(KM_LIST)  # (mmol) Michaelis-Menten constant
SOLVER = np.random.choice(solver_list)  # ODE solver method

n_dfba_fba_calls = 0

# Central log dictionary
experiment_logs["N_models"] = int(N)
experiment_logs["model_names"] = None
experiment_logs["sum_N_rxns"] = None
experiment_logs["sum_N_mets"] = None
experiment_logs["vmax"] = float(V_MAX_ALL)
experiment_logs["km"] = float(K_M_ALL)
experiment_logs["solver_method"] = SOLVER
experiment_logs["gifba_solve_time"] = None
experiment_logs["solve_ivp_time"] = None
experiment_logs["gifba_fba_calls"] = None
experiment_logs["dfba_fba_calls"] = None
experiment_logs["dfba_status"] = "Not Started"

def save_checkpoint(status=None):
    """Saves current state of experiment_logs to JSON."""
    if status:
        experiment_logs["status"] = status
    experiment_logs["dfba_fba_calls"] = int(n_dfba_fba_calls)
    
    # Save safely to disk
    with open(output_file, "w") as f:
        json.dump(experiment_logs, f, indent=4)

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
            # We cast to list() to avoid modifying a dictionary while iterating over it
            for metabolite in list(biomass_rxn.metabolites):
                metabolite.id = f"{metabolite.id}{suffix}"
                # Optional: update the metabolite name for clarity
                metabolite.name = f"{metabolite.name} (Species {idx + 1})"
            
            # 3. Rename the reaction itself
            biomass_rxn.id = f"{old_biomass_id}{suffix}"
            biomass_rxn.name = f"{biomass_rxn.name} (Species {idx + 1})"
            
    return updated_models


def dsdt(t, S, models, vmax, km, media_keys):
    global n_dfba_fba_calls
    # Initialize the rate of change for each metabolite to 0
    dsdt_rates = np.zeros(len(S))
    
    # Create a mapping for quick lookup: {metabolite_id: current_concentration}
    conc_map = dict(zip(media_keys, S))
    
    for mdl_idx, model in enumerate(models):
        # 1. Update lower bounds based on Michaelis-Menten kinetics
        for ex_id in media_keys:
            if ex_id in model.reactions and "bio" not in ex_id:
                v_max = vmax.loc[str(mdl_idx), ex_id]
                k_m = km.loc[str(mdl_idx), ex_id]
                c = conc_map[ex_id]
                
                if c < 1e-6:
                    c = 0.0  # Avoid very small concentrations causing numerical errors

                # Michaelis-Menten uptake limit 
                uptake_limit = (v_max * c) / (k_m + c) # (mmol/(gi*hr))
                
                # In COBRA, uptake is negative, so lower_bound is -uptake_limit
                model.reactions.get_by_id(ex_id).lower_bound = -uptake_limit

        # 2. Run FBA
        solution = model.optimize()
        n_dfba_fba_calls += 1  # Increment the global FBA call counter

        # 3. Add fluxes to the net rate of change (weighted by clamped relative abundance)
        if solution.status == 'optimal':
            for i, ex_id in enumerate(media_keys):
                flux_per_hr = solution.fluxes.get(ex_id, 0)
                
                # We add the RATE of change to our array. solve_ivp handles the integration over time.
                dsdt_rates[i] += flux_per_hr * rel_abund[mdl_idx]

    # 4. Add media fluxes to rate of change
    for i, ex_id in enumerate(media_keys):
        media_flux = community.env_fluxes.loc[0, ex_id]
        dsdt_rates[i] += media_flux 

    # Semi-frequent periodic logging
        if n_dfba_fba_calls % LOG_EVERY_N_CALLS == 0:
            save_checkpoint(f"dfba_in_progress_{n_dfba_fba_calls}")

    return dsdt_rates


# Optimized Tracker (progress bar, dsdt caller, early stopping)
class dFBATracker:
    def __init__(self, t_start, t_end, tol=1e-5):
        # self.pbar = tqdm(total=t_end - t_start, desc="Simulating ODE")
        self.highest_t = t_start
        self.tol = tol
        
        # State caching to prevent re-evaluating FBA
        self.prev_t = None
        self.prev_dsdt = None
        self.curr_t = None
        self.curr_dsdt = None

    def ode_wrapper(self, t, S, *args):
        # Calculate rates once per solver step
        rates = dsdt(t, S, *args)
        
        # Update progress bar only on forward progress
        # if t > self.highest_t:
        #     self.pbar.update(t - self.highest_t)
        #     self.highest_t = t
        
        # Cache the history. We only save a "previous" state if dt is large enough.
        # This ignores the tiny perturbation steps BDF takes for Jacobian estimation.
        if self.curr_t is None or abs(t - self.curr_t) > 1e-4:
            self.prev_t = self.curr_t
            self.prev_dsdt = self.curr_dsdt
            
        self.curr_t = t
        self.curr_dsdt = rates
        
        return rates

    def check_stop_condition(self, t, S, *args):
        # Prevent stopping before we have enough history
        if self.prev_t is None or self.prev_dsdt is None:
            return 1.0  
            
        # If the solver is evaluating the exact time we just cached, reuse it! (Saves FBA calculation)
        if t == self.curr_t:
            current_rates = self.curr_dsdt
        else:
            current_rates = dsdt(t, S, *args)
            
        dt = t - self.prev_t
        if abs(dt) < 1e-6:
            return 1.0
            
        max_accel = np.max(np.abs(current_rates - self.prev_dsdt)) / abs(dt)
        
        # Event triggers when this drops below tolerance (direction=-1)
        return max_accel - self.tol




# AGORA models

# randomly select N models across all directories weighting by the number of files in each directory
selected_models = []
selected_models_files = []
for i in range(N):    
     # randomly select a model from the selected directory
    model_files = [name for name in os.listdir(agora_dir_base) if os.path.isfile(os.path.join(agora_dir_base, name))]
    model_file = np.random.choice(model_files)
    
#     # load the model
    model_path = os.path.join(agora_dir_base, model_file)
    model = cb.io.load_matlab_model(model_path)
    
    selected_models.append(model)
    selected_models_files.append(model_file)

# update models for unique biomass reactions
renamed_models = rename_community_biomass(selected_models, old_biomass_id="EX_biomass(e)")

# parse community details (N rxns, N mets, etc.)
N_rxns = sum([len(model.reactions) for model in renamed_models])
N_mets = sum([len(model.metabolites) for model in renamed_models])


experiment_logs["model_names"] = selected_models_files
experiment_logs["sum_N_rxns"] = N_rxns
experiment_logs["sum_N_mets"] = N_mets

save_checkpoint(status="models_loaded")

# assigne equal relative abundance to each model
rel_abund = np.ones(N) / N

# define medium
euro_media = pd.read_csv(EURO_MEDIA_FILE, sep="\t", header=0, index_col=0).to_dict()
euro_media = euro_media["Flux Value"]
euro_media = {ex.replace("[e]", "(e)"): -flux for ex, flux in euro_media.items()}

# create a gifba community object and add any essential exchanges
community = gifba.gifbaObject(renamed_models, [euro_media, 0.1], # minimal media
                            rel_abund=rel_abund)




# Run gifba
gifba_start_time = time.time()
env_flux, org_flux = community.run_gifba(n_iterations=100, method="pfba", verbose=False)
gifba_end_time = time.time()
gifba_solve_time = gifba_end_time - gifba_start_time
gifba_fba_calls = community.simulation_count


experiment_logs["gifba_solve_time"] = gifba_solve_time
experiment_logs["gifba_fba_calls"] = gifba_fba_calls
save_checkpoint(status="gifba_complete")

# equal rel_abund
rel_abund = community.rel_abund.copy()

# media from gifba
substrate_0 = community.env_fluxes.loc[0, :].to_dict()

# kinetic params
vmax_vals = pd.DataFrame(np.full((len(renamed_models), len(substrate_0.keys())), V_MAX_ALL),
                              columns=list(substrate_0.keys()),
                              index=[str(mdl_idx) for mdl_idx in range(len(renamed_models))])
km_vals = pd.DataFrame(np.full((len(renamed_models), len(substrate_0.keys())), K_M_ALL),
                              columns=list(substrate_0.keys()),
                              index=[str(mdl_idx) for mdl_idx in range(len(renamed_models))])

# Preparing cleaner inputs for solve_ivp
m_keys = list(substrate_0.keys()) 
m_vals = np.array([substrate_0[k] for k in m_keys]) 
y0 = np.zeros(len(m_vals))  

# solve_ivp time
t_start, t_end = 0, 100
t_span = (t_start, t_end)

# Initialize the tracker
tracker = dFBATracker(t_start=t_start, t_end=t_end, tol=1e-5)

# Create a standalone wrapper function for the event
def steady_state_event(t, S, *args):
    return tracker.check_stop_condition(t, S, *args)

# Configure solve_ivp's event system on the standalone function
steady_state_event.terminal = True
steady_state_event.direction = -1  # Trigger only when dropping from > tol to < tol

dfba_time_start = time.time()
# Run the solver
sol = solve_ivp(
    tracker.ode_wrapper,  
    t_span=t_span,
    y0=y0,
    args=(copy.deepcopy(renamed_models), vmax_vals, km_vals, m_keys),
    method=SOLVER, 
    events=[steady_state_event]  
)
dfba_time_end = time.time()
dfba_solve_time = dfba_time_end - dfba_time_start
# print(f"ODE solver run time: {time_end - time_start:.2f} seconds")
# print("Success:", sol.success)
# print("Status:", sol.status)
# print("Message:", sol.message)
# print("Termination time (actual t_span):", sol.t[-1] if len(sol.t) > 0 else "No steps taken")

experiment_logs["N_models"] = int(N)
experiment_logs["model_names"] = selected_models_files
experiment_logs["sum_N_rxns"] = int(N_rxns)
experiment_logs["sum_N_mets"] = int(N_mets)
experiment_logs["vmax"] = float(V_MAX_ALL)
experiment_logs["km"] = float(K_M_ALL)
experiment_logs["solver_method"] = SOLVER
experiment_logs["gifba_solve_time"] = float(gifba_solve_time)
experiment_logs["solve_ivp_time"] = float(dfba_solve_time)
experiment_logs["gifba_fba_calls"] = int(gifba_fba_calls)
experiment_logs["dfba_fba_calls"] = int(n_dfba_fba_calls)

save_checkpoint(status="dfba_complete")
  

