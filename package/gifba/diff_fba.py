import numpy as np
import copy
import time
from scipy.integrate import solve_ivp
from tqdm import tqdm

def dsdt(t, S, models, vmax, km, media_keys, rel_abund, community):
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

    return dsdt_rates


# Optimized Tracker (progress bar, dsdt caller, early stopping)
class ChemostatTracker:
    def __init__(self, t_start, t_end, models, vmax, km, media_keys, rel_abund, community, tol=1e-5):
        self.pbar = tqdm(total=t_end - t_start, desc="Simulating ODE")
        self.highest_t = t_start
        self.tol = tol
        self.models = models
        self.vmax = vmax
        self.km = km
        self.media_keys = media_keys
        self.rel_abund = rel_abund
        self.community = community
        
        # State caching to prevent re-evaluating FBA
        self.prev_t = None
        self.prev_dsdt = None
        self.curr_t = None
        self.curr_dsdt = None

    def ode_wrapper(self, t, S, *args):
        # Calculate rates once per solver step using bound attributes
        rates = dsdt(t, S, self.models, self.vmax, self.km, self.media_keys, self.rel_abund, self.community)
        
        # Update progress bar only on forward progress
        if t > self.highest_t:
            self.pbar.update(t - self.highest_t)
            self.highest_t = t
        
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
            current_rates = dsdt(t, S, self.models, self.vmax, self.km, self.media_keys, self.rel_abund, self.community)
            
        dt = t - self.prev_t
        if abs(dt) < 1e-6:
            return 1.0
            
        max_accel = np.max(np.abs(current_rates - self.prev_dsdt)) / abs(dt)
        
        # Event triggers when this drops below tolerance (direction=-1)
        return max_accel - self.tol



def run_dfba(renamed_models, vmax_vals, km_vals, substrate_0, m_keys, community, rel_abund, t_span=(0, 100), tol=1e-5, method="RK45"):
    """
    Runs the dynamic FBA simulation using solve_ivp and the ChemostatTracker.
    """
    y0 = np.array([substrate_0[k] for k in m_keys]) 
    t_start, t_end = t_span

    # Initialize the tracker with required references
    tracker = ChemostatTracker(
        t_start=t_start, 
        t_end=t_end, 
        models=copy.deepcopy(renamed_models), 
        vmax=vmax_vals, 
        km=km_vals, 
        media_keys=m_keys, 
        rel_abund=rel_abund, 
        community=community, 
        tol=tol
    )

    # Create a standalone wrapper function for the event
    def steady_state_event(t, S, *args):
        return tracker.check_stop_condition(t, S, *args)

    # Configure solve_ivp's event system on the standalone function
    steady_state_event.terminal = True
    steady_state_event.direction = -1  # Trigger only when dropping from > tol to < tol

    time_start = time.time()
    
    # Run the solver
    sol = solve_ivp(
        tracker.ode_wrapper,  
        t_span=t_span,
        y0=y0,
        args=(), 
        method=method, 
        events=[steady_state_event]  
    )
    
    time_end = time.time()
    
    print(f"ODE solver run time: {time_end - time_start:.2f} seconds")
    print("Success:", sol.success)
    print("Status:", sol.status)
    print("Message:", sol.message)
    print("Termination time:", sol.t[-1] if len(sol.t) > 0 else "No steps taken")
    
    return sol