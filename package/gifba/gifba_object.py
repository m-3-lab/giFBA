import cobra as cb
import numpy as np
import pandas as pd
from cobra.util.solver import linear_reaction_coefficients
from . import utils
from .config import GROWTH_MIN_OBJ, ROUND
from .summary import CommunitySummary
import numpy as np
from scipy.optimize import root_scalar

class gifbaObject:
    """_summary_

    Attributes:
        models: List[cobra.Model], (n_organisms_or_models, )
            A list of cobra.Model objects.
        media: Dict[str: float] 
            The media conditions for the models.
        rel_abund: np.ndarray[float], (n_organisms_or_models, 1)
            The relative abundance of the models, stored as a column vector. 
        id: str, (optional, default=None)
            An optional identifier for the giFBA analysis.
        size: int 
            The number of models in the community. Length of models list.
        objective_rxns: Dict[int: str]
            A dictionary mapping model indices to their objective reaction IDs.
        iters: int
            The number of iterations to run the giFBA analysis.
        method: str (optional, default="pfba")
            The method to use for flux balance analysis ("fba" or "pfba").
        early_stop: bool (optional, default=True)
            A boolean indicating whether to stop early if convergence is reached.
        v: bool (optional, default=False)
            A boolean indicating whether to print verbose output.
        m_vals: List[int], (2, ) (optional, default=[1,1])
            A list containing two integers that define the number of sample points and 
            start points for different runs per iteration. This variable is mainly 
            used for sampling via gifba_sampling, and should remain [1,1] for standard 
            giFBA.
        ex_to_met: Dict[str: str] 
            A dictionary mapping exchange reaction IDs to metabolite IDs.
        metid_to_name: Dict[str: str]
            A dictionary mapping metabolite IDs to their human readable names.
        exchange_metabolites: List[str]
            A list of all unique exchange metabolite IDs across the models. This is 
            a redundant variable, containing all values of ex_to_met.
        exchanges: List[str]
            A list of all unique exchange reaction IDs across the models.
        org_exs: List[str]
            A list of all unique exchange reaction IDs across the models.
        org_rxns: List[str]
            A list of all unique reaction IDs across the models.
        env_fluxes: pd.DataFrame, (n_iterations * m_vals[0] * m_vals[1] + 1, n_exchanges)
            A DataFrame storing the environmental fluxes for each iteration and run. Dataframe
            index is multi-indexed by iteration & run (giFBA drops run index, only necessary
            for sampling). Columns are unique exchange reaction IDs for the entire community.
        org_fluxes: pd.DataFrame, (n_iterations * m_vals[0] * m_vals[1] * n_orgs, n_reactions)
            A DataFrame storing the fluxes of all reactions for each model, iteration, and run. 
            Dataframe index is multi-indexed by model, iteration, and run (giFBA drops run index,
            only necessary for sampling). Columns are unique reaction IDs for the entire community.
        model_names: Dict[int: str]
            A dictionary mapping model indices to their names.
        summary: CommunitySummary
            A CommunitySummary object summarizing the results of the giFBA analysis, see 
            CommunitySummary for more details.

    Methods:
        __init__(self, models, media, rel_abund="equal", id=None):
            Initializes the gifbaObject with the given parameters.
        
        run_gifba(self, iters, method, early_stop=True, v=False):
            Runs the giFBA analysis for a specified number of iterations using the chosen method.
        
        create_vars(self, m_vals=[1,1]):
            Initializes variables for the community giFBA analysis and interpretation. This includes
            setting up DataFrames for environmental and organism fluxes, as well as storing model
            names and reaction mappings.

        update_media(self, iter):
            Updates the media conditions for each iteration based on the fluxes of the models. This 
            method wraps around the _flux_function and handles the media update logic.

        _flux_function(self, iter):
            Runs the flux function for each model in the community for the given iteration. This method
            wraps around the _set_env & _sim_fba methods and handles the overconsumption check.

        _set_env(self, iter, model_idx):
            Sets the exchange reactions of a model to match the environment fluxes for a given iteration and
            model index. This is mainly provided to ensure a cleaner wrapper function.

        _sim_fba(self, iter, model_idx):
            Runs Basic FBA or parsimonious FBA (pFBA) on a model and stores the resulting
            fluxes in the org_fluxes DataFrame. If the model's objective value is below a minimum
             threshold (entailing no growth), the fluxes are not updated (remain zero).

        _check_overconsumption(self, iter):
            Checks for over-consumption of environmental metabolites, scales down overconsumed reactions, and
            re-runs the flux function if necessary.
        
        summarize(self, iter_shown=None):
            Summarizes the results of the giFBA analysis in a CommunitySummary object, formatted to match
            a COBRApy model summary. Also formatting iteration information for a cytoscape-compatible
            node/edge table. 
    """
    def __init__(self, models, media, rel_abund="equal", step_size=1, 
                 **kwargs):
                 #id=None, sim_type="standard", debug=False, v=False, OC_Rounding=ROUND, OC_Method="optim"):
        self.models = utils.check_models(models)
        self.media = media
        self.media = utils.check_media(self)
        self.size = len(self.models)
        self.rel_abund = utils.check_rel_abund(rel_abund, self.size)
        self.step_size = step_size
        self.flow = None
        self.iters = None
        
        # simulation parameters with defaults
        self.threshold = kwargs.get("threshold", 1e-12)
        self.sim_type = kwargs.get("sim_type", "standard")
        self.OC_Rounding = kwargs.get("OC_Rounding", ROUND)
        self.OC_Method = kwargs.get("OC_Method", "optim")

        # optional user parameters
        self.id = kwargs.get("id", None)
        self.debug = kwargs.get("debug", False)
        self.v = kwargs.get("v", False)

        # get obj rxn ids
        model_obj_rxns = []
        for model in self.models:
            obj_rxn = linear_reaction_coefficients(model).keys()
            model_obj_rxns.extend([rxn.id for rxn in obj_rxn])
        self.objective_rxns = dict(zip(range(self.size), 
                                      model_obj_rxns))

    def run_additive_gifba(self, iters, method, **kwargs):
        """_summary_

        Args:
            iters (_type_): _description_
            method (_type_): _description_
            flow (float, optional): _description_. Defaults to 0.
            threshold (float, optional): _description_. Defaults to 1e-12.
            v (bool, optional): _description_. Defaults to False.

        Returns:
            env_fluxes: _description_
            org_fluxes: _description_
        """
        self.iters = utils.check_iters(iters)
        self.method = utils.check_method(method)
        self.threshold = kwargs.get("threshold", 1e-12)
        self.v = kwargs.get("v", False)
        self.debug = kwargs.get("debug", False)
        self.flow = kwargs.get("flow", 0)
        self.sim_type = "additive"

        # create variables
        self.create_vars()

        # run iterations
        for iter in range(self.iters):
            self.iter= iter
            print("\nIteration:", iter)

            # update media for the iteration
            self._is_rerun = False # reset re-run flag for overconsumption
            self._update_media(iter)# maybe change name

            # check early stopping condition
            if self.v: print("Checking Convergence...")
            env_tmp = self.env_fluxes
            delta = env_tmp.loc[iter+1, 0] - env_tmp.loc[iter, 0]
            if np.all(np.abs(delta) < self.threshold):
                # copy last iter to all future iters
                self.env_fluxes.loc[(slice(iter+1, None),0), :] = self.env_fluxes.loc[(iter,0), :].values
                
                # copy last iter to all future iters
                vals = self.org_fluxes.loc[(slice(None),iter,0), :].values
                n_future = self.iters - (iter + 1)
                if n_future > 0:
                    tiled = np.tile(vals, (n_future, 1))  # shape (n_future * n_models, n_rxns)
                    self.org_fluxes.iloc[-(n_future*self.size):] = tiled


                if self.v: print("Converged at iteration", iter)
                self.iter_converged = iter
                break
        
        if self.iter_converged is None:
            self.iter_converged = self.iters - 1
        print("Total iterations run:", self.iter_converged)

        # drop run col
        self.org_fluxes = self.org_fluxes.droplevel("Run")
        self.env_fluxes = self.env_fluxes.droplevel("Run")
        
        # cumulative sum across iterations
        # self.org_fluxes = self.org_fluxes.groupby(level=["Model"]).cumsum()

        # normalize 
        # self.org_fluxes = self.org_fluxes.apply(
        #     lambda col: col / (1 + self.org_fluxes.index.get_level_values('Iteration') * self.flow),
        #     axis=0)
        
        # account for flow factor (remove flow for all iters after 0)
        # self.env_fluxes.loc[(slice(1, None)), :] -= self.flow * self.env_fluxes.loc[0, :].values
        # self.env_fluxes.loc[(slice(1, None)), :] *= 1/ (1-self.flow)

        # cumulative sum across iterations
        self.org_fluxes = self.org_fluxes.groupby(level=["Model"]).cumsum()

        # return results for total fluxes
        return self.env_fluxes.iloc[-1], self.org_fluxes.iloc[-self.size:]

    def create_vars(self, m_vals=[1,1]):
        """Initialize variables for giFBA.
        This function sets up the optimization variables for the giFBA analysis.

        """
        # default initialization of vars
        self.iters = 1 if self.iters is None else self.iters
        self.iter_converged = None
        self.periodicity = None
        self.m_vals = m_vals # default to [1,1] for community giFBA, can be set to [n, m] for sampling via giFBA_sampling m_vals arg
        # get list of all unique rxns and exchanges
        self.ex_to_met = {}
        self.metid_to_name = {}
        self.exchange_metabolites = []
        self.exchanges = []
        self.org_exs = set()
        self.org_rxns = set()
        self.biomass_exs = set()

        # rxns/echanges/boundary mets per model
        for model in self.models:
            exs_set = set(model.exchanges.list_attr("id"))
            self.org_exs = self.org_exs | exs_set # exchanges

            rxns_set = set(model.reactions.list_attr("id"))
            self.org_rxns = self.org_rxns | rxns_set # reactions

            for rxn in model.exchanges:
                mets = list(rxn.metabolites.keys())
                if len(mets) == 1:
                    self.ex_to_met[rxn.id] = mets[0].id if pd.notnull(mets[0].id) else rxn.id
                    self.metid_to_name[mets[0].id] = mets[0].name if pd.notnull(mets[0].name) else mets[0].id
                    self.exchange_metabolites.extend(mets)
                    self.exchanges.append(rxn.id)

                    # add biomass exs to separate set
                    if "biomass" in list(rxn.metabolites.keys())[0].id.lower():
                        self.biomass_exs = self.biomass_exs | {rxn.id}
        
        # convert to attribute lists
        self.org_exs = list(self.org_exs)
        self.org_rxns = list(self.org_rxns)
        self.exchange_metabolites = list(set(self.exchange_metabolites))
        self.exchanges = list(set(self.exchanges))
        self.biomass_exs = list(self.biomass_exs)

        # initialize env
        self.media = utils.check_media(self)
        rows = (self.iters) * self.m_vals[0] * self.m_vals[1] + 1 # add one iteration for initial env
        cols = len(self.org_exs)
        self.env_fluxes = np.zeros((rows, cols))
        env0_masks = [np.array(self.org_exs) == rxn_id for rxn_id in list(self.media.keys())]
        for flux_idx, flux in enumerate(list(self.media.values())):
            self.env_fluxes[0][env0_masks[flux_idx]] = -flux

        #set columns for multi-indexing
        iters_col = np.repeat(np.arange(1, self.iters+1), self.m_vals[0] * self.m_vals[1]) 
        run_col = np.tile(np.arange(self.m_vals[0] * self.m_vals[1]), self.iters)
        iters_col = np.insert(iters_col, 0, 0) # add 0th iteration
        run_col = np.insert(run_col, 0, 0) # add 0th run 
        multi_idx = [iters_col , run_col]
        self.env_fluxes = pd.DataFrame(self.env_fluxes, columns=self.org_exs, index=multi_idx) # convert to interprettable df
        self.env_fluxes.index.names = ["Iteration", "Run"]

        # initialize org_fluxes
        rows = self.iters * self.m_vals[0] * self.m_vals[1] * len(self.models)
        cols = len(self.org_rxns)
        self.org_fluxes = np.zeros((rows, cols)) # pfba will drop run column
        
        # create unique multi-index for 
        models_col = np.tile(np.arange(self.size), self.iters * self.m_vals[0] * self.m_vals[1]) 
        iters_col = np.repeat(np.arange(self.iters), self.m_vals[0] * self.m_vals[1] * self.size) 
        run_col = np.tile(np.repeat(np.arange(self.m_vals[0] * self.m_vals[1]), self.size), self.iters) 
        multi_idx = [models_col, iters_col , run_col]
        self.org_fluxes = pd.DataFrame(self.org_fluxes, columns=self.org_rxns, index=multi_idx)	# convert to interprettable df
        self.org_fluxes.index.names = ["Model", "Iteration", "Run"]

 

        # store model names
        self.model_names = {model_idx: model.name for model_idx, model in enumerate(self.models)}

        return
        

    def _update_media(self, iter):
        """
        Update the media (f_n,j) for each iteration
        f_{n+1, j} =(1-flow)( f_{n,j} + sum(V_{n,i,j}) ) + flow*(f_{0,j})
        """
        # run organism flux function
        self._flux_function(iter)

        # update media: f_n+1 = f_n - sum(v_nij)
        env_tmp = self.env_fluxes.loc[iter, 0][:].to_numpy().reshape(-1, 1)   # (row, col) = (n_ex, 1)     # uptake = positive
        run_exs = self.org_fluxes.loc[:, iter, 0][self.env_fluxes.columns].to_numpy().T # (row, col) = (n_ex, n_org) # uptake = negative flux
        sum_org_flux = run_exs.sum(axis=1).reshape(-1, 1) # (n_ex, n_org) -> (n_ex, ) sum across orgs

        if self.sim_type == "additive":
            self.env_fluxes.loc[iter+1, 0] = ((1-self.flow) * (env_tmp +  sum_org_flux) + (self.flow * self.env_fluxes.loc[0,0].to_numpy().reshape(-1, 1))).flatten().round(ROUND) # (n_ex, 1) + (n_ex, 1) -> (n_ex, 1)

        elif self.sim_type == "standard":
            # get init env for iter 0
            env_tmp = self.env_fluxes.loc[0, 0][:].to_numpy().reshape(-1, 1)

            # pull ex info for iter and set uptake to 0
            run_exs = self.org_fluxes.loc[:, iter, 0][self.env_fluxes.columns].to_numpy().T # (row, col) = (n_ex, n_org) # uptake = negative flux
            run_exs[run_exs < 0] = 0 # only secretion counts
            
            # sum org fluxes and media
            sum_org_flux = run_exs.sum(axis=1).reshape(-1, 1)
            self.env_fluxes.loc[iter+1, 0] = (env_tmp + sum_org_flux).flatten()#.round(ROUND) # (n_ex, 1) + (n_ex, 1) -> (n_ex, 1)
        return


    def _flux_function(self, iter):
        """
        run through flux function for organisms
        """
        # # define env bounds per organism for the current iteration
        if not(self._is_rerun): # if first run of iteration, just initialize scaled by rel abund only otherwise do nothing
            self._env_scaling_factors = np.ones((self.size, len(self.org_exs)))  # initialize update rate (used to scale ex flux bounds
            for model_idx in range(self.size):
                self._env_scaling_factors[model_idx, :] = self._env_scaling_factors[model_idx, :] / self.rel_abund[model_idx]

        # simulate each organism
        for model_idx in range(self.size):
            # if self.v: print(" Simulating model:", model_idx+1, " of ", self.size)
            # set media
            self._set_env(iter, model_idx)

            # simulate each org
            self._sim_fba(iter, model_idx)

        # check over consumption
        self._check_overconsumption(iter)

        # once all orgs have been simulated without overconsumption, update internal rxns
        if self.sim_type == "additive" and not(self._is_rerun):
            self._update_internal_reactions(iter)

        return

    def _set_env(self, iter, model_idx):
        """
        Function to set the exchange reactions of a model to match the environment fluxes
        for a given iteration and run. This is mainly provided to ensure a cleaner wrapper function.
        """
        for ex in self.models[model_idx].exchanges:
            mask = np.array(self.org_exs) == ex.id
            if mask.any():  # Check if the exchange reaction exists in org_exs
                ex.lower_bound = -self._env_scaling_factors[model_idx, mask] * self.env_fluxes.loc[iter, 0][ex.id]
       
        return

    def _sim_fba(self, iter, model_idx):
        """General function to run parsimonious FBA (pFBA) on a model and store the results.
        This function runs pFBA on a given model, checks if the solution is above a minimum growth objective,
        and stores the resulting fluxes in the provided DataFrame.
        """
        # run pFBA
        sol1 = self.models[model_idx].slim_optimize()
        
        if self.debug:
            if model_idx == 0:
                print("#"*45)
                print("Run Info")
            print(f"Objective value (model {model_idx}): {sol1}")

        if sol1 > GROWTH_MIN_OBJ:
            if self.method == "pfba":
                sol = cb.flux_analysis.parsimonious.pfba(self.models[model_idx])
            elif self.method == "fba":
                sol = self.models[model_idx].optimize()
            
            self.org_fluxes.loc[(model_idx, iter, 0), list(sol.fluxes.index)] = self.rel_abund[model_idx] * sol.fluxes.values 
        # do nothing otherwise - already initiated as zeros!
        return
    
    def _check_overconsumption(self, iter):
        """
        Check over-consumption of env. mets. If over-consumption occurs, 
        re-run flux function (recursive subroutine)
        """
        #pull iter info and establish array shapes
        env_tmp = self.env_fluxes.loc[iter, 0][:].to_numpy().reshape(-1, 1)   # (row, col) = (n_ex, 1)     # uptake = positive
        run_exs = self.org_fluxes.loc[:, iter, 0][self.env_fluxes.columns].to_numpy().T # (row, col) = (n_ex, n_org) # uptake = negative flux

        # get org fluxes
        total_org_flux = run_exs.sum(axis=1).reshape(-1, 1) # (n_ex, n_org) -> (n_ex, 1) sum across orgs

        # check if environment fluxes are under-saturated
        is_overconsumed = np.zeros_like(total_org_flux)
        with np.errstate(divide='ignore', invalid='ignore'): # ignore division by zero warnings
            is_overconsumed[np.abs(env_tmp) >= 1e-12] = -total_org_flux[np.abs(env_tmp) >= 1e-12].astype(np.float64) / env_tmp[np.abs(env_tmp) >= 1e-12].astype(np.float64) # only check non-zero env fluxes
        
        if self.debug:
            print("\nenv fluxes (mmol/(gT/hr)):")
            print(self.env_fluxes.loc[iter, 0].T)
            print("\norg fluxes (mmol/(gT/hr)):")
            print(self.org_fluxes.loc[:, iter, 0][self.env_fluxes.columns])
            print("#"*45)
            print()


        # check if iteration uses more flux than available in environment
        if not self._is_rerun:
            self._rerun_ct=0

        # initialize lists on first call fro newton method tracking
        if iter == 0 and not self._is_rerun:
            self._X_list = []
            self._OC_list = []
            self._rerun_list = []
            self._iter_list = []
            self._ex_over_dict = {ex: {"X_list": [], "OC_list": [], "iter_list": [], "rerun_list": []} for ex in self.env_fluxes.columns}
        
        # re-run flux if overconsumed, adjusting only the over-consumed reactions
        if is_overconsumed.max().round(self.OC_Rounding) > 1 or (self._rerun_ct !=0 and is_overconsumed.max().round(self.OC_Rounding) <1): # rounding avoids numerical issues with X being set to inf or nan
            if self.OC_Method == "optim":
                self._optim_method_x(iter, is_overconsumed, run_exs, env_tmp)
            if self.OC_Method == "newton":
                self._newton_method_x(iter, is_overconsumed, run_exs, env_tmp)
            self._is_rerun = True
            self._rerun_ct += 1
            self._flux_function(iter)
        
        return
    
    def _optim_method_x(self, iter, is_overconsumed, run_exs, env_tmp):
        """
        Finds a universal cap X for each metabolite to balance community consumption 
        with available media. Handles both over-consumption (pull down) and 
        under-consumption (push up).
        """
        ex_over = np.argmax(is_overconsumed) # index of flux causing over-consumed
        # Iterate through all metabolites in the media

        # Current total consumption factor (Total_Flux / Media)
        oc_factor = is_overconsumed[ex_over, 0]
        

        if self.v: print(self.env_fluxes.columns[ex_over], f"over-consumed by factor of {is_overconsumed.max():.12f} (rerun count: {self._rerun_ct})")


        # 1. Gather current individual fluxes (normalized by abundance)
        # run_exs is total weighted flux (a_i * v_i). We want internal flux v_i.
        rel_abund = self.rel_abund.flatten()
        v_ij_magnitudes = np.abs(run_exs[ex_over, :]) / rel_abund

        # 2. Define the community response function
        def residual(X):
            # Total = sum( abundance * min(Cap, Individual_Flux) )
            total_flux = np.sum(rel_abund * np.minimum(X, v_ij_magnitudes))
            return total_flux - self.env_fluxes.loc[(iter, 0), self.env_fluxes.columns[ex_over]] # residual = total_flux - media_flux (want to find X where residual = 0)

        # 3. Determine Search Brackets
        current_max_v = np.max(v_ij_magnitudes)
        
        if oc_factor > 1:
            # Overconsumption: Root is between 0 and current max
            low, high = 0, current_max_v
        else:
            # Underconsumption: Try to find a cap X > current flux to push uptake
            low = current_max_v
            high = current_max_v * 2
            
            
            # Expand 'high' until we find a bracket for underconsumption
            while residual(high) < 0 and high < 1e6:
                high *= 2

        # 4. Solve for the optimal cap X
        try:
            sol = root_scalar(residual, bracket=[low, high], method='brentq')
            X_opt = sol.root
        except (ValueError, RuntimeError):
            # Fallback to current best if root finding fails
            X_opt = high if oc_factor < 1 else low

        # 5. Apply the universal cap
        # scaling_factor * media = X_opt => scaling = X_opt / media
        self._env_scaling_factors[:, ex_over] = X_opt / env_tmp[ex_over, 0]
        
        if self.debug:
            print(f"  X = {X_opt:.6f}")

        # Flag for re-simulation
        self._is_rerun = True
        self._rerun_ct += 1
        return
    
    def _newton_method_x(self, iter, is_overconsumed, run_exs, env_tmp):
        # reset if different ex is overconsumed on re-run
        if self._is_rerun and is_overconsumed.max() != 1:
            ex_over = np.argmax(is_overconsumed) # index of flux causing over-consumed
            if ex_over != self._ex_over:
                self._ex_over_dict[self.env_fluxes.columns[ex_over]] = {}
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["X_list"] = []
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["OC_list"] = []
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["rerun_list"] = []
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["iter_list"] = []
                self._rerun_ct = 0

        if is_overconsumed.max() > 1 or (self._rerun_ct !=0 and is_overconsumed.max() <1):
            ex_over = np.argmax(is_overconsumed) # index of flux causing over-consumed
            if self._is_rerun and ex_over != self._ex_over:
                self._ex_over_dict[self.env_fluxes.columns[ex_over]] = {}
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["X_list"] = []
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["OC_list"] = []
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["rerun_list"] = []
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["iter_list"] = []
                self._rerun_ct = 0   

            if self.v: print(self.env_fluxes.columns[ex_over], f"over-consumed by factor of {is_overconsumed.max():.12f} (rerun count: {self._rerun_ct})")
            if self.debug: print("v"*45)

            # adjust only over-consumed bound
            x_denom = 0
            for model_idx in range(self.size):
                if self.env_fluxes.columns[ex_over] in self.models[model_idx].reactions:
                    lb_ij = self.models[model_idx].reactions.get_by_id(self.env_fluxes.columns[ex_over]).lower_bound
                    V_ij = run_exs[ex_over, model_idx]
                    a_i = self.rel_abund[model_idx]
                    x_denom += V_ij / lb_ij
                    
                    if self.debug:
                        print("Model idx", model_idx, "    (alpha =", a_i[0],")")
                        print(f"  big V: {V_ij: 3.6f}   mmol/(gT/hr)")
                        print(f"  lil v: {V_ij/a_i[0]: 3.6f}   mmol/(gi/hr)")
                        print(f"     lb: {lb_ij[0]: 3.6f}   mmol/(gi/hr)")
            
            if not self._is_rerun or (self._is_rerun and self._rerun_ct ==0):
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["X_list"].append(0)
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["OC_list"].append(0)
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["rerun_list"].append(-1)
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["iter_list"].append(iter)

                # assume n=1 uses this form
                x_n = env_tmp[ex_over, 0] / x_denom[0]
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["X_list"].append(x_n)

            if self._is_rerun and self._rerun_ct !=0:
                # just use one LB for the ex over if we have already re-run
                for model_idx in range(self.size):
                    if self.env_fluxes.columns[ex_over] in self.models[model_idx].reactions:
                        lb = self.models[model_idx].reactions.get_by_id(self.env_fluxes.columns[ex_over]).lower_bound
                        break
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["X_list"].append(-lb[0])
                

            self._ex_over_dict[self.env_fluxes.columns[ex_over]]["OC_list"].append(is_overconsumed[ex_over, 0])

            # infer next best X based on deg 1 Newton Method
            m = (self._ex_over_dict[self.env_fluxes.columns[ex_over]]["X_list"][-1] - self._ex_over_dict[self.env_fluxes.columns[ex_over]]["X_list"][-2]) / (self._ex_over_dict[self.env_fluxes.columns[ex_over]]["OC_list"][-1] - self._ex_over_dict[self.env_fluxes.columns[ex_over]]["OC_list"][-2])
            b = self._ex_over_dict[self.env_fluxes.columns[ex_over]]["X_list"][-1] - m * self._ex_over_dict[self.env_fluxes.columns[ex_over]]["OC_list"][-1]
            X_n_p_1 = m * 1 + b  # new env bound at OC = 1

            if self.debug:
                print(f"   X_n:    {self._ex_over_dict[self.env_fluxes.columns[ex_over]]['X_list'][-1]:>18.14f}")
                print(f"  OC_n:    {self._ex_over_dict[self.env_fluxes.columns[ex_over]]['OC_list'][-1]:>18.14f}")
                print(f" X_n-1:    {self._ex_over_dict[self.env_fluxes.columns[ex_over]]['X_list'][-2]:>18.14f}")
                print(f"OC_n-1:    {self._ex_over_dict[self.env_fluxes.columns[ex_over]]['OC_list'][-2]:>18.14f}")            
                print(f" X_n+1:    {X_n_p_1:>18.14f}")
                print("^"*45)

            # set new scaling factor for next run 
            # this is div by env bc gets re-multiplied in set_env
            self._env_scaling_factors[:, ex_over] = X_n_p_1 / env_tmp[ex_over, 0]

            # store for next run
            self._ex_over_dict[self.env_fluxes.columns[ex_over]]["rerun_list"].append(self._rerun_ct)
            self._ex_over_dict[self.env_fluxes.columns[ex_over]]["iter_list"].append(iter)
            self._ex_over = ex_over
            self._rerun_ct += 1
            self._is_rerun = True

        return

    def _update_internal_reactions(self, iter):
        """
        Update internal reactions based on total flux
        Only used for additive model (cumulative fluxes across iterations), 
        not for standard model (fluxes are per iteration and do not carry over, 
        so no need to update internal rxns based on cumulative flux)
        """
        for model_idx in range(self.size):
            for rxn in self.models[model_idx].reactions:
                if not(rxn in self.models[model_idx].exchanges):
                    # store previous bounds
                    lb_old = rxn.lower_bound
                    ub_old = rxn.upper_bound
                    
                    # change in flux for given iter
                    last_flux = self.org_fluxes.loc[(model_idx, iter, 0), rxn.id] / self.rel_abund[model_idx, 0]
                    
                    # update bounds
                    rxn.lower_bound = lb_old - last_flux
                    rxn.upper_bound = ub_old - last_flux
        return
    

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        return False

    def summarize(self, iter_shown=None):
        return CommunitySummary(self, iter_shown)

    def run_gifba(self, iters, method, threshold=1e-12, attractor_size=0.9, v=False, debug=False):
        """ After each iteration, add only the new fluxes, 
        and do not remove uptaken ones. If fluxes remains 
        the same, update the environment, otherwise- re-do this process """
        self.iters = utils.check_iters(iters)
        self.method = utils.check_method(method)
        self.threshold = threshold
        self.attractor_size = attractor_size
        self.v = v
        self.debug = debug # will print info on every iteration and re-run, so use with caution
        self.sim_type = "standard"
        

        # create storage variables
        self.create_vars()

        # run iterations
        for iter in range(self.iters):
            self.iter = iter
            if self.debug or self.v: print(f"\nIteration: {iter}")

            # update media for the iteration
            self._is_rerun = False # reset re-run flag for overconsumption
            self._update_media(iter)# maybe change name

            # check early stopping condition
            if (self.iter > 0) or (iter == self.iters - 1):
                if self.debug: print("Checking Convergence...")
                for per in range(1, iter+1):
                    # check if last (-1) and per+1 iteration from end are the same (accounting for rounding) 
                    env_delta = self.env_fluxes.iloc[iter].values - self.env_fluxes.iloc[iter-per].values
                    org_delta = self.org_fluxes.iloc[self.size*iter:self.size*(iter+1)].values - self.org_fluxes.iloc[self.size*(iter-per):self.size*(iter-per+1)].values
                    
                    if np.all(np.abs(org_delta) < self.threshold) and np.all(np.abs(env_delta) < self.threshold):
                        self.periodicity = per
                        self.iter_converged = iter
                        break

                if self.iter_converged is not None:
                    if self.v: print("Converged at iteration", iter)
                    break
                        
        # drop run col
        self.org_fluxes = self.org_fluxes.droplevel("Run")
        self.env_fluxes = self.env_fluxes.droplevel("Run")

        # copy converged rows to end of iterations after convergence (if applicable)
        if self.iter_converged is not None:
            for iters_copy in range(self.iter_converged, self.iters):
                # org fluxes
                vals = self.org_fluxes.iloc[self.size*(iters_copy-self.periodicity):self.size*(iters_copy-self.periodicity+1), :].values
                self.org_fluxes.iloc[self.size*iters_copy:self.size*(iters_copy+1), :] = vals

                # env fluxes
                self.env_fluxes.loc[iters_copy+1, :] = self.env_fluxes.loc[iters_copy+1-self.periodicity, :].values

        # check periodic/adjust
        env_final, self.org_final = self.average_periodicity()
        
        # return results for total fluxes
        return env_final, self.org_final
    
    def average_periodicity(self):
        """Calculate the average periodicity of the system based on the environmental fluxes."""
        # if no convergence, give warning and return average of all iterations
        if self.periodicity is None and self.iter_converged is None:
            self.periodicity = int(self.iters * self.attractor_size) # set periodicity to a percentage of total iters if no convergence, so at least some averaging is done

            print("Model did not converge or show periodicity within the iteration limit, results may be unreliable.")
            print(f"{self.periodicity} iterations ({self.attractor_size*100:.1f}%) will be used for flux calculations, but consider increasing the number of iterations or checking model setup.")
        
        if self.v: print("Model is periodic and average of the last", self.periodicity, "iterations will be used for flux calculations.")
        
        # calculate average for the period size
        env_flux_avg = self.env_fluxes.loc[(slice(self.iters - self.periodicity, self.iters -1)), :].mean()
        org_flux_avg = self.org_fluxes.iloc[-self.periodicity * self.size:].groupby(level="Model").mean()
        return env_flux_avg, org_flux_avg