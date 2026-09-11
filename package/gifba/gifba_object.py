import cobra as cb
import numpy as np
import pandas as pd
from cobra.util.solver import linear_reaction_coefficients
from scipy.optimize import root_scalar
from typing import Literal
from . import utils
from .config import GROWTH_MIN_OBJ, ROUND
from .summary import CommunitySummary


class gifbaObject:

    def __init__(self, models, media, rel_abund="equal",
                 **kwargs):
        self.models = utils.check_models(models)
        self.media = media
        self.media = utils.check_media(self)
        self.num_models = len(self.models)
        self.rel_abund = utils.check_rel_abund(rel_abund, self.num_models)
        self.flow = None
        self.n_iterations = None
        self.simulation_count = 0
        
        # simulation parameters with defaults
        self.threshold = kwargs.get("threshold", 1e-12)
        self.oc_rounding = kwargs.get("oc_rounding", ROUND)
        self.oc_method = kwargs.get("oc_method", "optim")

        # optional user parameters
        self.community_id = kwargs.get("community_id", None)
        self.debug = kwargs.get("debug", False)
        self.verbose = kwargs.get("verbose", False)

        # get obj rxn ids
        model_obj_rxns = []
        for model in self.models:
            obj_rxn = linear_reaction_coefficients(model).keys()
            model_obj_rxns.extend([rxn.id for rxn in obj_rxn])
        self.objective_rxns = dict(zip(range(self.num_models), 
                                       model_obj_rxns))

    def run_gifba(self, 
                  n_iterations: int, 
                  method: Literal["pfba", "fba"], 
                  threshold: float | None = None, 
                  attractor_size: float | None = None, 
                  relaxation_ratio: float | None = None, 
                  fp_method: Literal["picard", "relaxation", "anderson"] | None = None, 
                  v: bool = False, 
                  debug: bool = False
                  ) -> tuple[pd.Series, pd.DataFrame]:
        """ 
        Run giFBA for a given number of iterations on the community of models with the given media.

        [Flow / Hierarchy]
        Call `create_vars()` to initialize storage variables for the simulation.
        Call `_update_media()` for each iteration to simulate growth at each discrete step and update 
            state variables in accordance with fixed point solver method. See `_update_media()` for 
            more details.
        Check fixed point convergence by comparing current iteration to all previous iterations to 
            identify fixed points/periodic behavior.
        Upon completion/convergence, call `average_periodicity()` to perform element-wise average of
            state variables in fixed point/period/attractor.

        Args:
            n_iterations (int): Number of iterations to run the simulation.
            method (str): The FBA method to use for the simulation - must be "pfba" (recommended) or 
                "fba".
            threshold (float, optional): The numerical threshold for convergence. Defaults to 1e-12 
                if None provided.
            attractor_size (float, optional): The size of the attractor (in percentage of number of 
                iterations) if convergence is not achieved. Defaults to 0.9 if None provided.
            relaxation_ratio (float, optional): Ratio value chosen for relaxation method used in fixed 
                point solver. Defaults to 1.0 if None provided.
            fp_method (str, optional): The fixed point solver method to use for the simulation - must 
                be "picard", "relaxation" (recommended), or "anderson". Defaults to "picard" if None 
                provided. If "picard" is chosen, the relaxation_ratio parameter will be overridden to 1.0.
            v (bool, optional): Toggle verbose output. True will provide printed outputs for current 
                iteration and any re-runs from overconsumption adjustment. Defaults to False.
            debug (bool, optional): Development mode. True will provide additional debugging 
                information detailing the simulation process (media & org. fluxes at each step, 
                overconsumption ratio, overconsumption adjustment). Not recommended for real-world 
                community models. Defaults to False.
        
        State Modified (Side Effects):
            self.env_fluxes (pd.DataFrame): Full environmental fluxes DataFrame for all iterations and 
                runs. Multi-indexed by iteration and run, with columns as unique exchange reaction IDs 
                for the entire community (size=(n_iterations, n_exchanges)). (runs currently unused and 
                part of future development). Units in ( mmol/(gT * hr) ).
            self.org_fluxes (pd.DataFrame): Full organism fluxes DataFrame for all iterations and runs. 
                Multi-indexed by model, iteration, and run, with columns as unique (internal and 
                exchange) reaction IDs for the entire community 
                (size=(n_iterations*n_models, n_reactions)). (runs currently unused and part of future 
                development). Units in ( mmol/(gT * hr) ).
            self.periodicity (int): Period of the system if convergence is achieved. If convergence is 
                not achieved, this will be None.
            self.iter_converged (int): Iteration at which convergence is achieved. If convergence is 
                not achieved, this will be None.

        Returns:
            tuple[pd.Series, pd.DataFrame]: A tuple containing two pandas objects for simulation 
                steady-state fluxes:
                - pd.Series: Media/Environment fluxes ($f_{n,j}$) at steady state in units of 
                    ( mmol/(gT * hr) ). Size=(n_exchanges,). Returns fixed points, average of periodic 
                    fixed points, or average of last <attractor_size> * <iterations> if convergence is 
                    not achieved.
                - pd.DataFrame: Per-organism fluxes ($V_{i,j}$) at steady state in units of 
                    ( mmol/(gT * hr) ). Size=(n_models, n_reactions). Returns fixed points, average of 
                    periodic fixed points, or average of last <attractor_size> * <iterations> if 
                    convergence is not achieved.
        
        Calls:
            - `self.create_vars()`
            - `self._update_media()`
            - `self.average_periodicity()`
        """
        self.n_iterations = utils.check_n_iterations(n_iterations)
        self.method = utils.check_method(method)
        self.threshold = self.threshold if threshold is None else threshold
        self.attractor_size = 0.9 if attractor_size is None else attractor_size
        self.relaxation_ratio = 1.0 if relaxation_ratio is None or fp_method == "picard" else relaxation_ratio
        self.fp_method = "relaxation" if fp_method is None or fp_method == "picard" else fp_method
        self.verbose = v
        self.debug = debug # will print info on every iteration and re-run, so use with caution
        
        # create storage variables
        self.create_vars()
        self.simulation_count = 0

        # run iterations
        for iteration in range(self.n_iterations):
            self.current_iteration = iteration
            if self.debug or self.verbose: print(f"\nIteration: {iteration}")

            # update media for the iteration
            self._is_rerun = False # reset re-run flag for overconsumption
            self._update_media(iteration)

            # check early stopping condition
            if (self.current_iteration > 0) or (iteration == self.n_iterations - 1):
                if self.debug: print("Checking Convergence...")
                for lag in range(1, iteration+1):
                    # check if last (-1) and lag+1 iteration from end are the same (accounting for rounding) 
                    env_delta = self.env_fluxes.iloc[iteration].values - self.env_fluxes.iloc[iteration-lag].values
                    org_delta = self.org_fluxes.iloc[self.num_models*iteration:self.num_models*(iteration+1)].values - self.org_fluxes.iloc[self.num_models*(iteration-lag):self.num_models*(iteration-lag+1)].values
                    
                    if np.all(np.abs(org_delta) < self.threshold) and np.all(np.abs(env_delta) < self.threshold):
                        self.periodicity = lag
                        self.iter_converged = iteration
                        break

                if self.iter_converged is not None:
                    if self.verbose: print("Converged at iteration", iteration)
                    break
                        
        # drop run col
        self.org_fluxes = self.org_fluxes.droplevel("Run")
        self.env_fluxes = self.env_fluxes.droplevel("Run")

        # copy converged rows to end of iterations after convergence (if applicable)
        if self.iter_converged is not None:
            for iters_copy in range(self.iter_converged, self.n_iterations):
                # org fluxes
                vals = self.org_fluxes.iloc[self.num_models*(iters_copy-self.periodicity):self.num_models*(iters_copy-self.periodicity+1), :].values
                self.org_fluxes.iloc[self.num_models*iters_copy:self.num_models*(iters_copy+1), :] = vals

                # env fluxes
                self.env_fluxes.loc[iters_copy+1, :] = self.env_fluxes.loc[iters_copy+1-self.periodicity, :].values

        # check periodic/adjust
        env_final, self.org_final = self.average_periodicity()
        
        # return results for total fluxes
        return env_final, self.org_final

    def create_vars(self, m_vals=[1,1]):
        """ 
        Initialize storage for state variables, set initial media, and extract pertinent community 
        details.

        [Flow / Hierarchy]
        Setup method. Called at the beginning of the simulation before the iterative fixed-point solver 
        begins.

        Args:
            m_vals (list[int], optional): Currently unused; reserved for future package development. 
                Defaults to [1,1] representing each iteration uses 1 input run to modify 1 output run, 
                per iteration. Standard giFBA currently only uses 1 run per iteration.
        
        State Inputs (Attributes Used):
            self.media (dict[str, float]): The baseline media conditions for the community. Keys are 
                exchange IDs and values are fluxes (which must be negative).
            self.models (list[cb.Model]): A list of all Cobra models present in the community.
            self.n_iterations (int): The total number of iterations to run for the simulation.
            self.num_models (int): The total number of models in the community.

        State Modified (Side Effects):
            self.env_fluxes (pd.DataFrame): Full environmental fluxes DataFrame for all iterations and 
                runs. Multi-indexed by iteration and run, with columns as unique exchange reaction IDs 
                for the entire community. Size is (n_iterations, n_exchanges). Units are mmol/(gT * hr).
            self.org_fluxes (pd.DataFrame): Full organism fluxes DataFrame for all iterations and runs. 
                Multi-indexed by model, iteration, and run, with columns as unique (internal and 
                exchange) reaction IDs for the entire community. 
                Size is (n_iterations * n_models, n_reactions). Units are mmol/(gT * hr).
            self.exchange_to_metabolite_id (dict[str, str]): Mapping of exchange reaction IDs to their corresponding 
                metabolite IDs.
            self.metabolite_id_to_name (dict[str, str]): Mapping of metabolite IDs to their human-readable names 
                (or exchange reaction IDs).
            self.exchange_metabolites (list[cb.Metabolite]): A list of all Cobra Metabolite objects 
                across the community.
            self.exchanges (list[str]): A list of all unique exchange reaction IDs in the community.
            self.exchange_ids (list[str]): A list of organism-specific exchange reaction IDs.
            self.biomass_exchange_ids (list[str]): A list of all unique biomass exchange reaction IDs in the 
                community.
            self.model_names (dict[int, str]): A mapping of the community model index to its string 
                model ID.
        """
        # default initialization of vars
        self.n_iterations = 1 if self.n_iterations is None else self.n_iterations
        self.iter_converged = None
        self.periodicity = None
        self.m_vals = m_vals # default to [1,1] for community giFBA, can be set to [n, m] for sampling via giFBA_sampling m_vals arg
        # get list of all unique rxns and exchanges
        self.exchange_to_metabolite_id = {}
        self.metabolite_id_to_name = {}
        self.exchange_metabolites = []
        self.exchanges = []
        self.exchange_ids = set()
        self.reaction_ids = set()
        self.biomass_exchange_ids = set()

        # rxns/echanges/boundary mets per model
        for model in self.models:
            exchange_id_set = set(model.exchanges.list_attr("id"))
            self.exchange_ids = self.exchange_ids | exchange_id_set # exchanges

            reaction_id_set = set(model.reactions.list_attr("id"))
            self.reaction_ids = self.reaction_ids | reaction_id_set # reactions

            for rxn in model.exchanges:
                mets = list(rxn.metabolites.keys())
                if len(mets) == 1:
                    self.exchange_to_metabolite_id[rxn.id] = mets[0].id if pd.notnull(mets[0].id) else rxn.id
                    self.metabolite_id_to_name[mets[0].id] = mets[0].name if pd.notnull(mets[0].name) else mets[0].id
                    self.exchange_metabolites.extend(mets)
                    self.exchanges.append(rxn.id)

                    # add biomass exs to separate set
                    if "biomass" in list(rxn.metabolites.keys())[0].id.lower():
                        self.biomass_exchange_ids = self.biomass_exchange_ids | {rxn.id}
        
        # convert to attribute lists
        self.exchange_ids = list(self.exchange_ids)
        self.reaction_ids = list(self.reaction_ids)
        self.exchange_metabolites = list(set(self.exchange_metabolites))
        self.exchanges = list(set(self.exchanges))
        self.biomass_exchange_ids = list(self.biomass_exchange_ids)

        # initialize env
        self.media = utils.check_media(self)
        rows = (self.n_iterations) * self.m_vals[0] * self.m_vals[1] + 1 # add one iteration for initial env
        cols = len(self.exchange_ids)
        self.env_fluxes = np.zeros((rows, cols))
        initial_media_masks = [np.array(self.exchange_ids) == rxn_id for rxn_id in list(self.media.keys())]
        for flux_idx, flux in enumerate(list(self.media.values())):
            self.env_fluxes[0][initial_media_masks[flux_idx]] = -flux

        # set columns for multi-indexing
        iters_col = np.repeat(np.arange(1, self.n_iterations+1), self.m_vals[0] * self.m_vals[1]) 
        run_col = np.tile(np.arange(self.m_vals[0] * self.m_vals[1]), self.n_iterations)
        iters_col = np.insert(iters_col, 0, 0) # add 0th iteration
        run_col = np.insert(run_col, 0, 0) # add 0th run 
        multi_idx = [iters_col, run_col]
        self.env_fluxes = pd.DataFrame(self.env_fluxes, columns=self.exchange_ids, index=multi_idx) # convert to interprettable df
        self.env_fluxes.index.names = ["Iteration", "Run"]

        # initialize org_fluxes
        rows = self.n_iterations * self.m_vals[0] * self.m_vals[1] * len(self.models)
        cols = len(self.reaction_ids)
        self.org_fluxes = np.zeros((rows, cols)) # pfba will drop run column
        
        # create unique multi-index for org_fluxes
        models_col = np.tile(np.arange(self.num_models), self.n_iterations * self.m_vals[0] * self.m_vals[1]) 
        iters_col = np.repeat(np.arange(self.n_iterations), self.m_vals[0] * self.m_vals[1] * self.num_models) 
        run_col = np.tile(np.repeat(np.arange(self.m_vals[0] * self.m_vals[1]), self.num_models), self.n_iterations) 
        multi_idx = [models_col, iters_col, run_col]
        self.org_fluxes = pd.DataFrame(self.org_fluxes, columns=self.reaction_ids, index=multi_idx)	# convert to interprettable df
        self.org_fluxes.index.names = ["Model", "Iteration", "Run"]

        # store model names
        self.model_names = {model_idx: model.name for model_idx, model in enumerate(self.models)}

        return
        
    def _update_media(self, iteration): 
        """ 
        Update the media conditions for each iteration based on the fluxes simulated for each model. 
        This method wraps around the _flux_function and handles the media update logic in accordance 
        with fixed point solvers. 2 Fixed point solver methods can currently be used.

        [Flow / Hierarchy]
        Call to `_flux_function()` to simulate fluxes for each model in the community in the current 
            iteration's environment.
        Call to fixed point solver methods: `_relaxation_iteration_update()` or 
            `_anderson_iteration_update()` dependent on the self.fp_method attribute, to update
            the media conditions for the next iteration based on the current iteration's fluxes. The
            default method uses relaxation with a relaxation_ratio of 1.0 (picard-equivalent), which
            updates the media conditions based on the sum of the organism fluxes and the current media
            conditions.
            The Anderson method is currently under development and not fully implemented.

        Args:
            iteration (int): The current iteration number for the simulation. Used to index into the environmental and organism fluxes DataFrames to update the media conditions for the next iteration.
        
        State Inputs (Attributes Used):
            self.env_fluxes (pd.DataFrame): Full environmental fluxes DataFrame for all iterations and 
                runs. Multi-indexed by iteration and run, with columns as unique exchange reaction IDs 
                for the entire community. Size is (n_iterations, n_exchanges). Units are mmol/(gT * hr).
            self.org_fluxes (pd.DataFrame): Full organism fluxes DataFrame for all iterations and runs. 
                Multi-indexed by model, iteration, and run, with columns as unique (internal and 
                exchange) reaction IDs for the entire community. 
                Size is (n_iterations * n_models, n_reactions). Units are mmol/(gT * hr).

        State Modified (Side Effects):
            self.env_fluxes (pd.DataFrame): Updated environmental fluxes DataFrame for the next 
                iteration based on the current iteration's organism fluxes and media conditions. The 
                update is performed according to the chosen fixed point solver method.

        Calls:
            - `self._flux_function(iteration)`
            - `self._relaxation_iteration_update()`
            - `self._anderson_iteration_update()` - reserved for future development, not currently implemented.
        """
        # run organism flux function
        self._flux_function(iteration)

        # update media: f_n+1 = f_n - sum(v_nij)
        env_current = self.env_fluxes.loc[iteration, 0][:].to_numpy().reshape(-1, 1)   # (row, col) = (n_ex, 1)     # uptake = positive
        org_exchange_fluxes = self.org_fluxes.loc[:, iteration, 0][self.env_fluxes.columns].to_numpy().T # (row, col) = (n_ex, n_org) # uptake = negative flux
        sum_org_flux = org_exchange_fluxes.sum(axis=1).reshape(-1, 1) # (n_ex, n_org) -> (n_ex, ) sum across orgs

        # get init env for iteration 0
        env_current = self.env_fluxes.loc[0, 0][:].to_numpy().reshape(-1, 1)

        # pull ex info for iteration and set uptake to 0
        org_exchange_fluxes = self.org_fluxes.loc[:, iteration, 0][self.env_fluxes.columns].to_numpy().T # (row, col) = (n_ex, n_org) # uptake = negative flux
        org_exchange_fluxes[org_exchange_fluxes < 0] = 0 # only secretion counts
        
        # sum org fluxes and media
        sum_org_flux = org_exchange_fluxes.sum(axis=1).reshape(-1, 1)
        self.env_fluxes.loc[iteration+1, 0] = (env_current + sum_org_flux).flatten() # (n_ex, 1) + (n_ex, 1) -> (n_ex, 1)
        
        # add fixed point relaxation method
        if self.fp_method == "relaxation":
            self._relaxation_iteration_update()
        return

    def _relaxation_iteration_update(self):
        """
        Update the environmental fluxes for the next iteration using a relaxation/picard method. 
        If the `self.fp_method` attribute is set to "picard", the relaxation method applies with 
        a relaxation ratio of 1.0, effectively performing a standard Picard iteration.

        State Inputs (Attributes Used):
            self.env_fluxes (pd.DataFrame): Full environmental fluxes DataFrame for all iterations and 
                runs. Multi-indexed by iteration and run, with columns as unique exchange reaction IDs
                for the entire community. Size is (n_iterations, n_exchanges). Units are mmol/(gT * hr).
            self.relaxation_ratio (float): The relaxation ratio used for updating the environmental 
                fluxes. A value of 1.0 corresponds to a standard Picard iteration
        """
        self.env_fluxes.loc[self.current_iteration+1, 0] = (1- self.relaxation_ratio) * self.env_fluxes.loc[self.current_iteration, 0] + self.relaxation_ratio * self.env_fluxes.loc[self.current_iteration+1, 0]
        return

    def _anderson_iteration_update(self):
        """
        Update the environmental fluxes for the next iteration using an Anderson acceleration method. 
        This method is currently under development and not fully implemented.

        State Inputs (Attributes Used):
            self.env_fluxes (pd.DataFrame): Full environmental fluxes DataFrame for all iterations and 
                runs. Multi-indexed by iteration and run, with columns as unique exchange reaction IDs
                for the entire community. Size is (n_iterations, n_exchanges). Units are mmol/(gT * hr).
        """
        # Placeholder for Anderson acceleration method implementation
        raise NotImplementedError("Anderson acceleration method is not yet implemented.")
        
    def _flux_function(self, iteration):
        """
        Apply the flux function for each model in the community for the given iteration. This method 
        wraps around methods to initialize the environemnt and simulation. The method applies the giFBA 
        method to handle overconsumption through recursive subroutine calls to itself, adjusting the 
        environmental fluxes as necessary.

        [Flow / Hierarchy]
        If this is the first run of the iteration, initialize the environmental scaling factors based 
            on the relative abundances of the models.
        Per Model, call `_set_env()` to set the exchange reactions of the model to match the current 
            environmental fluxes for the iteration (scaled by fractions defined by 
            `self._env_scaling_factors`).
        Per Model, call `_sim_fba()` to simulate the fluxes for the model using the specified FBA method
            (either "pfba" or "fba") and store the resulting fluxes in the `self.org_fluxes` DataFrame.
        After simulating all models, call `_check_overconsumption()` to check for overconsumption of 
            environmental metabolites. If overconsumption is detected, the method will recursively call 
            itself to re-run the flux function with adjusted environmental fluxes
        
        Args:
            iteration (int): The current iteration number for the simulation. Used to index into the 
            environmental and organism fluxes DataFrames to update the media conditions for the next 
            iteration.
        
        State Inputs (Attributes Used):
            self._is_rerun (bool): A flag indicating whether this is a re-run of the flux function due 
                to overconsumption. If True, the environmental scaling factors are not re-initialized.
            self._env_scaling_factors (np.ndarray): An array of scaling factors for the environmental 
                fluxes. Initialized to be reciprocal of the relative abundances of the models on the 
                first run of the iteration. After overconsumption, array is updated at given 
                overconsumed indices to scale down the environmental fluxes for the next run.
        
        State Modified (Side Effects):
            self._env_scaling_factors (np.ndarray): Updated/Initialized scaling factors for the 
                environmental fluxes.
        
        Calls:
            - `self._set_env(iteration, model_idx)`
            - `self._sim_fba(iteration, model_idx)`
            - `self._check_overconsumption(iteration)`
        """
        # # define env bounds per organism for the current iteration
        if not(self._is_rerun): # if first run of iteration, just initialize scaled by rel abund only otherwise do nothing
            self._env_scaling_factors = np.ones((self.num_models, len(self.exchange_ids)))  # initialize update rate (used to scale ex flux bounds
            for model_idx in range(self.num_models):
                self._env_scaling_factors[model_idx, :] = self._env_scaling_factors[model_idx, :] / self.rel_abund[model_idx]

        # simulate each organism
        for model_idx in range(self.num_models):
            # if self.verbose: print(" Simulating model:", model_idx+1, " of ", self.num_models)
            # set media
            self._set_env(iteration, model_idx)

            # simulate each org
            self._sim_fba(iteration, model_idx)

        # check over consumption
        self._check_overconsumption(iteration)

        return

    def _set_env(self, iteration, model_idx):
        """
        Function to set the exchange reactions of a model to match the environment fluxes
        for a given iteration and run. This is mainly provided to ensure a cleaner wrapper function.

        Args:
            iteration (int): The current iteration number for the simulation. Used to index into the 
                environmental fluxes DataFrame to set the exchange reaction bounds for the model.
            model_idx (int): The index of the model for which to set the environment.
        
        State Inputs (Attributes Used):
            self.models (list[cb.Model]): A list of all Cobra models present in the community. Each
                model's exchange reactions will be set to match the environmental fluxes for the current
                iteration.
            self.env_fluxes (pd.DataFrame): Full environmental fluxes DataFrame for all iterations and 
                runs. Multi-indexed by iteration and run, with columns as unique exchange reaction IDs
                for the entire community. Size is (n_iterations, n_exchanges). Units are mmol/(gT * hr).
            self._env_scaling_factors (np.ndarray): An array of scaling factors for the environmental 
                fluxes. Used to scale the exchange reaction bounds for each model based on the relative 
                abundances and/or overconsumption of the models in the community.
        """
        for ex in self.models[model_idx].exchanges:
            mask = np.array(self.exchange_ids) == ex.id
            if mask.any():  # Check if the exchange reaction exists in exchange_ids
                # .item() because boolean-indexing _env_scaling_factors yields a
                # 1-element array, and cobra needs a scalar bound: numpy >= 2.0
                # raises TypeError when it calls isinf() on a 1-d array.
                ex.lower_bound = (-self._env_scaling_factors[model_idx, mask] * self.env_fluxes.loc[iteration, 0][ex.id]).item()
       
        return

    def _sim_fba(self, iteration, model_idx):
        """
        Simulate parsimonious FBA (pFBA) or FBA on a model and store the results, checks if the 
        solution is above a minimum growth objective, and stores the resulting fluxes in the provided 
        DataFrame.

        Args:
            iteration (int): The current iteration number for the simulation. Used to index into the 
                organism fluxes DataFrame to store the resulting fluxes for the model.
            model_idx (int): The index of the model for which to simulate FBA.
        
        State Inputs (Attributes Used):
            self.models (list[cb.Model]): A list of all Cobra models present in the community. Each
                model will be simulated using the specified FBA method.
            self.method (str): The FBA method to use for the simulation
        
        State Modified (Side Effects):
            self.org_fluxes (pd.DataFrame): Updated organism fluxes DataFrame with the results of the 
                FBA simulation for the model at the given iteration. The fluxes are scaled by the model's
                relative abundance in the community, converting v_ij (mmol/(g_i * hr)) to 
                V_ij (mmol/(gT * hr)).
        """
        # run check growth
        growth_check = self.models[model_idx].slim_optimize()
        
        if self.debug:
            if model_idx == 0:
                print("#"*45)
                print("Run Info")
            print(f"Objective value (model {model_idx}): {growth_check}")


        if growth_check > GROWTH_MIN_OBJ:
            self.simulation_count +=1
            if self.method == "pfba":
                solution = cb.flux_analysis.parsimonious.pfba(self.models[model_idx])

            elif self.method == "fba":
                solution = self.models[model_idx].optimize()
            
            self.org_fluxes.loc[(model_idx, iteration, 0), list(solution.fluxes.index)] = self.rel_abund[model_idx] * solution.fluxes.values 
        # do nothing otherwise - already initiated as zeros!
        return
    
    def _check_overconsumption(self, iteration):
        """
        Check over-consumption of environmental metabolites. If over-consumption occurs, environmental 
        bounds are scaled down in accordance with the over-consumption method chosen (Newton method or
        Optimization method) followed by re-running the flux function. If no over-consumption occurs, 
        the function returns without modifying the environmental fluxes.

        [Flow / Hierarchy]
        Pull the environmental fluxes and organism fluxes for the current iteration.
        Calculate the total organism fluxes for each environmental metabolite.
        Check if any environmental metabolite is over-consumed (i.e., total organism flux exceeds
            available environmental flux).
        If over-consumption is detected, call the appropriate method (either `self._optim_method_x` or 
            `self._newton_method_x`) to adjust the environmental fluxes. Either method will determine a
            scaling factor (identical for each model) to apply to the environmental fluxes for the next 
            run, to ensure metabolite consumption matches availability.
        Re-run the flux function with the adjusted environmental fluxes.

        Args:
            iteration (int): The current iteration number for the simulation. Used to index into the 
                environmental and organism fluxes DataFrames to check for over-consumption of metabolites.
        
        State Inputs (Attributes Used):
            self.env_fluxes (pd.DataFrame): Full environmental fluxes DataFrame for all iterations and 
                runs. Multi-indexed by iteration and run, with columns as unique exchange reaction IDs
                for the entire community. Size is (n_iterations, n_exchanges). Units are mmol/(gT * hr).
            self.org_fluxes (pd.DataFrame): Full organism fluxes DataFrame for all iterations and runs. 
                Multi-indexed by model, iteration, and run, with columns as unique (internal and 
                exchange) reaction IDs for the entire community. Size is (n_models, n_iterations, 
                n_runs). Units are mmol/(gT * hr).
            self.oc_method (str): The method to use for adjusting environmental fluxes in case of over-
                consumption. Must be either "optim" (optimization method) or "newton" (Newton-Raphson 
                method).
            self.oc_rounding (int): The number of decimal places to round the over-consumption values to.
            self._is_rerun (bool): A flag indicating whether this is a re-run of the flux function due 
                to previous over-consumption. 
            self._rerun_count (int): A counter for the number of times the flux function has been re-run 
                due to over-consumption in the current iteration. Used to prevent infinite recursion and
                for general simulation verbose outputs - this value is overridden at the start of each 
                iteration.
            
        State Modified (Side Effects):
            self._is_rerun (bool): Updated to True if over-consumption is detected and the flux function 
                is re-run with adjusted environmental fluxes.
            self._rerun_count (int): Incremented by 1 if over-consumption is detected and the flux function 
                is re-run.
            self._env_scaling_factors (np.ndarray): Updated scaling factors for the environmental fluxes
                if over-consumption is detected, to ensure that the next run of the flux function uses
                adjusted environmental fluxes that match the available metabolites.
        
        Calls:
            - `self._optim_method_x(iteration, overconsumption_ratio, org_exchange_fluxes, env_current)` 
            - `self._newton_method_x(iteration, overconsumption_ratio, org_exchange_fluxes, env_current)`
            - `self._flux_function(iteration)`

        """
        # pull iteration info and establish array shapes
        env_current = self.env_fluxes.loc[iteration, 0][:].to_numpy().reshape(-1, 1)   # (row, col) = (n_ex, 1)     # uptake = positive
        org_exchange_fluxes = self.org_fluxes.loc[:, iteration, 0][self.env_fluxes.columns].to_numpy().T # (row, col) = (n_ex, n_org) # uptake = negative flux

        # get org fluxes
        total_org_flux = org_exchange_fluxes.sum(axis=1).reshape(-1, 1) # (n_ex, n_org) -> (n_ex, 1) sum across orgs

        # check if environment fluxes are under-saturated
        overconsumption_ratio = np.zeros_like(total_org_flux)
        with np.errstate(divide='ignore', invalid='ignore'): # ignore division by zero warnings
            overconsumption_ratio[np.abs(env_current) >= 1e-12] = -total_org_flux[np.abs(env_current) >= 1e-12].astype(np.float64) / env_current[np.abs(env_current) >= 1e-12].astype(np.float64) # only check non-zero env fluxes
        
        if self.debug:
            print("\nenv fluxes (mmol/(gT/hr)):")
            print(self.env_fluxes.loc[iteration, 0].T)
            print("\norg fluxes (mmol/(gT/hr)):")
            print(self.org_fluxes.loc[:, iteration, 0][self.env_fluxes.columns])
            print("#"*45)
            print()

        # check if iteration uses more flux than available in environment
        if not self._is_rerun:
            self._rerun_count=0

        # initialize lists on first call fro newton method tracking
        if iteration == 0 and not self._is_rerun:
            self._x_history = []
            self._oc_history = []
            self._rerun_history = []
            self._iteration_history = []
            self._ex_over_dict = {ex: {"x_history": [], "oc_history": [], "iteration_history": [], "rerun_history": []} for ex in self.env_fluxes.columns}
        
        # re-run flux if overconsumed, adjusting only the over-consumed reactions
        if overconsumption_ratio.max().round(self.oc_rounding) > 1 or (self._rerun_count !=0 and overconsumption_ratio.max().round(self.oc_rounding) <1): # rounding avoids numerical issues with X being set to inf or nan
            if self.oc_method == "optim":
                self._optim_method_x(iteration, overconsumption_ratio, org_exchange_fluxes, env_current)
            if self.oc_method == "newton":
                self._newton_method_x(iteration, overconsumption_ratio, org_exchange_fluxes, env_current)
            self._is_rerun = True
            self._rerun_count += 1
            self._flux_function(iteration)
        
        return
    
    def _optim_method_x(self, iteration, overconsumption_ratio, org_exchange_fluxes, env_current):
        """
        Finds a universal cap X for each metabolite to balance community consumption with available 
        media. Handles both over-consumption (pull down) and under-consumption (push up).

        Args:
            iteration (int): The current iteration number for the simulation. Used to index into the 
                environmental and organism fluxes DataFrames to check for over-consumption of 
                metabolites.
            overconsumption_ratio (np.ndarray): An array indicating the over-consumption factor for each
                environmental metabolite. Values greater than 1 indicate over-consumption, while
                values less than 1 indicate under-consumption. Maximum value (above 1) indicates there
                exists a scaling factor for the metabolite to balance consumption with availability.
            org_exchange_fluxes (np.ndarray): An array of the organism exchange fluxes for the current iteration.
            env_current (np.ndarray): An array of the environmental fluxes for the current iteration.
        
        State Inputs (Attributes Used):
            self.rel_abund (np.ndarray): An array of the relative abundances of the models in the 
                community.
            
        State Modified (Side Effects):
            self._env_scaling_factors (np.ndarray): Updated scaling factors for the environmental fluxes.
        """
        ex_over = np.argmax(overconsumption_ratio) # index of flux causing over-consumed
        # Iterate through all metabolites in the media

        # Current total consumption factor (Total_Flux / Media)
        oc_factor = overconsumption_ratio[ex_over, 0]
        
        if self.verbose: print(self.env_fluxes.columns[ex_over], f"over-consumed by factor of {overconsumption_ratio.max():.12f} (rerun count: {self._rerun_count})")

        # 1. Gather current individual fluxes (normalized by abundance)
        # org_exchange_fluxes is total weighted flux (a_i * v_i). We want internal flux v_i.
        rel_abund = self.rel_abund.flatten()
        v_ij_magnitudes = np.abs(org_exchange_fluxes[ex_over, :]) / rel_abund

        # 2. Define the community response function
        def residual(X):
            # Total = sum( abundance * min(Cap, Individual_Flux) )
            total_flux = np.sum(rel_abund * np.minimum(X, v_ij_magnitudes))
            return total_flux - self.env_fluxes.loc[(iteration, 0), self.env_fluxes.columns[ex_over]] # residual = total_flux - media_flux (want to find X where residual = 0)

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
            solution = root_scalar(residual, bracket=[low, high], method='brentq')
            X_opt = solution.root
        except (ValueError, RuntimeError):
            # Fallback to current best if root finding fails
            X_opt = high if oc_factor < 1 else low

        # 5. Apply the universal cap
        # scaling_factor * media = X_opt => scaling = X_opt / media
        self._env_scaling_factors[:, ex_over] = X_opt / env_current[ex_over, 0]
        
        if self.debug:
            print(f"  X = {X_opt:.6f}")

        return
    
    def _newton_method_x(self, iteration, overconsumption_ratio, org_exchange_fluxes, env_current):
        """
        Finds a universal cap X for each metabolite to balance community consumption with available 
        media. Handles both over-consumption (pull down) and under-consumption (push up). Uses a 
        deg 1 Newton method to infer the next best X based on the previous two runs of the flux function.

        Args:
            iteration (int): The current iteration number for the simulation. Used to index into the 
                environmental and organism fluxes DataFrames to check for over-consumption of 
                metabolites.
            overconsumption_ratio (np.ndarray): An array indicating the over-consumption factor for each
                environmental metabolite. Values greater than 1 indicate over-consumption, while values 
                less than 1 indicate under-consumption. Maximum value (above 1) indicates there exists 
                a scaling factor for the metabolite to balance consumption with availability.
            org_exchange_fluxes (np.ndarray): An array of the organism exchange fluxes for the current iteration.
            env_current (np.ndarray): An array of the environmental fluxes for the current iteration.

        State Inputs (Attributes Used):
            self.rel_abund (np.ndarray): An array of the relative abundances of the models in the 
                community.
            self._ex_over_dict (dict): A dictionary storing the history of X values, Over-Consumption 
                factors, rerun counts, and iteration numbers for each exchange reaction that has been 
                over-consumed. Used to track the previous two runs of the flux function for each 
                over-consumed metabolite.
        
        State Modified (Side Effects):
            self._env_scaling_factors (np.ndarray): Updated scaling factors for the environmental fluxes.
            self._ex_over_dict (dict): Updated with the new X value, Over-Consumption factor, rerun 
                count, and iteration number for the over-consumed metabolite.
            self._ex_over (int): Updated with the index of the currently over-consumed exchange reaction.
        """
        # reset if different ex is overconsumed on re-run
        if self._is_rerun and overconsumption_ratio.max() != 1:
            ex_over = np.argmax(overconsumption_ratio) # index of flux causing over-consumed
            if ex_over != self._ex_over:
                self._ex_over_dict[self.env_fluxes.columns[ex_over]] = {}
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["x_history"] = []
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["oc_history"] = []
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["rerun_history"] = []
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["iteration_history"] = []
                self._rerun_count = 0

        if overconsumption_ratio.max() > 1 or (self._rerun_count !=0 and overconsumption_ratio.max() <1):
            ex_over = np.argmax(overconsumption_ratio) # index of flux causing over-consumed
            if self._is_rerun and ex_over != self._ex_over:
                self._ex_over_dict[self.env_fluxes.columns[ex_over]] = {}
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["x_history"] = []
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["oc_history"] = []
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["rerun_history"] = []
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["iteration_history"] = []
                self._rerun_count = 0   

            if self.verbose: print(self.env_fluxes.columns[ex_over], f"over-consumed by factor of {overconsumption_ratio.max():.12f} (rerun count: {self._rerun_count})")
            if self.debug: print("v"*45)

            # adjust only over-consumed bound
            x_denom = 0
            for model_idx in range(self.num_models):
                if self.env_fluxes.columns[ex_over] in self.models[model_idx].reactions:
                    lb_ij = self.models[model_idx].reactions.get_by_id(self.env_fluxes.columns[ex_over]).lower_bound
                    V_ij = org_exchange_fluxes[ex_over, model_idx]
                    a_i = self.rel_abund[model_idx]
                    x_denom += V_ij / lb_ij
                    
                    if self.debug:
                        print("Model idx", model_idx, "    (alpha =", a_i[0],")")
                        print(f"  big V: {V_ij: 3.6f}   mmol/(gT/hr)")
                        print(f"  lil v: {V_ij/a_i[0]: 3.6f}   mmol/(gi/hr)")
                        print(f"     lb: {lb_ij[0]: 3.6f}   mmol/(gi/hr)")
            
            if not self._is_rerun or (self._is_rerun and self._rerun_count ==0):
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["x_history"].append(0)
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["oc_history"].append(0)
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["rerun_history"].append(-1)
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["iteration_history"].append(iteration)

                # assume n=1 uses this form
                x_n = env_current[ex_over, 0] / x_denom[0]
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["x_history"].append(x_n)

            if self._is_rerun and self._rerun_count !=0:
                # just use one LB for the ex over if we have already re-run
                for model_idx in range(self.num_models):
                    if self.env_fluxes.columns[ex_over] in self.models[model_idx].reactions:
                        lb = self.models[model_idx].reactions.get_by_id(self.env_fluxes.columns[ex_over]).lower_bound
                        break
                self._ex_over_dict[self.env_fluxes.columns[ex_over]]["x_history"].append(-lb[0])
                
            self._ex_over_dict[self.env_fluxes.columns[ex_over]]["oc_history"].append(overconsumption_ratio[ex_over, 0])

            # infer next best X based on deg 1 Newton Method
            m = (self._ex_over_dict[self.env_fluxes.columns[ex_over]]["x_history"][-1] - self._ex_over_dict[self.env_fluxes.columns[ex_over]]["x_history"][-2]) / (self._ex_over_dict[self.env_fluxes.columns[ex_over]]["oc_history"][-1] - self._ex_over_dict[self.env_fluxes.columns[ex_over]]["oc_history"][-2])
            b = self._ex_over_dict[self.env_fluxes.columns[ex_over]]["x_history"][-1] - m * self._ex_over_dict[self.env_fluxes.columns[ex_over]]["oc_history"][-1]
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
            self._env_scaling_factors[:, ex_over] = X_n_p_1 / env_current[ex_over, 0]

            # store for next run
            self._ex_over_dict[self.env_fluxes.columns[ex_over]]["rerun_history"].append(self._rerun_count)
            self._ex_over_dict[self.env_fluxes.columns[ex_over]]["iteration_history"].append(iteration)
            self._ex_over = ex_over

        return
    
    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        return False

    def summarize(self, iteration_shown=None):
        return CommunitySummary(self, iteration_shown)
    
    def average_periodicity(self):
        """Calculate the average periodicity of the system based on the environmental fluxes."""
        # if no convergence, give warning and return average of all iterations
        if self.periodicity is None and self.iter_converged is None:
            self.periodicity = int(self.n_iterations * self.attractor_size) # set periodicity to a percentage of total n_iterations if no convergence, so at least some averaging is done

            print("Model did not converge or show periodicity within the iteration limit, results may be unreliable.")
            print(f"{self.periodicity} iterations ({self.attractor_size*100:.1f}%) will be used for flux calculations, but consider increasing the number of iterations or checking model setup.")

        if self.periodicity is not None and self.iter_converged is not None:
            if self.periodicity == 1:
                if self.verbose: print("System Fixed Point found after", self.iter_converged, "iterations.")
            else:
                if self.verbose: print("System has periodic behavior with a period of", self.periodicity, "iterations. Last ", self.periodicity, "iterations will be averaged and stored.")
        
        # calculate average for the period size
        env_flux_avg = self.env_fluxes.loc[(slice(self.n_iterations - self.periodicity, self.n_iterations -1)), :].mean()
        org_flux_avg = self.org_fluxes.iloc[-self.periodicity * self.num_models:].groupby(level="Model").mean()
        return env_flux_avg, org_flux_avg
