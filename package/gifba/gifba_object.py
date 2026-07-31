import cobra as cb
import numpy as np
import pandas as pd
from cobra.util.solver import linear_reaction_coefficients
from . import utils
from .config import GROWTH_MIN_OBJ, ROUND
from .summary import CommunitySummary
import numpy as np
from scipy.optimize import root_scalar
from typing import Literal

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
    def __init__(self, models, media, rel_abund="equal", 
                 **kwargs):
                 #id=None, debug=False, v=False, OC_Rounding=ROUND, OC_Method="optim"):
        self.models = utils.check_models(models)
        self.media = media
        self.media = utils.check_media(self)
        self.size = len(self.models)
        self.rel_abund = utils.check_rel_abund(rel_abund, self.size)
        self.flow = None
        self.iters = None
        
        # simulation parameters with defaults
        self.threshold = kwargs.get("threshold", 1e-12)
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

    def run_gifba(self, 
                  iters: int, 
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
        Call `_create_vars()` to initialize storage variables for the simulation.
        Call `_update_media()` for each iteration to simulate growth at each discrete step and update 
            state variables in accordance with fixed point solver method. See `_update_media()` for 
            more details.
        Check fixed point convergence by comparing current iteration to all previous iterations to 
            identify fixed points/periodic behavior.
        Upon completion/convergence, call `_average_periodicity()` to perform element-wise average of 
            state variables in fixed point/period/attractor.

        Args:
            iters (int): Number of iterations to run the simulation.
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
            - `self._create_vars()`
            - `self._update_media()`
            - `self._average_periodicity()`
        """
        self.iters = utils.check_iters(iters)
        self.method = utils.check_method(method)
        self.threshold = self.threshold if threshold is None else threshold
        self.attractor_size = self.attractor_size if attractor_size is None else attractor_size
        self.relaxation_ratio = 1.0 if relaxation_ratio is None or fp_method == "picard" else relaxation_ratio
        self.fp_method = "relaxation" if fp_method is None or fp_method == "picard" else fp_method
        self.v = v
        self.debug = debug # will print info on every iteration and re-run, so use with caution
        

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
            self.iters (int): The total number of iterations to run for the simulation.
            self.size (int): The total number of models in the community.

        State Modified (Side Effects):
            self.env_fluxes (pd.DataFrame): Full environmental fluxes DataFrame for all iterations and 
                runs. Multi-indexed by iteration and run, with columns as unique exchange reaction IDs 
                for the entire community. Size is (n_iterations, n_exchanges). Units are mmol/(gT * hr).
            self.org_fluxes (pd.DataFrame): Full organism fluxes DataFrame for all iterations and runs. 
                Multi-indexed by model, iteration, and run, with columns as unique (internal and 
                exchange) reaction IDs for the entire community. 
                Size is (n_iterations * n_models, n_reactions). Units are mmol/(gT * hr).
            self.ex_to_met (dict[str, str]): Mapping of exchange reaction IDs to their corresponding 
                metabolite IDs.
            self.metid_to_name (dict[str, str]): Mapping of metabolite IDs to their human-readable names 
                (or exchange reaction IDs).
            self.exchange_metabolites (list[cb.Metabolite]): A list of all Cobra Metabolite objects 
                across the community.
            self.exchanges (list[str]): A list of all unique exchange reaction IDs in the community.
            self.org_exs (list[str]): A list of organism-specific exchange reaction IDs.
            self.biomass_exs (list[str]): A list of all unique biomass exchange reaction IDs in the 
                community.
            self.model_names (dict[int, str]): A mapping of the community model index to its string 
                model ID.
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
        Update the media conditions for each iteration based on the fluxes simulated for each model. 
        This method wraps around the _flux_function and handles the media update logic in accordance 
        with fixed point solvers. 2 Fixed point solver methods can currently be used.

        [Flow / Hierarchy]
        Call to `_flux_function()` to simulate fluxes for each model in the community in the current 
            iteration's environment.
        Call to fixed point solver methods: `_relaxation_iteration_update()` or 
            `_anderson_iteration_update()` dependent on self.fixed_point_solver attribute, to update 
            the media conditions for the next iteration based on the current iteration's fluxes. The 
            default method uses relaxation with no relaxation (relaxation_prop=0), which updates the 
            media conditions based on the sum of the organism fluxes and the current media conditions. 
            The Anderson method is currently under development and not fully implemented.

        Args:
            iter (int): The current iteration number for the simulation. Used to index into the environmental and organism fluxes DataFrames to update the media conditions for the next iteration.
        
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
            - `self._flux_function(iter)`
            - `self._relaxation_iteration_update()`
            - `self._anderson_iteration_update()` - reserved for future development, not currently implemented.
        """
        # run organism flux function
        self._flux_function(iter)

        # update media: f_n+1 = f_n - sum(v_nij)
        env_tmp = self.env_fluxes.loc[iter, 0][:].to_numpy().reshape(-1, 1)   # (row, col) = (n_ex, 1)     # uptake = positive
        run_exs = self.org_fluxes.loc[:, iter, 0][self.env_fluxes.columns].to_numpy().T # (row, col) = (n_ex, n_org) # uptake = negative flux
        sum_org_flux = run_exs.sum(axis=1).reshape(-1, 1) # (n_ex, n_org) -> (n_ex, ) sum across orgs


        # get init env for iter 0
        env_tmp = self.env_fluxes.loc[0, 0][:].to_numpy().reshape(-1, 1)

        # pull ex info for iter and set uptake to 0
        run_exs = self.org_fluxes.loc[:, iter, 0][self.env_fluxes.columns].to_numpy().T # (row, col) = (n_ex, n_org) # uptake = negative flux
        run_exs[run_exs < 0] = 0 # only secretion counts
        
        # sum org fluxes and media
        sum_org_flux = run_exs.sum(axis=1).reshape(-1, 1)
        self.env_fluxes.loc[iter+1, 0] = (env_tmp + sum_org_flux).flatten()#.round(ROUND) # (n_ex, 1) + (n_ex, 1) -> (n_ex, 1)
        
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
        self.env_fluxes.loc[self.iter+1, 0] = (1- self.relaxation_ratio) * self.env_fluxes.loc[self.iter, 0] + self.relaxation_ratio * self.env_fluxes.loc[self.iter+1, 0]
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
        

    def _flux_function(self, iter):
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
            iter (int): The current iteration number for the simulation. Used to index into the 
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
            - `self._set_env(iter, model_idx)`
            - `self._sim_fba(iter, model_idx)`
            - `self._check_overconsumption(iter)`
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


        return

    def _set_env(self, iter, model_idx):
        """
        Function to set the exchange reactions of a model to match the environment fluxes
        for a given iteration and run. This is mainly provided to ensure a cleaner wrapper function.

        Args:
            iter (int): The current iteration number for the simulation. Used to index into the 
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
            mask = np.array(self.org_exs) == ex.id
            if mask.any():  # Check if the exchange reaction exists in org_exs
                ex.lower_bound = -self._env_scaling_factors[model_idx, mask] * self.env_fluxes.loc[iter, 0][ex.id]
       
        return

    def _sim_fba(self, iter, model_idx):
        """
        Simulate parsimonious FBA (pFBA) or FBA on a model and store the results, checks if the 
        solution is above a minimum growth objective, and stores the resulting fluxes in the provided 
        DataFrame.

        Args:
            iter (int): The current iteration number for the simulation. Used to index into the 
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
            iter (int): The current iteration number for the simulation. Used to index into the 
                environmental and organism fluxes DataFrames to check for over-consumption of metabolites.
        
        State Inputs (Attributes Used):
            self.env_fluxes (pd.DataFrame): Full environmental fluxes DataFrame for all iterations and 
                runs. Multi-indexed by iteration and run, with columns as unique exchange reaction IDs
                for the entire community. Size is (n_iterations, n_exchanges). Units are mmol/(gT * hr).
            self.org_fluxes (pd.DataFrame): Full organism fluxes DataFrame for all iterations and runs. 
                Multi-indexed by model, iteration, and run, with columns as unique (internal and 
                exchange) reaction IDs for the entire community. Size is (n_models, n_iterations, 
                n_runs). Units are mmol/(gT * hr).
            self.OC_Method (str): The method to use for adjusting environmental fluxes in case of over-
                consumption. Must be either "optim" (optimization method) or "newton" (Newton-Raphson 
                method).
            self.OC_Rounding (int): The number of decimal places to round the over-consumption values to.
            self._is_rerun (bool): A flag indicating whether this is a re-run of the flux function due 
                to previous over-consumption. 
            self._rerun_ct (int): A counter for the number of times the flux function has been re-run 
                due to over-consumption in the current iteration. Used to prevent infinite recursion and
                for general simulation verbose outputs - this value is overridden at the start of each 
                iteration.
            
        State Modified (Side Effects):
            self._is_rerun (bool): Updated to True if over-consumption is detected and the flux function 
                is re-run with adjusted environmental fluxes.
            self._rerun_ct (int): Incremented by 1 if over-consumption is detected and the flux function 
                is re-run.
            self._env_scaling_factors (np.ndarray): Updated scaling factors for the environmental fluxes
                if over-consumption is detected, to ensure that the next run of the flux function uses
                adjusted environmental fluxes that match the available metabolites.
        
        Calls:
            - `self._optim_method_x(iter, is_overconsumed, run_exs, env_tmp)` 
            - `self._newton_method_x(iter, is_overconsumed, run_exs, env_tmp)`
            - `self._flux_function(iter)`

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
        Finds a universal cap X for each metabolite to balance community consumption with available 
        media. Handles both over-consumption (pull down) and under-consumption (push up).

        Args:
            iter (int): The current iteration number for the simulation. Used to index into the 
                environmental and organism fluxes DataFrames to check for over-consumption of 
                metabolites.
            is_overconsumed (np.ndarray): An array indicating the over-consumption factor for each
                environmental metabolite. Values greater than 1 indicate over-consumption, while
                values less than 1 indicate under-consumption. Maximum value (above 1) indicates there
                exists a scaling factor for the metabolite to balance consumption with availability.
            run_exs (np.ndarray): An array of the organism exchange fluxes for the current iteration.
            env_tmp (np.ndarray): An array of the environmental fluxes for the current iteration.
        
        State Inputs (Attributes Used):
            self.rel_abund (np.ndarray): An array of the relative abundances of the models in the 
                community.
            
        State Modified (Side Effects):
            self._env_scaling_factors (np.ndarray): Updated scaling factors for the environmental fluxes.
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

        return
    
    def _newton_method_x(self, iter, is_overconsumed, run_exs, env_tmp):
        """
        Finds a universal cap X for each metabolite to balance community consumption with available 
        media. Handles both over-consumption (pull down) and under-consumption (push up). Uses a 
        deg 1 Newton method to infer the next best X based on the previous two runs of the flux function.

        Args:
            iter (int): The current iteration number for the simulation. Used to index into the 
                environmental and organism fluxes DataFrames to check for over-consumption of 
                metabolites.
            is_overconsumed (np.ndarray): An array indicating the over-consumption factor for each
                environmental metabolite. Values greater than 1 indicate over-consumption, while values 
                less than 1 indicate under-consumption. Maximum value (above 1) indicates there exists 
                a scaling factor for the metabolite to balance consumption with availability.
            run_exs (np.ndarray): An array of the organism exchange fluxes for the current iteration.
            env_tmp (np.ndarray): An array of the environmental fluxes for the current iteration.

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

        return
    

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        return False

    def summarize(self, iter_shown=None):
        return CommunitySummary(self, iter_shown)
    
    def average_periodicity(self):
        """Calculate the average periodicity of the system based on the environmental fluxes."""
        # if no convergence, give warning and return average of all iterations
        if self.periodicity is None and self.iter_converged is None:
            self.periodicity = int(self.iters * self.attractor_size) # set periodicity to a percentage of total iters if no convergence, so at least some averaging is done

            print("Model did not converge or show periodicity within the iteration limit, results may be unreliable.")
            print(f"{self.periodicity} iterations ({self.attractor_size*100:.1f}%) will be used for flux calculations, but consider increasing the number of iterations or checking model setup.")

        if self.periodicity is not None and self.iter_converged is not None:
            if self.periodicity == 1:
                if self.v: print("System Fixed Point found after", self.iter_converged, "iterations.")
            else:
                if self.v: print("System has periodic behavior with a period of", self.periodicity, "iterations. Last ", self.periodicity, "iterations will be averaged and stored.")
        
        # calculate average for the period size
        env_flux_avg = self.env_fluxes.loc[(slice(self.iters - self.periodicity, self.iters -1)), :].mean()
        org_flux_avg = self.org_fluxes.iloc[-self.periodicity * self.size:].groupby(level="Model").mean()
        return env_flux_avg, org_flux_avg