"""Validation helpers and standalone utilities.

Most `check_*` functions are used internally by `gifbaObject.__init__` and
`gifbaObject.run_gifba` but are safe to call directly.
"""

from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

import cobra as cb
from cobra.util.solver import linear_reaction_coefficients
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gifba.gifba_object import gifbaObject


def load_simple_models(case: str) -> tuple[list[cb.Model], dict[str, float]]:
    """Load one of the bundled toy-model cases from `gifba/Toy_Models`.

    Args:
        case (str): Case name, one of `"1_1_single"`, `"1_2_single"`, `"1_3_parallel"`,
            `"2_1_competition"`, `"3_1_crossfeed"`, `"3_2_layered"`,
            `"4_1_crossfeed_competition"`, `"4_2_superfluous_crossfeed"`,
            `"5_1_coupling"` or `"5_2_dynamical"`.

    Returns:
        tuple[list[cobra.Model], dict[str, float]]: `(models, media)`, the case's models
            and the media it is run on.

    Raises:
        KeyError: If `case` is not a bundled case name.
    """
    case_models = {
        "1_1_single"               : ["sim1_1.json"],
        "1_2_single"               : ["sim1_2.json"],
        "1_3_parallel"             : ["sim1_3_org1.json", "sim1_3_org2.json"],
        "2_1_crossfeed"            : ["sim2_1_org1.json", "sim2_1_org2.json"],
        "2_2_layered"              : ["sim2_2_org1.json", "sim2_2_org2.json"],
        "2_3_superfluous_crossfeed": ["sim2_3_org1.json", "sim2_3_org2.json"],
        "3_competition"            : ["sim3_org1.json", "sim3_org2.json"],
        "4_1_crossfeed_competition": ["sim4_1_org1.json", "sim4_1_org2.json"],
        "4_2_coupling"             : ["sim4_2_org1.json", "sim4_2_org2.json"],
        "4_3_dynamical"            : ["sim5_2_org1.json", "sim5_2_org2.json"]
    }

    case_media = None
    if case in ["1_1_single", "1_2_single", "3_competition", "2_2_layered", "4_1_crossfeed_competition", "4_2_coupling", "4_3_dynamical"]: # A only in media 
        case_media = {"EX_A(e)": -10}
    elif case in ["1_3_parallel", "2_3_superfluous_crossfeed"]:
        case_media = {"EX_A(e)": -10, "EX_B(e)": -10}
    elif case in ["2_1_crossfeed"]:
        case_media = {"EX_A(e)": -10, "EX_C(e)": -10}

    models = []
    for file_name in case_models[case]:
        model_path = files("gifba").joinpath("Toy_Models", file_name)
        models.append(cb.io.load_json_model(str(model_path)))
    
    return models, case_media

def find_min_medium(
    community: gifbaObject | None = None,
    models: cb.Model | list[cb.Model] | None = None,
    base_media: dict[str, float] | None = None,
    min_growth: float | None = None,
) -> dict[str, float]:
    """Derive a minimal medium that supports growth of every model.

    Runs `cobra.medium.minimal_medium` on each model with `base_media` added as sinks,
    then takes the per-exchange maximum across models, unioned with `base_media`. Pass
    either a `community` or explicit `models`/`base_media`/`min_growth`.

    Args:
        community (gifbaObject, optional): Community whose `media` is a
            `[base_media, min_growth]` list. Its models are used directly.
        models (cobra.Model | list[cobra.Model], optional): Models to use when no
            community is given. They are copied, so the caller's models are untouched.
        base_media (dict[str, float], optional): Exchange IDs to fluxes always included in
            the medium. Signs are ignored. Defaults to an empty medium.
        min_growth (float, optional): Minimum growth each model must reach. Defaults to
            0.1.

    Returns:
        dict[str, float]: Media dict of negative (uptake) fluxes keyed by exchange ID.

    Raises:
        ValueError: If neither `community` nor `models` is given.
    """
    if community is not None:
        if isinstance(community.media, list):
            base_media, min_growth = community.media[0], community.media[1]
        models = community.models
    elif models is None:
        raise ValueError("find_min_medium needs either a community or a list of models.")
    else:
        if isinstance(models, cb.Model):
            models = [models]
        models = [model.copy() for model in models]

    base_media = {ex: abs(flux) for ex, flux in (base_media or {}).items()}
    min_growth = 0.1 if min_growth is None else min_growth

    min_medium = []
    for model in models:
        with model as model_copy:
            for rxn_id, uptake in base_media.items():
                if rxn_id in model_copy.exchanges:
                    met = list(model_copy.exchanges.get_by_id(rxn_id).metabolites.keys())[0]
                    model_copy.add_boundary(met, type="sink", reaction_id=rxn_id+'_tmp',lb=-1*uptake,ub=1000)
                
            for ex in model_copy.exchanges:
                ex.lower_bound = -1000
                ex.upper_bound = 1000

            model_min_medium = cb.medium.minimal_medium(model_copy, min_growth,minimize_components=True)

            if model_min_medium is not None:
                min_medium.append(pd.Series(model_min_medium.to_dict(), dtype=float))
    min_medium.append(pd.Series(base_media, dtype=float)) # add base media to ensure all components are included
    
    min_medium = pd.concat(min_medium, axis=1).fillna(0)
    min_medium = (- 1* min_medium.max(axis=1)).to_dict() # convert to uptake and dict

    return min_medium


def check_rel_abund(rel_abund, n_models):
    """Validate and normalize relative abundances.

    Args:
        rel_abund (str | array-like | None): Abundance per model. `None` or any string
            (e.g. `"equal"`) gives an even split. Values are normalized to sum to 1.
        n_models (int): Number of models in the community.

    Returns:
        np.ndarray: Relative abundances of shape `(n_models, 1)` summing to 1.

    Raises:
        ValueError: If the length doesn't match `n_models`, or values are negative or
            sum to zero.
    """
    if rel_abund is None:
        rel_abund = np.ones(n_models) / n_models
    elif isinstance(rel_abund, str):
        rel_abund = np.ones(n_models) / n_models
    elif not isinstance(rel_abund, np.ndarray):
        rel_abund = np.array(rel_abund)
    if rel_abund.ndim != 1:
        rel_abund = rel_abund.flatten()
    if rel_abund.shape[0] != n_models:
        raise ValueError(f"Relative abundances must be a 1D array of length {n_models}.")
    if np.any(rel_abund < 0) or np.sum(rel_abund) == 0:
        raise ValueError("Relative abundances must be non-negative and sum to a positive value.")
    if rel_abund.sum() != 1:
        rel_abund = rel_abund / rel_abund.sum()
        print("Relative abundances set to:", rel_abund)

    rel_abund = rel_abund.astype(float).reshape(-1, 1)
    return rel_abund

def check_n_iterations(n_iterations):
    """Coerce the iteration count to a positive integer.

    Args:
        n_iterations (int | float | None): Requested number of iterations. Defaults to 10
            if None; values below 1 are raised to 1.

    Returns:
        int: Number of iterations to run.
    """
    if n_iterations is None:
        n_iterations = 10
    elif not isinstance(n_iterations, int):
        n_iterations = int(n_iterations)
    if n_iterations < 1:
        n_iterations = 1
        print("Iterations set to:", n_iterations)
    
    return n_iterations

def check_media(community):
    """Resolve a community's media into a plain flux dict.

    Args:
        community (gifbaObject): Community whose `media` is `None`, `"complete"`, a
            `[base_media, min_growth]` list, or a dict of exchange IDs to fluxes. `None`
            and `"complete"` open every exchange in `community.exchange_ids` to -1000; a
            list is passed to `find_min_medium`.

    Returns:
        dict[str, float]: A copy of the resolved media. `community.media` is also updated
            in place.

    Raises:
        ValueError: If the media type is unsupported, or a key isn't a string or a value
            isn't a number.
    """

    # None or "complete" == Set all exchanges to -1000
    community.media = "complete" if community.media is None else community.media
    if isinstance(community.media, str):
        if community.media.lower() == "complete":
            community.media = dict(zip(community.exchange_ids, np.full(len(community.exchange_ids), -1000)))
        else:
            raise ValueError("Media must be None, 'complete', float, or a dict with reaction IDs as keys and flux values as values.")
    
    if isinstance(community.media, (list)):
        community.media = find_min_medium(community)
    elif not isinstance(community.media, (dict, str)):
        raise ValueError("Media must be None, 'complete', float, or a dict with reaction IDs as keys and flux values as values.")

    for rxn_id, flux in community.media.items():
        if not isinstance(rxn_id, str):
            raise ValueError(f"Reaction ID {rxn_id} must be a string.")
        if not isinstance(flux, (int, float)):
            raise ValueError(f"Flux value for reaction {rxn_id} must be a number.")

    return community.media.copy()

def check_models(models):
    """Validate and copy the community models.

    Args:
        models (cobra.Model | list[cobra.Model]): A single model or a list of models.

    Returns:
        list[cobra.Model]: Copies of the input models.

    Raises:
        ValueError: If `models` is None, not a model/list, or contains a non-model.
    """
    if models is None:
        raise ValueError("Models must be provided as a list of cobra.Model objects or single cobra.Model.")

    elif not isinstance(models, (list, cb.Model)):
        raise ValueError("Models must be provided as a list of cobra.Model objects or single cobra.Model.")
    else:
        if isinstance(models, cb.Model):
            models = [models]

    models_list = []
    for model in models:
        if not isinstance(model, cb.Model):
            raise ValueError(f"Model {model} is not a valid cobra.Model object.")
        
        models_list.append(model.copy())
    
    return models_list

def check_method(method):
    """Validate the FBA method name.

    Args:
        method (str | None): `"pfba"` or `"fba"`, case-insensitive. Defaults to `"pfba"`
            if None.

    Returns:
        str: The lower-case method name.

    Raises:
        ValueError: If `method` isn't a string or isn't one of the two methods.
    """
    if method is None:
        method = "pfba"
    elif not isinstance(method, str):
        raise ValueError("Method must be a string, either 'pfba' or 'fba'.")
    else:
        if isinstance(method, str):
            if method.lower() == "pfba":
                method = "pfba"
            elif method.lower() == "fba":
                method = "fba"
            else:
                raise ValueError("method must be either 'pfba' or 'fba'.")

    return method

def prep_micom_cfba(community_id, ids, paths, rel_abund=None):
    """Build a compartmentalized cFBA model from a MICOM community.

    Requires the `micom` extra. The objective is the abundance-weighted sum of each
    organism's biomass reaction.

    Args:
        community_id (str): ID given to the returned `cobra.Model`.
        ids (list[str]): Organism IDs, used as suffixes on MICOM's reaction IDs.
        paths (list[str]): Model file paths, in the same order as `ids`.
        rel_abund (list[float], optional): Abundance per organism. Defaults to an even
            split.

    Returns:
        tuple[cobra.Model, micom.Community, dict]: `(cfba_model, micom_community,
            objective_dict)`, where `objective_dict` maps each biomass reaction to its
            objective coefficient.
    """
    from micom import Community
    import cobra as cb
    import pandas as pd

    abund = rel_abund if rel_abund is not None else [1/len(ids) for _ in range(len(ids))]
    community_df = pd.DataFrame({
        "id": ids,
        "file": paths,
        "abundance": abund
    })

    # create micom community
    micom_community = Community(community_df)

    cfba_model = cb.Model(community_id)

    for met in micom_community.metabolites:
        cfba_model.add_metabolites([met.copy()])
    
    for rxn in micom_community.reactions:
        new_rxn = rxn.copy()
        new_rxn.id = rxn.id
        new_rxn.lower_bound = rxn.lower_bound
        new_rxn.upper_bound = rxn.upper_bound
        
        # Remap stoichiometry to the cfba_model's metabolites
        new_stoichiometry = {cfba_model.metabolites.get_by_id(m.id): c for m, c in new_rxn.metabolites.items()}
        new_rxn.subtract_metabolites(new_rxn.metabolites)  # Clear old objects
        new_rxn.add_metabolites(new_stoichiometry)          # Attach new objects
        
        cfba_model.add_reactions([new_rxn])

    objective_dict = {}
    for constraint in micom_community.constraints:
        if "community" in constraint.name:
            coefficients_dict = constraint.expression.as_coefficients_dict()

            for var in constraint.variables:
                for idx, model_id in enumerate(ids):
                    if var.name.endswith(f"_{model_id}") and var.name != "community_objective":
                        rxn = cfba_model.reactions.get_by_id(var.name)
                        objective_dict[rxn] = abs(coefficients_dict[var])

    # set coeffs to relative abundances for each organism's biomass reaction
    for idx, model_id in enumerate(ids):
        for rxn in objective_dict.keys():
            if rxn.id.endswith(f"_{model_id}"):
                objective_dict[rxn] = abund[idx]

    # Set the objective of the cfba_model
    cfba_model.objective = objective_dict
    cfba_model.objective_direction = "max"

    return cfba_model, micom_community, objective_dict

def prepare_compartmentalized_model(
    community: gifbaObject,
    rel_abund: list[float] | np.ndarray | None = None,
    obj_rxn_ids: list[str] | None = None,
) -> tuple[cb.Model, list[cb.Reaction]]:
    """Build a compartmentalized cFBA model from a giFBA community.

    Each organism gets its own compartments (`c1`, `e1`, `c2`, `e2`, ...) and `e0` is the
    shared pool, so the result can be compared against giFBA. The objective is the
    unweighted sum of the organisms' objective reactions. The community's models are
    copied first and are left unchanged.

    Args:
        community (gifbaObject): Community providing the models, media and
            `community_id`.
        rel_abund (list[float] | np.ndarray, optional): Abundance per organism, flat or
            shaped `(n_models, 1)`. Defaults to `community.rel_abund`.
        obj_rxn_ids (list[str], optional): Reaction IDs (after renaming) to use as the
            objective. Defaults to every per-organism `biomass(e...)` reaction.

    Returns:
        tuple[cobra.Model, list[cobra.Reaction]]: `(comp_model, objective_reactions)`.
    """
    from cobra import Model

    models = [model.copy() for model in community.models]
    media = community.media
    rel_abund = community.rel_abund if rel_abund is None else rel_abund
    rel_abund = check_rel_abund(rel_abund, len(models)).flatten()
    community_id = community.community_id

    # community.create_vars()

    for model_idx, model in enumerate(models):
        # these will be converted to the internal reactions between compartment e1 or e2 moving to e0
        for ex in model.exchanges:
            ex.lower_bound = -1000
        for med_ex in media.keys():
            if med_ex in model.reactions:
                model.reactions.get_by_id(med_ex).lower_bound = media[med_ex] / rel_abund[model_idx]


    comp_model = Model("compartmentalized_model_"+str(community_id))
    met_mapping = dict({})
    compartments = []
    comp_to_model_idx = {}
    #======= Change Compartments (each model uses e0 and c{i+1} compartments)============	
    for model_idx, model in enumerate(models):
        # Change compartments for each model so that models[i] uses e0 and c{i+1} compartments
        for met in model.metabolites:
            # store original compartment and id
            orig_comp = met.compartment
            if orig_comp not in compartments:
                compartments.append(orig_comp)
            orig_id = met.id

            # adjust new compartment and id
            id = met.id.replace(f"biomass", f"biomass{model_idx +1}").replace(f"[c]", f"[c{model_idx +1}]").replace(f"[e]", f"[e{model_idx +1}]")
            met.id = id
            met.compartment = f"{orig_comp}{model_idx +1}"
            comp_to_model_idx[met.compartment] = model_idx
            # map original id to new id
            met_mapping[id] = orig_id

            # add metabolite to compartmentalized model
            comp_model.add_metabolites([met.copy()])
    
    
    # change reaction names
    for model_idx, model in enumerate(models):
        for rxn in model.reactions:
            orig_id = rxn.id
            # orig_mets = rxn.metabolites
            orig_ub = rxn.upper_bound
            orig_lb = rxn.lower_bound

            
            id = orig_id.replace("(e)", f"(e{model_idx+1})").replace("(c)", f"(c{model_idx+1})")
            if id == orig_id:
                id = orig_id + f"_m{model_idx+1}"

            new_rxn = rxn.copy()
            new_rxn.id = id
            new_rxn.upper_bound = orig_ub 
            new_rxn.lower_bound = orig_lb
            
            # rxn.copy() still points at the source model's metabolites
            new_stoichiometry = {comp_model.metabolites.get_by_id(m.id): c for m, c in new_rxn.metabolites.items()}
            new_rxn.subtract_metabolites(new_rxn.metabolites) # Clear old objects
            new_rxn.add_metabolites(new_stoichiometry)        # Attach new objects
            
            comp_model.add_reactions([new_rxn])
            
    # add e0 exchange rxns
    for rxn in comp_model.reactions:
        if rxn.boundary:
            # copy original ex_met 
            met_orig = list(rxn.metabolites.keys())[0]
            met_e0 = met_orig.copy()

            # adjust to move met from e{i+1} to e0
            met_e0.compartment = "e0"
            met_e0.id = met_orig.id.replace(f"[{met_orig.compartment}]", "[e0]")
            model_num = comp_to_model_idx[met_orig.compartment]
            comp_model.add_metabolites([met_e0])
            rxn.add_metabolites({
                met_e0: 1,
                # adjust for relative abundance and subtract 1 to account for original metabolite
                met_orig.id: (-1/rel_abund[model_num] + 1) 
            })

    # change bounds for e0 exchange reactions and add media uptake reactions
    for met in comp_model.metabolites:
        if met.compartment == "e0":
            ex = cb.Reaction(f"EX_{met.id}")
            ex.id = ex.id.replace("[", "(").replace("]", ")")
            ex.name = f"Exchange for {met.id}"
            ex.lower_bound = -1000  # allow uptake
            ex.upper_bound = 1000   # allow secretion
            ex.add_metabolites({met: -1})
            comp_model.add_reactions([ex])
        

    # Set objective to weighted sum of individual model biomass reactions
    objective_reactions = []
    for rxn in comp_model.reactions:

        if obj_rxn_ids is None:
            if "biomass(e" in rxn.id and not(rxn.id.endswith("(e0)")):
                objective_reactions.append(rxn)
        else:
            if rxn.id in obj_rxn_ids:
                objective_reactions.append(rxn)
    comp_model.objective = {rxn: 1 for rxn in objective_reactions}

    return comp_model, objective_reactions


def prepare_compartmentalized_model_with_micom(
    gifba_community: gifbaObject,
    model_paths: list[str],
    media: dict[str, float] | None = None,
    rel_abund: list[float] | np.ndarray | None = None,
    obj_rxn_ids: list[str] | None = None,
) -> tuple[cb.Model, list[cb.Reaction]]:
    """Build a compartmentalized cFBA model from MICOM's reaction set.

    Requires the `micom` extra. Same idea as `prepare_compartmentalized_model`, but the
    reactions come from a MICOM community built from `model_paths`, with internal
    reaction bounds copied from the giFBA models. The organisms are named `Org1`,
    `Org2`, ... so every per-organism reaction ID carries a `__Org{i}` suffix.

    Args:
        gifba_community (gifbaObject): Community providing the models, objective
            reactions, abundances and `community_id`. It is not modified.
        model_paths (list[str]): Model file paths, in the same order as the community's
            models.
        media (dict[str, float], optional): giFBA-style media, e.g. `{"EX_A(e)": -10}`.
            Converted to MICOM's `EX_A_m` form and applied to the shared exchanges;
            exchanges not listed are closed to uptake. Defaults to MICOM's medium,
            which is taken from the model files' exchange bounds.
        rel_abund (list[float] | np.ndarray, optional): Abundance per organism. Defaults
            to `gifba_community.rel_abund`.
        obj_rxn_ids (list[str], optional): MICOM reaction IDs (with the `__Org{i}`
            suffix) to use as the objective. Defaults to each organism's objective
            reaction in `gifba_community`.

    Returns:
        tuple[cobra.Model, list[cobra.Reaction]]: `(comp_model, objective_reactions)`.
            The objective is the abundance-weighted sum of `objective_reactions`,
            matching MICOM's community growth rate.
    """
    from micom import Community

    models = gifba_community.models
    n_models = len(models)
    rel_abund = gifba_community.rel_abund if rel_abund is None else rel_abund
    abund = check_rel_abund(rel_abund, n_models).flatten()
    ids = [f"Org{i+1}" for i in range(n_models)]

    community_df = pd.DataFrame({
        "id": ids,
        "file": model_paths,
        "abundance": abund,
    })
    micom_community = Community(community_df)
    if media is not None:
        micom_community.medium = {k.replace("(e)", "_m"): -v for k, v in media.items()}

    comp_model = cb.Model(f"compartmentalized_model_{gifba_community.community_id}")
    comp_model.add_reactions([rxn.copy() for rxn in micom_community.reactions])

    # Internal reactions take their bounds from the giFBA models. Exchanges keep
    # MICOM's bounds, since MICOM rescales them by abundance.
    for rxn in comp_model.reactions:
        for model_idx, org_id in enumerate(ids):
            suffix = f"__{org_id}"
            if not rxn.id.endswith(suffix):
                continue
            orig_id = rxn.id[:-len(suffix)]
            if orig_id in models[model_idx].reactions and not orig_id.startswith("EX_"):
                rxn.bounds = models[model_idx].reactions.get_by_id(orig_id).bounds
            break

    if obj_rxn_ids is None:
        obj_rxn_ids = [f"{gifba_community.objective_rxns[i]}__{ids[i]}" for i in range(n_models)]

    objective = {}
    for rxn_id in obj_rxn_ids:
        rxn = comp_model.reactions.get_by_id(rxn_id)
        org_idx = next((i for i, org_id in enumerate(ids) if rxn_id.endswith(f"__{org_id}")), None)
        objective[rxn] = abund[org_idx] if org_idx is not None else 1.0
    comp_model.objective = objective
    comp_model.objective_direction = "max"

    return comp_model, list(objective)
