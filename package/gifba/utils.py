import matplotlib.pyplot as plt
import cobra as cb
from cobra.util.solver import linear_reaction_coefficients
from importlib.resources import files
import numpy as np
import pandas as pd

# from package import gifba


def load_simple_models(number):        
    situation_models = {
        "1_1_single"               : ["sim1_1.json"],
        "1_2_single"               : ["sim1_2.json"],
        "1_3_parallel"             : ["sim1_3_org1.json", "sim1_3_org2.json"],
        "2_1_A_varied_abundance"   : ["sim2_1_org1.json", "sim2_1_org2.json"],
        "2_1_B_varied_abundance"   : ["sim2_1_org1.json", "sim2_1_org2.json"],
        "2_2_A_varied_rates"       : ["sim2_2_A_org1.json", "sim2_2_A_org2.json"],
        "2_2_B_varied_rates"       : ["sim2_2_B_org1.json", "sim2_2_B_org2.json"],
        "2_2_C_varied_rates"       : ["sim2_2_C_org1.json", "sim2_2_C_org2.json"],
        "2_3_rates_and_abundance"  : ["sim2_2_C_org1.json", "sim2_2_C_org2.json"],
        "3_1_crossfeed"            : ["sim3_1_org1.json", "sim3_1_org2.json"],
        "3_2_layered"              : ["sim3_2_org1.json", "sim3_2_org2.json"],
        "4_1_crossfeed_competition": ["sim4_1_org1.json", "sim4_1_org2.json"],
        "4_2_superfluous_crossfeed": ["sim4_2_org1.json", "sim4_2_org2.json"],
        "4_3_efficient_crossfeed"  : ["sim4_3_org1.json", "sim4_3_org2.json"],
        "5_1_coupling"             : ["sim5_1_org1.json", "sim5_1_org2.json"],
        "5_2_dynamical"            : ["sim5_2_org1.json", "sim5_2_org2.json"]
    }

    situation_media = None
    if number in ["1_1_single", "1_2_single", "2_1_A_varied_abundance", "2_1_B_varied_abundance", "2_2_A_varied_rates", "2_2_B_varied_rates", "2_2_C_varied_rates", "2_3_rates_and_abundance", "3_2_layered", "4_1_crossfeed_competition", "5_1_coupling", "5_2_dynamical"]: # A only in media 
        situation_media = {"EX_A(e)": -10}
    elif number in ["1_3_parallel"]:
        situation_media = {"EX_A(e)": -10, "EX_B(e)": -10}
    elif number in ["3_1_crossfeed"]:
        situation_media = {"EX_A(e)": -10, "EX_C(e)": -10}
    elif number in ["4_2_superfluous_crossfeed"]:
        situation_media = {"EX_A(e)": -10, "EX_B(e)": -10, "EX_D(e)": -10}
    elif number in ["4_3_efficient_crossfeed"]:
        situation_media = {"EX_A(e)": -10, "EX_D(e)": -10}

    models = []
    for file_name in situation_models[number]:
        model_path = files("gifba").joinpath("Toy_Models", file_name)
        models.append(cb.io.load_json_model(str(model_path)))
    
    return models, situation_media

def find_min_medium(community=None, models=None, base_media=None, min_growth=None):
    """result = {k: max(dict1.get(k, float('-inf')), dict2.get(k, float('-inf')))
          for k in set(dict1) | set(dict2)}"""
    
    if community is not None:
        if isinstance(community.media, (list)):
            base_media = {ex: np.abs(flux) for ex, flux in community.media[0].items()}
            min_growth = community.media[1]
        
        models = community.models
    else:
        models = models.deepcopy()
        base_media = {ex: np.abs(flux) for ex, flux in base_media.items()}
        min_growth = min_growth if min_growth is not None else 0.1

    min_medium = []
    for model in models:
        with model as model_t:
            for rxn_id, uptake in base_media.items():
                if rxn_id in model_t.exchanges:
                    met = list(model_t.exchanges.get_by_id(rxn_id).metabolites.keys())[0]
                    model_t.add_boundary(met, type="sink", reaction_id=rxn_id+'_tmp',lb=-1*uptake,ub=1000)
                
            for ex in model_t.exchanges:
                ex.lower_bound = -1000
                ex.upper_bound = 1000

            mm = cb.medium.minimal_medium(model_t, min_growth,minimize_components=True)

            model_min_med = mm.to_dict()
            min_medium.append(pd.Series(model_min_med))
    min_medium.append(pd.Series(base_media)) # add base media to ensure all components are included
    
    min_medium = pd.concat(min_medium, axis=1).fillna(0)
    min_medium = (- 1* min_medium.max(axis=1)).to_dict() # convert to uptake and dict

    return min_medium


def check_rel_abund(rel_abund, n_models):
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

def check_iters(iters):
    if iters is None:
        iters = 10
    elif not isinstance(iters, int):
        iters = int(iters)
    if iters < 1:
        iters = 1
        print("Iterations set to:", iters)
    
    return iters

def check_media(community):
    """None, complete, [min, 0.10], dict"""

    # None or "complete" == Set all exchanges to -1000
    community.media = "complete" if community.media is None else community.media
    if isinstance(community.media, str):
        if community.media.lower() == "complete":
            community.media = dict(zip(community.org_exs, np.full(len(community.org_exs), -1000)))
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

def prepare_compartmentalized_model(community, rel_abund=None, obj_rxn_ids=None):
    from cobra import Model
    import cobra as cb

    models = community.models
    media = community.media
    rel_abund = list(community.rel_abund.flatten()) if rel_abund is None else rel_abund #check_rel_abund(rel_abund, community.size)
    community_id = community.id

    # community.create_vars()

    for model_idx, model in enumerate(models):
        # these will be converted to the internal reactions between compartment e1 or e2 moving to e0
        for ex in model.exchanges:
            ex.lower_bound = -1000  # Set lower bound to 0 for all exchange reactions
        for med_ex in media.keys():
            if med_ex in model.reactions:
                model.reactions.get_by_id(med_ex).lower_bound = media[med_ex] / rel_abund[model_idx]


    comp_model = Model("compartmentalized_model_"+str(community_id))
    met_mapping = dict({})
    compartments = []
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
            model_num = int(met_orig.compartment[-1]) -1
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
    objective_rxns_coef = [1 for _ in range(len(models))]
    comp_model.objective = dict(zip(objective_reactions, objective_rxns_coef))

    return comp_model, objective_reactions
        






