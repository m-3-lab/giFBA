"""Tests for the standalone helpers in ``gifba.utils``."""

from importlib.resources import files

import cobra as cb
from cobra.util.solver import linear_reaction_coefficients
import pytest

import gifba
from gifba import utils


def _model_state(model: cb.Model) -> list[tuple]:
    """Snapshot what the helpers under test used to mutate: IDs, compartments, bounds."""
    mets = [(met.id, met.compartment) for met in model.metabolites]
    rxns = [(rxn.id, rxn.lower_bound, rxn.upper_bound) for rxn in model.reactions]
    return mets + rxns


def test_load_simple_models_4_2_media(toy_models) -> None:
    models, media = toy_models("4_2_superfluous_crossfeed")
    assert media == {"EX_A(e)": -10, "EX_B(e)": -10}
    exchanges = {rxn.id for model in models for rxn in model.exchanges}
    assert set(media) <= exchanges


def test_find_min_medium_with_model_list(toy_models) -> None:
    models, _ = toy_models("4_1_crossfeed_competition")
    before = [_model_state(model) for model in models]

    medium = utils.find_min_medium(models=models, base_media={"EX_A(e)": -10}, min_growth=0.1)

    assert medium == {"EX_A(e)": -10.0}
    assert [_model_state(model) for model in models] == before


def test_find_min_medium_with_single_model(toy_models) -> None:
    models, _ = toy_models("1_1_single")
    medium = utils.find_min_medium(models=models[0], min_growth=0.1)
    assert set(medium) == {"EX_A(e)"}
    assert medium["EX_A(e)"] < 0


def test_find_min_medium_matches_community_path(toy_models) -> None:
    models, _ = toy_models("4_1_crossfeed_competition")
    community = gifba.gifbaObject(models, [{"EX_A(e)": -10}, 0.1])
    assert community.media == utils.find_min_medium(
        models=models, base_media={"EX_A(e)": -10}, min_growth=0.1
    )


def test_find_min_medium_needs_models() -> None:
    with pytest.raises(ValueError):
        utils.find_min_medium()


@pytest.mark.parametrize("case", ["3_1_crossfeed", "4_2_superfluous_crossfeed"])
def test_prepare_compartmentalized_model_leaves_community_untouched(toy_community, case: str) -> None:
    community = toy_community(case, rel_abund=[0.3, 0.7])
    before = [_model_state(model) for model in community.models]

    utils.prepare_compartmentalized_model(community, rel_abund=community.rel_abund)

    assert [_model_state(model) for model in community.models] == before


def test_prepare_compartmentalized_model_is_repeatable(toy_community) -> None:
    community = toy_community("3_1_crossfeed")
    obj_ids = ["EX_Bio(e1)", "EX_Bio(e2)"]

    first, first_obj = utils.prepare_compartmentalized_model(community, obj_rxn_ids=obj_ids)
    second, second_obj = utils.prepare_compartmentalized_model(community, obj_rxn_ids=obj_ids)

    assert [rxn.id for rxn in first_obj] == obj_ids
    assert sorted(rxn.id for rxn in first.reactions) == sorted(rxn.id for rxn in second.reactions)
    assert first.slim_optimize() == pytest.approx(second.slim_optimize())


def test_prepare_compartmentalized_model_scales_by_abundance(toy_community) -> None:
    community = toy_community("3_1_crossfeed", rel_abund=[0.25, 0.75])
    comp_model, _ = utils.prepare_compartmentalized_model(community)

    # Each organism's exchange carries its share of the shared e0 pool.
    ex1 = comp_model.reactions.get_by_id("EX_A(e1)")
    assert ex1.metabolites[comp_model.metabolites.get_by_id("A[e1]")] == pytest.approx(-4.0)
    assert ex1.lower_bound == pytest.approx(-10 / 0.25)



def test_prepare_compartmentalized_model_more_than_nine_organisms(toy_models) -> None:
    models = []
    for i in range(11):
        model = toy_models("1_1_single")[0][0]
        model.id = f"org{i + 1}"
        models.append(model)
    rel_abund = [i + 1 for i in range(11)]
    community = gifba.gifbaObject(models, {"EX_A(e)": -10}, rel_abund=rel_abund)

    comp_model, _ = utils.prepare_compartmentalized_model(community)

    # e11 must map to organism 11, not organism 1 from its last digit.
    abund = community.rel_abund.flatten()
    for idx in (0, 9, 10):
        ex = comp_model.reactions.get_by_id(f"EX_A(e{idx + 1})")
        met = comp_model.metabolites.get_by_id(f"A[e{idx + 1}]")
        assert ex.metabolites[met] == pytest.approx(-1 / abund[idx])

def test_prepare_compartmentalized_model_with_micom(toy_community) -> None:
    pytest.importorskip("micom")
    case = "4_2_superfluous_crossfeed"
    community = toy_community(case)
    before = [_model_state(model) for model in community.models]
    paths = [str(files("gifba").joinpath("Toy_Models", f"sim4_2_org{i}.json")) for i in (1, 2)]

    comp_model, obj = utils.prepare_compartmentalized_model_with_micom(
        community, paths, {"EX_A(e)": -5}, rel_abund=[0.2, 0.8]
    )

    assert [rxn.id for rxn in obj] == [f"{community.objective_rxns[i]}__Org{i+1}" for i in range(2)]
    coefs = linear_reaction_coefficients(comp_model)
    assert coefs[obj[0]] == pytest.approx(0.2)
    assert coefs[obj[1]] == pytest.approx(0.8)
    assert comp_model.reactions.get_by_id("EX_A_m").lower_bound == -5
    assert comp_model.reactions.get_by_id("EX_B_m").lower_bound == 0
    assert [_model_state(model) for model in community.models] == before
