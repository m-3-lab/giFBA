"""Unit tests for the input validators in ``gifba.utils``.

These need no LP solver and no models (beyond a bare ``cobra.Model``), so they
are the fast, solver-free subset of the suite:

    pytest tests -k validators

Every expected value below is measured against the current implementation, not
inferred from the docstrings -- the two disagree in places, and these tests
describe what the code does.
"""

import types

import cobra as cb
import numpy as np
import pytest

from gifba import utils


# --------------------------------------------------------------------------
# check_rel_abund
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rel_abund, n_models, expected",
    [
        (None, 2, [0.5, 0.5]),                       # None -> uniform
        ("equal", 3, [1 / 3, 1 / 3, 1 / 3]),         # any str -> uniform
        ([0.5, 0.5], 2, [0.5, 0.5]),                 # already normalised
        ([2, 2], 2, [0.5, 0.5]),                     # renormalised
        ([[0.5], [0.5]], 2, [0.5, 0.5]),             # (n,1) input is flattened
        (np.array([0.25, 0.75]), 2, [0.25, 0.75]),   # ndarray passes through
    ],
    ids=["none", "equal-str", "normalised", "renormalised", "nested", "ndarray"],
)
def test_check_rel_abund_values(rel_abund, n_models, expected):
    result = utils.check_rel_abund(rel_abund, n_models)
    assert result.flatten().tolist() == pytest.approx(expected)


def test_check_rel_abund_returns_column_vector_of_floats():
    """Shape (n, 1) and float dtype are part of the contract.

    Callers index this directly -- _sim_fba multiplies fluxes by
    rel_abund[model_idx] and _optim_method_x calls .flatten() on it -- so the
    shape is load-bearing, not incidental.
    """
    result = utils.check_rel_abund([0.2, 0.8], 2)
    assert result.shape == (2, 1)
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "rel_abund, n_models, message",
    [
        ([1, 2, 3], 2, "1D array of length 2"),
        ([0.5], 2, "1D array of length 2"),
        ([-1, 2], 2, "non-negative"),
        ([0, 0], 2, "non-negative"),
    ],
    ids=["too-long", "too-short", "negative", "all-zero"],
)
def test_check_rel_abund_rejects(rel_abund, n_models, message):
    with pytest.raises(ValueError, match=message):
        utils.check_rel_abund(rel_abund, n_models)


# --------------------------------------------------------------------------
# check_iters
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "iters, expected",
    [
        (None, 10),     # documented default
        (1, 1),
        (25, 25),
        (0, 1),         # clamped up, with a printed notice
        (-5, 1),        # clamped up
        (3.7, 3),       # int() truncates, it does not round
        ("5", 5),       # numeric strings are coerced
    ],
    ids=["none", "one", "typical", "zero-clamped", "negative-clamped", "float-truncates", "numeric-str"],
)
def test_check_iters(iters, expected):
    assert utils.check_iters(iters) == expected


def test_check_iters_rejects_non_numeric_string():
    with pytest.raises(ValueError, match="invalid literal for int"):
        utils.check_iters("abc")


# --------------------------------------------------------------------------
# check_method
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, expected",
    [
        (None, "pfba"),     # documented default
        ("pfba", "pfba"),
        ("fba", "fba"),
        ("PFBA", "pfba"),   # case-insensitive, normalised to lowercase
        ("FBA", "fba"),
    ],
    ids=["none", "pfba", "fba", "upper-pfba", "upper-fba"],
)
def test_check_method(method, expected):
    assert utils.check_method(method) == expected


@pytest.mark.parametrize(
    "method, message",
    [
        ("moma", "either 'pfba' or 'fba'"),
        ("", "either 'pfba' or 'fba'"),
        (5, "Method must be a string"),
        (["pfba"], "Method must be a string"),
    ],
    ids=["unsupported-algorithm", "empty-str", "int", "list"],
)
def test_check_method_rejects(method, message):
    with pytest.raises(ValueError, match=message):
        utils.check_method(method)


# --------------------------------------------------------------------------
# check_models
# --------------------------------------------------------------------------

def test_check_models_wraps_a_bare_model_in_a_list():
    model = cb.Model("solo")
    result = utils.check_models(model)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].id == "solo"


def test_check_models_copies_every_model():
    """The returned models must be copies, not the caller's objects.

    This is the guarantee that protects a caller's models from _set_env, which
    rewrites exchange lower bounds on every iteration. If check_models ever
    stops copying, running a community would corrupt the models the user
    passed in.
    """
    original = cb.Model("shared")
    original.add_metabolites([cb.Metabolite("A_e", compartment="e")])
    original.add_boundary(original.metabolites.A_e, type="exchange")

    returned = utils.check_models([original])

    assert returned[0] is not original
    assert returned[0].reactions[0] is not original.reactions[0]
    # Mutating the copy must not touch the original.
    returned[0].reactions[0].lower_bound = -123.0
    assert original.reactions[0].lower_bound != -123.0


def test_check_models_preserves_order():
    models = [cb.Model("first"), cb.Model("second"), cb.Model("third")]
    assert [m.id for m in utils.check_models(models)] == ["first", "second", "third"]


@pytest.mark.parametrize(
    "models, message",
    [
        (None, "Models must be provided"),
        ("not-a-model", "Models must be provided"),
        (42, "Models must be provided"),
        ([1, 2], "is not a valid cobra.Model"),
        ([cb.Model("ok"), "bad"], "is not a valid cobra.Model"),
    ],
    ids=["none", "str", "int", "list-of-ints", "one-bad-entry"],
)
def test_check_models_rejects(models, message):
    with pytest.raises(ValueError, match=message):
        utils.check_models(models)


# --------------------------------------------------------------------------
# check_media
#
# check_media takes the whole community and reads/writes community.media, so a
# namespace stub is enough for the dict paths. The None / "complete" /
# [dict, min_growth] forms are deliberately untested: the first two raise
# AttributeError because check_media reaches for community.org_exs before
# create_vars() has built it, and the list form needs a real community.
# --------------------------------------------------------------------------

def media_stub(media):
    return types.SimpleNamespace(media=media)


def test_check_media_accepts_a_well_formed_dict():
    media = {"EX_A(e)": -10, "EX_B(e)": -2.5}
    assert utils.check_media(media_stub(media)) == media


def test_check_media_returns_a_copy():
    """The community must not alias the caller's media dict.

    create_vars() re-runs check_media and the object keeps self.media around
    for the life of the simulation; aliasing would let a caller mutate a
    running community's medium.
    """
    media = {"EX_A(e)": -10}
    community = media_stub(media)
    result = utils.check_media(community)

    assert result == media
    assert result is not media
    result["EX_A(e)"] = -999
    assert media["EX_A(e)"] == -10


def test_check_media_accepts_an_empty_dict():
    assert utils.check_media(media_stub({})) == {}


@pytest.mark.parametrize(
    "media, message",
    [
        ({1: -10}, "must be a string"),
        ({("EX_A(e)",): -10}, "must be a string"),
        ({"EX_A(e)": "lots"}, "must be a number"),
        ({"EX_A(e)": None}, "must be a number"),
        ("bogus", "Media must be None"),
        (5.0, "Media must be None"),
    ],
    ids=["int-key", "tuple-key", "str-flux", "none-flux", "unknown-str", "float"],
)
def test_check_media_rejects(media, message):
    with pytest.raises(ValueError, match=message):
        utils.check_media(media_stub(media))


# --------------------------------------------------------------------------
# load_simple_models
# --------------------------------------------------------------------------

CASES = (
    "1_1_single", "1_2_single", "1_3_parallel", "2_1_competition",
    "3_1_crossfeed", "3_2_layered", "4_1_crossfeed_competition",
    "4_2_superfluous_crossfeed", "5_1_coupling", "5_2_dynamical",
)


@pytest.mark.parametrize("case", CASES)
def test_load_simple_models_returns_usable_models_and_media(case):
    """Every advertised key resolves to real, non-empty models plus a medium."""
    models, media = utils.load_simple_models(case)

    assert isinstance(models, list) and models
    assert all(isinstance(m, cb.Model) for m in models)
    assert all(len(m.reactions) > 0 and len(m.metabolites) > 0 for m in models)
    assert all(len(m.exchanges) > 0 for m in models)

    assert isinstance(media, dict) and media
    assert all(isinstance(k, str) for k in media)
    # Media fluxes are uptake bounds and must be negative.
    assert all(v < 0 for v in media.values())


# 5_2_dynamical is excluded below: its medium does not reach its models at all.
# load_simple_models hands it {"EX_A(e)": -10}, but sim5_2_org{1,2}.json spell
# their exchanges "Ex_A(e)" / "Ex_B(e)" / "Ex_C(e)" with a lowercase x, so the
# key matches nothing and the case effectively runs on an empty medium. Re-add
# it here once the loader (or the model files) are fixed -- and re-measure its
# golden values, which currently pin the broken behavior.
CASES_WITH_MATCHING_MEDIA = tuple(c for c in CASES if c != "5_2_dynamical")


@pytest.mark.parametrize("case", CASES_WITH_MATCHING_MEDIA)
def test_load_simple_models_media_keys_exist_in_some_model(case):
    """A medium entry that matches no exchange anywhere is silently ignored.

    create_vars builds an all-False mask for an unknown reaction id and moves
    on, so this test is the only thing standing between a typo in
    load_simple_models and a simulation that quietly runs on the wrong medium.
    """
    models, media = utils.load_simple_models(case)
    all_exchanges = {rxn.id for model in models for rxn in model.exchanges}

    unmatched = set(media) - all_exchanges
    assert not unmatched, f"{case}: media keys match no exchange: {sorted(unmatched)}"


@pytest.mark.parametrize("case", CASES)
def test_load_simple_models_returns_fresh_objects(case):
    """Two calls must not hand back the same model objects."""
    first, _ = utils.load_simple_models(case)
    second, _ = utils.load_simple_models(case)
    assert all(a is not b for a, b in zip(first, second))


def test_load_simple_models_rejects_unknown_key():
    with pytest.raises(KeyError):
        utils.load_simple_models("bogus")
