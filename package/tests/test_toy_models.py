"""End-to-end regression tests over the ten bundled toy models.

These are the suite's load-bearing tests. They do not check that giFBA is
*correct* -- that is a modelling question -- but they pin exactly what it
computes today, so that refactoring the fixed-point solver in
``gifba_object.py`` cannot silently move a single flux.
"""

import pandas as pd
import pytest

from golden_values import GOLDEN_ENV, GOLDEN_ITERS, GOLDEN_RUNS, TOY_CASES

# Golden values are exactly representable today, but compare with a tolerance
# anyway: a solver or convergence-threshold change should fail with a readable
# numeric delta rather than an inscrutable float mismatch.
TOL = 1e-9


def objectives_of(community, org_final):
    """Per-model objective flux, ordered by model index."""
    return [
        float(org_final.loc[model_idx, rxn_id])
        for model_idx, rxn_id in community.objective_rxns.items()
    ]


@pytest.mark.parametrize("case", TOY_CASES)
def test_convergence_matches_golden(run_toy, case):
    """Convergence iteration, period, and FBA call count are unchanged."""
    expected_iter, expected_period, expected_sim_ct, _ = GOLDEN_RUNS[case]
    community, _, _ = run_toy(case)

    assert community.iter_converged == expected_iter
    assert community.periodicity == expected_period
    # simulation_ct counts LP solves that cleared GROWTH_MIN_OBJ. It jumps if
    # the overconsumption re-run loop changes behavior, which makes it a cheap
    # canary for _check_overconsumption regressions.
    assert community.simulation_ct == expected_sim_ct


@pytest.mark.parametrize("case", TOY_CASES)
def test_objectives_match_golden(run_toy, case):
    """Per-organism biomass flux at steady state is unchanged."""
    *_, expected_objectives = GOLDEN_RUNS[case]
    community, _, org_final = run_toy(case)

    assert objectives_of(community, org_final) == pytest.approx(
        expected_objectives, abs=TOL
    )


@pytest.mark.parametrize("case", TOY_CASES)
def test_env_fixed_point_matches_golden(run_toy, case):
    """The environment's fixed point is unchanged, metabolite by metabolite."""
    expected = GOLDEN_ENV[case]
    _, env_final, _ = run_toy(case)

    assert set(env_final.index) == set(expected)
    for exchange, flux in expected.items():
        assert float(env_final[exchange]) == pytest.approx(flux, abs=TOL), exchange


@pytest.mark.parametrize("case", TOY_CASES)
def test_return_shapes(run_toy, case):
    """run_gifba's return contract: (pd.Series, pd.DataFrame) with known axes."""
    community, env_final, org_final = run_toy(case)

    assert isinstance(env_final, pd.Series)
    assert set(env_final.index) == set(community.org_exs)

    assert isinstance(org_final, pd.DataFrame)
    assert org_final.index.name == "Model"
    assert list(org_final.index) == list(range(community.size))
    assert set(org_final.columns) == set(community.org_rxns)


@pytest.mark.parametrize("case", TOY_CASES)
def test_stored_flux_frames_drop_run_level(run_toy, case):
    """The 'Run' index level is dropped by run_gifba.

    Callers depend on this: compare_clamped_dfba.ipynb slices
    ``org_fluxes.loc[(slice(None), slice(0, iter_converged - 1)), :]``, which
    silently returns the wrong thing if a third level reappears.
    """
    community, _, _ = run_toy(case)

    assert list(community.org_fluxes.index.names) == ["Model", "Iteration"]
    assert list(community.env_fluxes.index.names) == ["Iteration"]

    # One row per (model, iteration); env carries an extra row for iteration 0.
    assert len(community.org_fluxes) == community.size * GOLDEN_ITERS
    assert len(community.env_fluxes) == GOLDEN_ITERS + 1

    # The documented slicing idiom still works.
    sliced = community.org_fluxes.sort_index().loc[
        (slice(None), slice(0, community.iter_converged - 1)), :
    ]
    assert not sliced.empty


def test_rerunning_a_community_is_deterministic(toy_community):
    """A second run_gifba on the same object reproduces the first exactly.

    run_gifba mutates the exchange bounds of the community's models via
    _set_env, so a second run starts from already-modified bounds. That it
    still lands on the same fixed point is what makes the golden values above
    trustworthy rather than an artifact of call ordering.
    """
    community = toy_community("3_1_crossfeed")

    first_env, first_org = community.run_gifba(iters=GOLDEN_ITERS, method="pfba")
    # Copy: run_gifba reassigns these attributes, but be explicit about it.
    first_env, first_org = first_env.copy(), first_org.copy()
    second_env, second_org = community.run_gifba(iters=GOLDEN_ITERS, method="pfba")

    pd.testing.assert_series_equal(first_env, second_env)
    pd.testing.assert_frame_equal(first_org, second_org)


def test_fresh_community_reproduces_same_fixed_point(toy_community):
    """Two independently constructed communities agree."""
    first = toy_community("3_1_crossfeed")
    second = toy_community("3_1_crossfeed")

    first_env, first_org = first.run_gifba(iters=GOLDEN_ITERS, method="pfba")
    second_env, second_org = second.run_gifba(iters=GOLDEN_ITERS, method="pfba")

    pd.testing.assert_series_equal(first_env, second_env)
    pd.testing.assert_frame_equal(first_org, second_org)


@pytest.mark.parametrize("case", ["1_1_single", "3_1_crossfeed", "5_2_dynamical"])
def test_fba_and_pfba_agree_on_objectives(run_toy, case):
    """"fba" and "pfba" find the same growth rates on the toy models.

    pfba additionally minimises total flux, so internal distributions may
    differ, but the objective values should not. Restricted to three
    representative cases -- including the period-4 attractor -- to keep the
    matrix small.
    """
    *_, expected_objectives = GOLDEN_RUNS[case]
    community, _, org_final = run_toy(case, method="fba")

    assert objectives_of(community, org_final) == pytest.approx(
        expected_objectives, abs=TOL
    )


@pytest.mark.parametrize("case", TOY_CASES)
def test_exchange_bounds_stay_scalar(run_toy, case):
    """_set_env must write plain floats into cobra bounds, not 1-element arrays.

    Boolean-indexing _env_scaling_factors naturally produces a (1,) array. If
    that reaches ``ex.lower_bound``, cobra stores it and later calls isinf() on
    it -- a DeprecationWarning on numpy 1.x and a hard TypeError on numpy >= 2,
    which makes run_gifba fail on iteration 0. This test is the guard on the
    .item() call in _set_env.
    """
    community, _, _ = run_toy(case)

    for model in community.models:
        for rxn in model.exchanges:
            assert isinstance(rxn.lower_bound, float), (
                f"{model.id}.{rxn.id}.lower_bound is "
                f"{type(rxn.lower_bound).__name__}, not float"
            )
            assert isinstance(rxn.upper_bound, float)
