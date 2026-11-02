"""Shared fixtures for the gifba test suite.

The whole suite is fast: all ten toy models converge in well under a second
each, so nothing here needs caching or a session scope beyond convenience.
"""

import pytest

import gifba
from gifba import utils

# Pinned so results are reproducible across environments: the gifba_test conda
# env defaults to Gurobi and CI has no Gurobi license, but cobra ships GLPK via
# swiglpk everywhere. The golden values in golden_values.py were measured under
# GLPK; they have been verified to be identical under Gurobi, so the pin buys
# reproducibility rather than papering over a solver disagreement. Keep it
# anyway -- it means a future solver-dependent regression shows up as a test
# failure rather than as drifting numbers.
SOLVER = "glpk"


@pytest.fixture
def toy_models():
    """Factory returning ``(models, media)`` for a load_simple_models key.

    The models come back solver-pinned and are fresh per call, so a test may
    mutate them freely.
    """
    def _load(case):
        models, media = utils.load_simple_models(case)
        for model in models:
            model.solver = SOLVER
        return models, media

    return _load


@pytest.fixture
def toy_community(toy_models):
    """Factory returning an unrun ``gifbaObject`` for a load_simple_models key.

    Extra keyword arguments are forwarded to the ``gifbaObject`` constructor.
    Note that ``gifbaObject`` deep-copies the models it is handed, so the
    solver pinning applied by ``toy_models`` is what the copies inherit.
    """
    def _build(case, **kwargs):
        models, media = toy_models(case)
        return gifba.gifbaObject(models, media, **kwargs)

    return _build


@pytest.fixture
def run_toy(toy_community):
    """Factory that builds a community, runs it, and returns everything.

    Returns ``(community, env_final, org_final)``. Defaults match the
    conditions under which the golden values were measured.
    """
    def _run(case, n_iterations=None, method="pfba", **kwargs):
        from golden_values import GOLDEN_ITERS

        community = toy_community(case, **kwargs)
        env_final, org_final = community.run_gifba(
            n_iterations=GOLDEN_ITERS if n_iterations is None else n_iterations,
            method=method,
        )
        return community, env_final, org_final

    return _run
