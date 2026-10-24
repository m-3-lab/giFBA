"""Tests for how ``run_gifba`` interprets ``fp_method`` and ``relaxation_ratio``."""

import pandas as pd
import pytest

CASE = "5_2_dynamical"
N_ITER = 25


def _run(toy_community, **kwargs):
    community = toy_community(CASE)
    community.run_gifba(n_iterations=N_ITER, method="pfba", **kwargs)
    return community


def test_default_is_relaxation_with_ratio_one(toy_community):
    community = _run(toy_community)
    assert community.fp_method == "relaxation"
    assert community.relaxation_ratio == 1.0


def test_default_keeps_user_relaxation_ratio(toy_community):
    community = _run(toy_community, relaxation_ratio=0.5)
    assert community.fp_method == "relaxation"
    assert community.relaxation_ratio == 0.5


def test_default_matches_explicit_relaxation(toy_community):
    implicit = _run(toy_community, relaxation_ratio=0.5)
    explicit = _run(toy_community, relaxation_ratio=0.5, fp_method="relaxation")
    pd.testing.assert_frame_equal(implicit.env_fluxes, explicit.env_fluxes)


def test_picard_overrides_relaxation_ratio(toy_community):
    picard = _run(toy_community, relaxation_ratio=0.5, fp_method="picard")
    default = _run(toy_community)
    assert picard.fp_method == "relaxation"
    assert picard.relaxation_ratio == 1.0
    pd.testing.assert_frame_equal(picard.env_fluxes, default.env_fluxes)


def test_relaxation_ratio_changes_trajectory(toy_community):
    relaxed = _run(toy_community, relaxation_ratio=0.5)
    default = _run(toy_community)
    assert not relaxed.env_fluxes.equals(default.env_fluxes)


def test_anderson_raises_before_solving(toy_community, monkeypatch):
    community = toy_community(CASE)

    def _fail(*args, **kwargs):
        raise AssertionError("solver ran before fp_method was rejected")

    monkeypatch.setattr(community, "_flux_function", _fail)
    monkeypatch.setattr(community, "create_vars", _fail)
    with pytest.raises(NotImplementedError, match="anderson"):
        community.run_gifba(n_iterations=N_ITER, method="pfba", fp_method="anderson")


def test_unknown_fp_method_raises(toy_community):
    community = toy_community(CASE)
    with pytest.raises(ValueError, match="fp_method"):
        community.run_gifba(n_iterations=N_ITER, method="pfba", fp_method="newton")
