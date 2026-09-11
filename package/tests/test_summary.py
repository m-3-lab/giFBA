"""Output-contract tests for ``CommunitySummary``.

These assert *structure* -- index levels and column names -- rather than
numbers. That is deliberate: the two most recent commits to this repo
(``4859d00`` and ``6387c51``, both "modify flux df") were a two-step hotfix to
``summary.py::_build_summary_frames`` in which an index level moved from ``Metabolite`` to
``Exchange`` and the C-number lookup was left reading the old level. The
intermediate state raised at runtime and no test existed to catch it. The three
assertions below would have.
"""

import pandas as pd
import pytest

from golden_values import GOLDEN_ITERS

CASE = "3_1_crossfeed"


@pytest.fixture
def summary(run_toy):
    community, _, _ = run_toy(CASE)
    return community.summarize()


def test_flux_frame_index_and_columns(summary):
    """Per-organism frame is indexed by (Model, Exchange) with element columns.

    Note the asymmetry with total_flux below: this frame puts Exchange in the
    index and Metabolite in a column, while total_flux does the opposite. Both
    conventions are load-bearing for downstream callers, so both are pinned.
    """
    assert list(summary.flux.index.names) == ["Model", "Exchange"]
    assert list(summary.flux.columns) == ["Flux", "Metabolite", "C-Number", "C-Flux"]


def test_total_flux_frame_index_and_columns(summary):
    """Community-level frame is indexed by Metabolite with Exchange as a column."""
    assert list(summary.total_flux.index.names) == ["Metabolite"]
    assert list(summary.total_flux.columns) == ["Exchange", "Flux", "C-Number", "C-Flux"]


def test_element_columns_track_the_element_attribute(run_toy):
    """The element column names follow ``element``, which defaults to carbon."""
    from gifba import CommunitySummary

    community, _, _ = run_toy(CASE)
    nitrogen = CommunitySummary(community, element="N")

    assert "N-Number" in nitrogen.flux.columns
    assert "N-Flux" in nitrogen.flux.columns
    assert "C-Number" not in nitrogen.flux.columns


def test_c_flux_is_c_number_times_absolute_flux(summary):
    """The derived C-Flux column stays consistent with its inputs.

    This is the specific arithmetic that commit 6387c51 had to repair after the
    index level moved out from under it.
    """
    expected = summary.flux["C-Number"] * summary.flux["Flux"].abs()
    pd.testing.assert_series_equal(
        summary.flux["C-Flux"], expected, check_names=False
    )


def test_zero_fluxes_are_dropped(summary):
    """Both frames exclude unused exchanges."""
    assert (summary.flux["Flux"] != 0).all()
    assert (summary.total_flux["Flux"] != 0).all()


def test_objective_values_agree_with_the_run(run_toy):
    """Summary objectives are read straight from org_final, in model order."""
    community, _, org_final = run_toy(CASE)
    summary = community.summarize()

    expected = [
        float(org_final.loc[idx, rxn]) for idx, rxn in community.objective_rxns.items()
    ]
    assert summary.objective_vals == pytest.approx(expected)
    assert summary.objective_total == pytest.approx(sum(expected))


def test_iteration_shown_defaults_to_last_iteration(run_toy):
    community, _, _ = run_toy(CASE)
    assert community.summarize().iteration_shown == GOLDEN_ITERS - 1


def test_to_cytoscape_column_contract(summary):
    """Cytoscape export column names are consumed verbatim by the notebooks.

    Examples/2_real_models/2_1_ahallii_binfantis.ipynb filters on
    edges["Value"] and edges["Target"] and reindexes nodes on nodes["ID"], so
    renaming any of these breaks the published figure pipeline.
    """
    edges, nodes = summary.to_cytoscape()

    assert isinstance(edges, pd.DataFrame) and isinstance(nodes, pd.DataFrame)
    assert list(edges.columns) == ["Source", "Target", "Type", "Value"]
    assert list(nodes.columns) == ["ID", "Name", "Type"]

    assert set(edges["Type"]) <= {"Uptake", "Secretion"}
    assert set(nodes["Type"]) <= {"Organism", "Metabolite"}

    # Value is a magnitude; direction lives in Type.
    assert (edges["Value"] >= 0).all()

    # Every edge endpoint must appear in the node table.
    endpoints = set(edges["Source"]) | set(edges["Target"])
    assert endpoints <= set(nodes["ID"])


def test_renderers_do_not_raise(summary):
    """str() and _repr_html_() are the notebook display path; smoke them only.

    Asserting on rendered text would be brittle -- the convergence wording has
    already changed once (commit b56a1bc) -- so this only checks that both
    produce non-trivial output without raising.
    """
    assert len(str(summary)) > 100
    assert len(summary._repr_html_()) > 100
    assert "Community Summary" in str(summary)
