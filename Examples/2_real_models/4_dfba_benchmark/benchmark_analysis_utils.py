"""Shared helpers for analyzing giFBA vs. dFBA benchmark results.

These functions back ``process_results.py`` and the ``4_1``/``4_2``
analysis notebooks, which load, bin, and compare giFBA/dFBA benchmark JSON
results across experiment families (e.g. the fixed two-organism "AH_BI"
community and the variable 1-10 organism "N_orgs" communities).

Everything here is organism-count-agnostic except where noted (the
``get_org_index``/``build_org_color_map``/``get_org_label``/``add_flux_traces``
family, which exists specifically to replace a hardcoded 2-organism "AH"/"BI"
split with one that generalizes to any N).
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import plotly.graph_objects as go
import seaborn as sns
import tol_colors as tol


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinSpec:
    """Specification for discretizing one continuous hyperparameter.

    Attributes:
        spread: "linear" or "log" spacing between bin edges.
        bins: Number of bins (bin edges = ``bins + 1``).
        min: Literal lower bound for the bin edges. When ``None`` (the
            default), the lower bound is derived from the observed data for
            the current run instead of a hardcoded literal.
        max: Literal upper bound for the bin edges, with the same
            data-driven-when-``None`` behavior as ``min``.
        fmt_bins: Format spec used to render each bin edge in labels.
        zero_floor: When ``True``, force the lowest bin edge to exactly 0.0
            after computing the edges (used for ``y0_factor``/``y0_offset``,
            whose values are only meaningfully nonzero when a metabolite is
            present in the media).
    """

    spread: str
    bins: int
    min: float | None = None
    max: float | None = None
    fmt_bins: str = "{:.2g}"
    zero_floor: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """One named benchmark experiment (a results directory + its settings).

    Attributes:
        name: Short experiment label, used as the output-filename suffix so
            different experiments never clobber each other's figures.
        results_dir: Name of the results directory under ``data/raw/``,
            holding one ``*.json`` file per run.
        max_time_s: Wall-clock seconds substituted for ``dfba_solve_time`` on
            runs that were only checkpointed mid-progress (status
            ``dfba_in_progress_*``) rather than completed.
        binning: Per-hyperparameter bin specifications. Keys absent from a
            given experiment's result schema (e.g. ``y0_factor``/``y0_offset``
            for "N_orgs") are silently skipped rather than raising.
    """

    name: str
    results_dir: str
    max_time_s: float = 24 * 3600 # max wall-clock seconds to substitute for dfba_solve_time on in-progress runs
    binning: Mapping[str, BinSpec] = field(default_factory=dict)


#: Display name (LaTeX-ish label prefix) for every hyperparameter/metadata
#: column this analysis bins or groups by.
VARIABLE_NAMES: dict[str, str] = {
    "vmax": r"$V_{max}$",
    "km": r"$K_m$",
    "early_stop_window": r"$N_{LR}$",
    "early_stop_tol": r"$tol_{LR}$",
    "rel_abund": r"$\alpha_0$",
    "y0_factor": r"$C_0$ Scaling",
    "y0_offset": r"$C_0$ Offset",
    "N_models": r"$N_{models}$",
    "solver_method": "Solver Method",
}

#: Numeric encoding of `solver_method`. Its key order is also the plotting
#: order, so BDF is always the first hue level.
SOLVER_CODES: dict[str, int] = {"BDF": 0, "RK45": 1}

#: Fixed `solver_method` colors from the Tol "bright" palette, so a solver
#: keeps its color regardless of which one appears first in the data.
SOLVER_COLORS: dict[str, str] = {"BDF": tol.bright.blue, "RK45": tol.bright.red}

#: Fixed giFBA/dFBA colors for the method-comparison violin row, also from Tol
#: "bright" but distinct from `SOLVER_COLORS`. Key order is the plotting order.
METHOD_COLORS: dict[str, str] = {"giFBA": tol.bright.green, "dFBA": tol.bright.cyan}

#: Bin spread/count per hyperparameter, shared by every experiment preset.
#: Edges are data-driven (``min``/``max`` left as ``None``) rather than
#: hardcoded literals, since the actual Monte Carlo sampling ranges already
#: differ between experiments and drift over time as generator scripts change.
DEFAULT_BINNING: dict[str, BinSpec] = {
    "vmax": BinSpec(spread="log", bins=2, fmt_bins="{:.0f}"),
    "km": BinSpec(spread="log", bins=2, fmt_bins="{:.1f}"),
    "early_stop_window": BinSpec(spread="linear", bins=3, fmt_bins="{:.0f}"),
    "early_stop_tol": BinSpec(spread="log", bins=3, fmt_bins="{:.1e}"),
    "rel_abund": BinSpec(spread="linear", bins=3, fmt_bins="{:.2f}"),
    "y0_factor": BinSpec(spread="log", bins=2, fmt_bins="{:.2f}", zero_floor=True),
    "y0_offset": BinSpec(spread="log", bins=2, fmt_bins="{:.2f}", zero_floor=True),
}

#: Known experiment presets. Both currently share `DEFAULT_BINNING`; a future
#: experiment with its own hyperparameter ranges can override `binning`.
PRESETS: dict[str, ExperimentConfig] = {
    "AH_BI": ExperimentConfig(
        name="AH_BI", results_dir="Benchmark_AH_BI", binning=DEFAULT_BINNING
    ),
    "N_orgs": ExperimentConfig(
        name="N_orgs", results_dir="Benchmark_N_orgs", binning=DEFAULT_BINNING
    ),
}

#: Columns kept on the giFBA side of the efficiency comparison.
GIFBA_COLS: list[str] = [
    "N_models",
    "sum_N_rxns",
    "sum_N_mets",
    "rel_abund",
    "log10_time_ratio",
    "log10_pfba_call_ratio",
    "gifba_solve_time",
    "gifba_pfba_calls",
]

#: Columns kept on the dFBA side of the efficiency comparison (hyperparameters
#: plus their binned counterparts).
DFBA_COLS: list[str] = [
    "N_models",
    "sum_N_rxns",
    "sum_N_mets",
    "rel_abund",
    "rel_abund_binned",
    "y0_factor",
    "y0_factor_binned",
    "y0_offset",
    "y0_offset_binned",
    "vmax",
    "vmax_binned",
    "km",
    "km_binned",
    "solver_method",
    "early_stop_window",
    "early_stop_window_binned",
    "early_stop_tol",
    "early_stop_tol_binned",
    "log10_time_ratio",
    "log10_pfba_call_ratio",
    "dfba_solve_time",
    "dfba_pfba_calls",
]

#: Groupings for the violin+strip efficiency plots: each entry is
#: ``[N_models]`` or ``[N_models, subgroup_binned_col]``.
VARS_EVAL: list[list[str]] = [
    ["N_models"],
    ["N_models", "y0_factor_binned"],
    ["N_models", "y0_offset_binned"],
    ["N_models", "solver_method"],
    ["N_models", "vmax_binned"],
    ["N_models", "km_binned"],
    ["N_models", "early_stop_window_binned"],
    ["N_models", "early_stop_tol_binned"],
    ["N_models", "rel_abund_binned"],
]

#: Hyperparameter/metric columns whose Best-MAE-vs-Worst-MAE density is
#: plotted; looked up from the already-joined best/worst-MAE tables (so both
#: raw hyperparameters and derived equivalence metrics resolve the same way).
KDE_PARAM_COLS: list[str] = [
    "vmax",
    "km",
    "rel_abund",
    "y0_factor",
    "y0_offset",
    "early_stop_window",
    "early_stop_tol",
    "N_nonzero_rxns",
    "gifba_iter_converged",
    "gifba_periodicity",
    "dfba_pfba_calls",
    "gifba_pfba_calls",
    "solver_method",
    "gifba_abs_flux_sum",
    "dfba_abs_flux_sum",
]

#: Axis scale ("linear"/"log") for every column in the pairwise scatter grid.
PAIRWISE_SCALE_COLS: dict[str, str] = {
    "dfba_pfba_calls": "log",
    "y0_factor": "log",
    "y0_offset": "log",
    "vmax": "log",
    "km": "log",
    "early_stop_window": "linear",
    "early_stop_tol": "log",
    "rel_abund": "linear",
    "log10_time_ratio": "linear",
    "log10_pfba_call_ratio": "linear",
    "dfba_solve_time": "log",
    "dfba_termination_t": "log",
    "solver_method": "linear",
}

#: Curated column subset for displaying best/worst-MAE run tables, excluding
#: the raw `gifba_fluxes`/`dfba_fluxes` nested-dict columns (which are
#: unreadable when displayed directly).
DISPLAY_COLS: list[str] = [
    "N_models",
    "solver_method",
    "vmax",
    "km",
    "early_stop_window",
    "early_stop_tol",
    "rel_abund",
    "y0_factor",
    "y0_offset",
    "gifba_solve_time",
    "dfba_solve_time",
    "log10_time_ratio",
    "log10_pfba_call_ratio",
    "gifba_pfba_calls",
    "dfba_pfba_calls",
    "gifba_iter_converged",
    "gifba_periodicity",
    "dfba_termination_t",
    "N_rxns",
    "N_nonzero_rxns",
    "gifba_abs_flux_sum",
    "dfba_abs_flux_sum",
    "MAE_all",
    "RPE_all",
]


def present(cols: Sequence[str], df: pd.DataFrame) -> list[str]:
    """Filters a column-name list down to the ones actually in `df`.

    Used everywhere a hyperparameter/metric list is iterated, since the set of
    logged fields differs by experiment (e.g. "N_orgs" never logs
    `y0_factor`/`y0_offset`).

    Args:
        cols: Candidate column names, in the order they should be kept.
        df: DataFrame to check membership against.

    Returns:
        The subset of `cols` that are columns of `df`, preserving order.
    """
    return [c for c in cols if c in df.columns]


# ---------------------------------------------------------------------------
# Loading and column derivation
# ---------------------------------------------------------------------------


def load_and_filter_results(results_dir: Path, max_time_s: float) -> pd.DataFrame:
    """Loads every benchmark result JSON in a directory into one DataFrame.

    Keeps only runs that reached `dfba_complete` or were checkpointed
    mid-progress (`dfba_in_progress_*`, treated as having timed out at
    `max_time_s`), then adds the efficiency-ratio columns
    `log10_time_ratio`/`log10_pfba_call_ratio`.

    Args:
        results_dir: Directory of `*.json` benchmark result files.
        max_time_s: Wall-clock seconds to substitute for `dfba_solve_time` on
            runs that were still in progress when checkpointed.

    Returns:
        One row per completed/in-progress run, indexed by file stem.
    """
    results = {
        file.stem: json.loads(file.read_text()) for file in results_dir.glob("*.json")
    }
    df = pd.DataFrame.from_dict(results, orient="index")

    complete_mask = df["status"] == "dfba_complete"
    in_progress_mask = df["status"].str.startswith("dfba_in_progress_")
    df = df[complete_mask | in_progress_mask]

    # in_progress_mask still has the pre-filter index; .loc aligns it to the
    # (smaller) filtered df by label, so this only ever touches kept rows.
    df.loc[in_progress_mask, "dfba_solve_time"] = max_time_s

    df["log10_time_ratio"] = np.log10(df["dfba_solve_time"] / df["gifba_solve_time"])
    df["log10_pfba_call_ratio"] = np.log10(
        df["dfba_pfba_calls"] / df["gifba_pfba_calls"]
    )
    return df


def compute_rel_abund_metrics(rel_abund: pd.Series, config_name: str) -> pd.Series:
    """Summarizes each run's relative-abundance vector as its dominant fraction.

    Replaces the fixed-2-organism convention of reading off organism 0's share
    (`rel_abund[0]`) with `max(rel_abund)`. This stays meaningful for any
    community size N (trivially 1.0 at N=1) and captures the same "how skewed
    is this community" signal the original two-organism analysis used
    `rel_abund[0]` for.

    Args:
        rel_abund: Per-run relative-abundance vectors (lists of floats
            summing to ~1), or already-missing/non-list values.

    Returns:
        Per-run dominant abundance fraction; NaN where `rel_abund` is not a
        non-empty list.
    """
    if config_name == "AH_BI":
        col_name = "rel_abund_org_0"
        return col_name, rel_abund.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
    else:
        if config_name != "N_orgs":
            raise ValueError(f"Unknown config_name {config_name!r}; expected 'AH_BI' or 'N_orgs'.")
        col_names = ["min_rel_abund", "max_rel_abund", "mean_rel_abund", "std_rel_abund"]
        rel_abund_df = pd.DataFrame(rel_abund.tolist(), index=rel_abund.index)
        return col_names, rel_abund_df.agg(["min", "max", "mean", "std"], axis=1)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def get_bin_borders(spec: BinSpec, observed: pd.Series | None = None) -> np.ndarray:
    """Computes bin edges for one hyperparameter.

    Args:
        spec: Bin specification (spread, bin count, optional literal bounds).
        observed: That hyperparameter's observed values for the current run,
            used to derive bin edges when `spec.min`/`spec.max` are `None`.

    Returns:
        Array of `spec.bins + 1` bin edges.

    Raises:
        ValueError: If a bound is unspecified and no `observed` data was
            given, or if `spec.spread` is neither "linear" nor "log".
    """
    lo, hi = spec.min, spec.max
    if lo is None or hi is None:
        if observed is None:
            raise ValueError(
                "BinSpec has no explicit min/max and no observed data was given."
            )
        clean = observed.dropna()
        if lo is None:
            lo = float(clean.min())
        if hi is None:
            hi = float(clean.max())

    if spec.spread == "linear":
        bins = np.linspace(lo, hi, spec.bins + 1)
    elif spec.spread == "log":
        # log10(0) is undefined; nudge a zero/negative lower bound up to the
        # smallest representable positive float just for spacing the edges.
        lo_safe = lo if lo > 0 else np.nextafter(0, 1)
        bins = np.logspace(np.log10(lo_safe), np.log10(hi), spec.bins + 1)
    else:
        raise ValueError(f"Unknown spread {spec.spread!r}; expected 'linear' or 'log'.")

    if spec.zero_floor:
        bins[0] = 0.0
    return bins


def get_bin_labels(variable_name: str, bins: np.ndarray, fmt: str = "{:.2g}") -> list[str]:
    """Builds human-readable interval labels for a set of bin edges.

    Args:
        variable_name: Display name prefixed to each label (e.g. `r"$V_{max}$"`).
        bins: Bin edges, as returned by `get_bin_borders`.
        fmt: Format spec applied to each edge value.

    Returns:
        One label per bin, e.g. `r"$V_{max}$\\in$[1, 10]"`.
    """
    return [
        variable_name + r"$\in$" + f"[{fmt.format(bins[i])}, {fmt.format(bins[i + 1])}]"
        for i in range(len(bins) - 1)
    ]


def bin_hyperparameters(
    df: pd.DataFrame,
    binning: Mapping[str, BinSpec],
    variable_names: Mapping[str, str],
) -> pd.DataFrame:
    """Adds a `{key}_binned` categorical column for each hyperparameter present.

    Keys in `binning` that are not columns of `df` (e.g. `y0_factor`/
    `y0_offset` for the "N_orgs" experiment, which never samples them) are
    silently skipped rather than raising, since the set of logged
    hyperparameters differs by experiment. A key that *is* present but was
    held constant for every run in this experiment (e.g. `early_stop_window`
    for "N_orgs", which fixes it at 20 rather than sampling it) is also
    skipped: a single observed value has no meaningful bin boundaries, and
    `pd.cut` would otherwise raise on the resulting duplicate edges.

    Args:
        df: Loaded, filtered results table.
        binning: Per-hyperparameter bin specifications.
        variable_names: Per-hyperparameter display name, used as the label
            prefix; falls back to the raw key when a name isn't registered.

    Returns:
        A copy of `df` with one additional `{key}_binned` column per key that
        is present in `df.columns` and varies across at least 2 bin edges.
    """
    df = df.copy()
    for key in present(list(binning.keys()), df):
        spec = binning[key]
        # Skip hyperparameters this experiment held constant (e.g. N_orgs
        # fixes early_stop_window=20) - a single value gives duplicate bin
        # edges, which pd.cut rejects below.
        if df[key].dropna().nunique() <= 1:
            continue
        bins = get_bin_borders(spec, observed=df[key])
        if len(np.unique(bins)) < len(bins):
            continue
        labels = get_bin_labels(variable_names.get(key, key), bins, spec.fmt_bins)
        df[f"{key}_binned"] = pd.cut(df[key], bins=bins, labels=labels, include_lowest=True)
    return df


# ---------------------------------------------------------------------------
# Flux parsing and equivalence metrics
# ---------------------------------------------------------------------------


def parse_simulation_series(s: pd.Series, fill_value: float = 0.0) -> pd.DataFrame:
    """Flattens a Series of nested per-reaction, per-organism flux dicts.

    Each entry of `s` is expected to be a dict `{rxn_id: {org_idx: flux}}` (as
    stored in the `gifba_fluxes`/`dfba_fluxes` result fields). Builds the
    output column-by-column as a sparse-backed DataFrame so a large, mostly-
    zero flux matrix is never densely allocated.

    Args:
        s: Series of nested flux dicts, one per simulation run.
        fill_value: Value used for reaction/organism pairs absent in a run.

    Returns:
        DataFrame indexed like `s`, with one column `"{rxn_id}___{org_idx}"`
        per (reaction, organism) pair seen across any run. The organism index
        is always separated from the reaction id by exactly 3 underscores.
    """
    col_data: dict[str, dict[object, float]] = {}

    for row_idx, sim_data in s.items():
        if isinstance(sim_data, dict):
            for rxn_id, org_dict in sim_data.items():
                if isinstance(org_dict, dict):
                    for org_idx, flux in org_dict.items():
                        col_data.setdefault(f"{rxn_id}___{org_idx}", {})[row_idx] = flux

    sparse_dtype = pd.SparseDtype(float, fill_value=fill_value)
    sparse_cols = {
        col_name: pd.arrays.SparseArray(
            [val_dict.get(idx, fill_value) for idx in s.index], dtype=sparse_dtype
        )
        for col_name, val_dict in col_data.items()
    }
    return pd.DataFrame(sparse_cols, index=s.index)


def calc_equivalence_metrics(
    gifba_fluxes_df: pd.DataFrame,
    dfba_fluxes_df: pd.DataFrame,
    n_rxns_series: pd.Series,
    mask_threshold: float = 1e-6,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Calculates per-run equivalence metrics between giFBA and dFBA fluxes.

    Organism-agnostic: operates on whole flattened flux rows (every
    reaction x organism column at once), so it needs no changes for any
    community size N.

    Args:
        gifba_fluxes_df: giFBA fluxes, one row per run, columns as produced
            by `parse_simulation_series` (sorted to align with `dfba_fluxes_df`).
        dfba_fluxes_df: dFBA fluxes, same shape/row-and-column order as
            `gifba_fluxes_df`.
        n_rxns_series: Per-run total reaction count (e.g. `df["sum_N_rxns"]`),
            reindexed to match `gifba_fluxes_df`'s index.
        mask_threshold: Minimum absolute flux difference to count a reaction
            as meaningfully "different" between the two methods.

    Returns:
        A 3-tuple:
            - `equivalence_df`: per-run `Abs_Diff_Sum`, `Sum_Rel_Error`,
              `N_nonzero_rxns`, `N_rxns`, `Sum_Abs_Flux`, `gifba_abs_flux_sum`,
              `dfba_abs_flux_sum`.
            - `global_counts`: a contiguous histogram of log10(|flux|)
              magnitudes across every nonzero flux value seen (both methods).
            - `all_bin_edges`: the bin edges matching `global_counts`.
    """
    n_rxns_series = n_rxns_series.reindex(gifba_fluxes_df.index)

    equivalence_df = pd.DataFrame(
        index=gifba_fluxes_df.index,
        columns=["Abs_Diff_Sum", "Sum_Rel_Error", "N_nonzero_rxns", "N_rxns"],
    )

    counts_dict: dict[int, int] = {}
    for (row, gifba_fluxes), (_, dfba_fluxes), n_rxns in zip(
        gifba_fluxes_df.iterrows(), dfba_fluxes_df.iterrows(), n_rxns_series
    ):
        # Accumulate a running log10(|flux|) magnitude histogram across every
        # nonzero flux value seen in either method, for this run.
        gifba_vals = gifba_fluxes[gifba_fluxes != 0].values
        dfba_vals = dfba_fluxes[dfba_fluxes != 0].values
        combined_fluxes = np.concatenate([gifba_vals, dfba_vals])

        log_vals = np.log10(np.abs(combined_fluxes))
        bin_keys = np.floor(log_vals).astype(int)
        unique_keys, counts = np.unique(bin_keys, return_counts=True)
        for key, count in zip(unique_keys, counts):
            counts_dict[key] = counts_dict.get(key, 0) + count

        abs_diff = (gifba_fluxes - dfba_fluxes).abs().round(12)
        diff = gifba_fluxes - dfba_fluxes

        non_zero_mask = abs_diff >= mask_threshold
        gifba_zeros_mask = gifba_fluxes == 0
        dfba_zeros_mask = dfba_fluxes == 0

        if non_zero_mask.sum() > 0:
            abs_diff_sum = abs_diff.sum()
            # Symmetric relative error (avoids blowing up near zero-valued
            # reactions): skip any reaction where both methods agree it's
            # off (zero in both), since 0/0 there is undefined, not "wrong".
            inf_mask = non_zero_mask | (dfba_zeros_mask & gifba_zeros_mask)
            rpe = 2 * diff[~inf_mask] / (
                gifba_fluxes.abs()[~inf_mask] + dfba_fluxes.abs()[~inf_mask]
            )
            relative_error_sum = np.array(rpe).sum()
            number_nonzero = non_zero_mask.sum()
        else:
            abs_diff_sum = 0.0
            relative_error_sum = 0.0
            number_nonzero = 0

        equivalence_df.loc[row, "Abs_Diff_Sum"] = abs_diff_sum
        equivalence_df.loc[row, "Sum_Rel_Error"] = relative_error_sum
        equivalence_df.loc[row, "N_nonzero_rxns"] = number_nonzero
        equivalence_df.loc[row, "N_rxns"] = n_rxns
        equivalence_df.loc[row, "Sum_Abs_Flux"] = (gifba_fluxes.abs() + dfba_fluxes.abs()).sum()
        equivalence_df.loc[row, "gifba_abs_flux_sum"] = gifba_fluxes.abs().sum()
        equivalence_df.loc[row, "dfba_abs_flux_sum"] = dfba_fluxes.abs().sum()

    min_bin, max_bin = min(counts_dict), max(counts_dict)
    all_bin_edges = np.arange(min_bin, max_bin + 2, 1)
    global_counts = np.array([counts_dict.get(b, 0) for b in range(min_bin, max_bin + 1)])
    return equivalence_df, global_counts, all_bin_edges


# ---------------------------------------------------------------------------
# Organism-aware plotting (the N-organism generalization)
# ---------------------------------------------------------------------------


def get_org_index(column_name: str) -> str:
    """Extracts the organism index from a parsed flux-column name.

    Args:
        column_name: A column name produced by `parse_simulation_series`, of
            the form `"{rxn_id}___{org_idx}"` (always exactly 3 underscores
            between the reaction id and the organism index).

    Returns:
        The organism index as a string, matching the keys of the original
        `gifba_fluxes`/`dfba_fluxes` dicts (e.g. "0", "1", ...).
    """
    return column_name.rsplit("___", 1)[1]


def build_org_color_map(org_indices: Sequence[str], palette: Sequence[str]) -> dict[str, str]:
    """Assigns one color per organism index, cycling a palette.

    Args:
        org_indices: Organism index strings (as returned by `get_org_index`).
        palette: Colors to cycle through (e.g. `tol_colors.bright`).

    Returns:
        Mapping from organism index to color, ordered by the index's integer
        value so index "10" sorts after "2" rather than before it.
    """
    unique_sorted = sorted(set(org_indices), key=int)
    if hasattr(palette, 'colors'):
        colors = palette.colors
        colors = [matplotlib.colors.to_hex(c) for c in colors]
    else:
        colors = palette
    cycled = itertools.cycle(colors)
    return {org: next(cycled) for org in unique_sorted}


def get_org_label(model_names_row: Sequence[str] | None, org_idx: str) -> str:
    """Builds a legend label for one organism in one run.

    Args:
        model_names_row: That run's `model_names` list (species/model
            filenames), or `None`/shorter-than-expected if unavailable.
        org_idx: The organism's index as a string.

    Returns:
        The species name with a trailing ".mat" stripped, or `f"org {org_idx}"`
        if `model_names_row` has no entry for `org_idx`.
    """
    i = int(org_idx)
    if model_names_row is not None and i < len(model_names_row):
        name = model_names_row[i]
        return name[:-4] if name.endswith(".mat") else name
    return f"org {org_idx}"


def add_flux_traces(
    fig: go.Figure,
    gifba_s: pd.Series,
    dfba_s: pd.Series,
    model_names_row: Sequence[str] | None,
    color_map: Mapping[str, str],
    legend: str,
    col: int,
) -> tuple[float, float]:
    """Adds one best/worst-example flux-comparison panel to the 3D-scatter figure.

    Draws one giFBA-vs-dFBA scatter trace per organism that actually belongs
    to this run (not a fixed 2-organism pair, and not every org index that
    appears anywhere in the shared flux frame). `gifba_s`/`dfba_s` come from a
    "common columns" frame built across every run in the experiment, so for a
    variable-N experiment ("N_orgs") a run with fewer than the maximum N
    organisms still has columns for the higher org indices other runs used -
    always exactly 0 for this run, since `parse_simulation_series` fills
    absent (reaction, organism) pairs with 0. Those filler columns are
    skipped here via `model_names_row`'s length, which is this run's actual
    organism count.

    Args:
        fig: The multi-subplot Plotly figure to add traces to (row 1 is used).
        gifba_s: giFBA flux values for one run, indexed by
            `"{rxn_id}___{org_idx}"` column names.
        dfba_s: dFBA flux values for the same run, same index as `gifba_s`.
        model_names_row: That run's `model_names` list; also used to limit
            plotted organisms to `range(len(model_names_row))`. If `None`,
            every org index present in `gifba_s` is plotted (no way to tell
            real organisms from filler columns in that case).
        color_map: Organism index -> color, from `build_org_color_map`.
        legend: Name of the Plotly legend group this panel's traces join.
        col: Subplot column (1-indexed) to draw into.

    Returns:
        The `(min, max)` bounds of the y=x reference line, so the caller can
        pin both axes of this panel to the same range for a 1:1 aspect ratio.
    """
    org_ids = pd.Index(gifba_s.index).map(get_org_index)
    present_org_idx = sorted(set(org_ids), key=int)
    if model_names_row is not None:
        present_org_idx = [o for o in present_org_idx if int(o) < len(model_names_row)]

    for org_idx in present_org_idx:
        mask = np.asarray(org_ids == org_idx)
        name = get_org_label(model_names_row, org_idx)
        fig.add_trace(
            go.Scatter(
                x=gifba_s[mask],
                y=dfba_s[mask],
                mode="markers",
                name=name,
                text=gifba_s.index[mask],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "giFBA: %{x:.4g}<br>"
                    "dFBA: %{y:.4g}"
                    f"<extra>{name}</extra>"
                ),
                marker=dict(color=color_map[org_idx], opacity=0.6),
                legend=legend,
            ),
            row=1,
            col=col,
        )

    min_y_x = min(gifba_s.min(), dfba_s.min())
    max_y_x = max(gifba_s.max(), dfba_s.max())
    fig.add_trace(
        go.Scatter(
            x=[min_y_x, max_y_x],
            y=[min_y_x, max_y_x],
            mode="lines",
            name="y=x",
            line=dict(color="rgba(0, 0, 0, 0.5)", dash="dash"),
            legend=legend,
        ),
        row=1,
        col=col,
    )
    # add x/y=0 reference lines for context
    fig.add_shape( 
        type="line", line_color="black", line_width=1.5, opacity=0.2,
        x0=0, x1=0, xref="x", y0=min_y_x, y1=max_y_x, yref="y", row=1, col=col
    )    
    fig.add_shape(
        type="line", line_color="black", line_width=1.5, opacity=0.2,
        x0=min_y_x, x1=max_y_x, xref="x", y0=0, y1=0, yref="y", row=1, col=col
    )

    fig.update_xaxes(title_text="giFBA Fluxes", range=(min_y_x, max_y_x), showgrid=True, row=1, col=col)
    fig.update_yaxes(title_text="dFBA Fluxes", range=(min_y_x, max_y_x), showgrid=True, row=1, col=col)
    return min_y_x, max_y_x


# ---------------------------------------------------------------------------
# Efficiency violin plots
# ---------------------------------------------------------------------------


def melt_method_costs(results: pd.DataFrame, n_models_col: str = "N_models") -> pd.DataFrame:
    """Stacks the giFBA and dFBA solve costs into one long table.

    Args:
        results: Results table carrying `n_models_col` and the
            `{gifba,dfba}_solve_time` and `{gifba,dfba}_pfba_calls` columns.
        n_models_col: Community-size column to carry over.

    Returns:
        Two rows per run (one per method) with columns `n_models_col`,
        `method` ("giFBA"/"dFBA"), `log10_solve_time` and `log10_pfba_calls`.
    """
    return pd.concat(
        [
            pd.DataFrame({
                n_models_col: results[n_models_col],
                "method": label,
                "log10_solve_time": np.log10(results[f"{prefix}_solve_time"]),
                "log10_pfba_calls": np.log10(results[f"{prefix}_pfba_calls"]),
            })
            for prefix, label in (("gifba", "giFBA"), ("dfba", "dFBA"))
        ],
        ignore_index=True,
    )


def plot_subgrouped(
    results: pd.DataFrame,
    groupby_cols: Sequence[str],
    row: int,
    axs: np.ndarray,
    y_min: float | None = None,
    y_max: float | None = None,
    y_scale: str = "linear",
    compare_methods: bool = False,
) -> None:
    """Draws one row (2 panels) of violin+strip plots of solve efficiency.

    By default plots `log10_time_ratio` and `log10_pfba_call_ratio`, each
    grouped by `N_models` on the x-axis and optionally sub-grouped (hue) by a
    second column. With `compare_methods`, plots log10 solve time and log10
    pFBA calls instead, sub-grouped by method (giFBA vs. dFBA). Draws into
    `axs[row*2]`/`axs[row*2 + 1]`.

    Args:
        results: Results table carrying the value columns and
            `groupby_cols`. With `compare_methods` it must hold both the
            giFBA and dFBA solve-time and pFBA-call columns.
        groupby_cols: `[n_models_col]` or `[n_models_col, subgroup_col]`.
            Only `[n_models_col]` is allowed with `compare_methods`.
        row: Which row of the overall figure grid to draw into.
        axs: Flattened array of all the figure's axes.
        y_min: Lower y-axis limit for both panels; `None` keeps matplotlib's.
        y_max: Upper y-axis limit for both panels; `None` keeps matplotlib's.
        y_scale: Matplotlib y-axis scale ("linear" or "log").
        compare_methods: Plot raw giFBA vs. dFBA costs instead of their ratios.

    Raises:
        ValueError: If `compare_methods` is combined with a subgroup column.
    """
    n_models_col = groupby_cols[0]

    if compare_methods:
        if len(groupby_cols) > 1:
            raise ValueError("compare_methods subgroups by method; pass only [n_models_col].")
        data = melt_method_costs(results, n_models_col)
        value_cols = ["log10_solve_time", "log10_pfba_calls"]
        y_labels = [r"$\log_{10}(\text{Solve Time [s]})$", r"$\log_{10}(\text{pFBA Calls})$"]
        subgroup_col = "method"
        hue_order = list(METHOD_COLORS)
        violin_palette = METHOD_COLORS
    else:
        data = results
        value_cols = ["log10_time_ratio", "log10_pfba_call_ratio"]
        y_labels = [r"$\log_{10}(\text{Time Ratio})$", r"$\log_{10}(\text{pFBA Calls Ratio})$"]
        subgroup_col = groupby_cols[1] if len(groupby_cols) > 1 else None
        hue_order = list(SOLVER_CODES) if subgroup_col == "solver_method" else None
        violin_palette = SOLVER_COLORS if subgroup_col == "solver_method" else None
    has_subgroup = subgroup_col is not None

    
    for i, value_col in enumerate(value_cols):
        ax = axs[row * 2 + i]
        y_label = y_labels[i]

        sns.violinplot(
            data=data,
            x=n_models_col,
            y=value_col,
            hue=subgroup_col if has_subgroup else None,
            hue_order=hue_order,
            palette=violin_palette,
            ax=ax,
            inner="quartile",
            cut=0,
            zorder=2,
        )

        stripplot_kwargs = {
            "data": data,
            "x": n_models_col,
            "y": value_col,
            "hue": subgroup_col if has_subgroup else None,
            "hue_order": hue_order,
            "dodge": has_subgroup,
            "jitter": 0.05,
            "color": "black",
            "alpha": 0.2,
            "size": 3,
            "ax": ax,
            "zorder": 3,
        }

        if has_subgroup:
            n_subgroups = data[subgroup_col].nunique()
            stripplot_kwargs["palette"] = ["black"] * n_subgroups

        sns.stripplot(**stripplot_kwargs)

        if i == 1 and ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_yscale(y_scale)
        ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
        ax.set_ylim(y_min, y_max)
        ax.margins(y=0.05)

        if len(np.unique(data[n_models_col])) > 1:
            ax.set_xlabel("# of models")
            ax.tick_params(labelbottom=True, labelrotation=0)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
            ax.set_xticks([])
            
        ax.set_ylabel(y_label)

        if i == 1:
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(True)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.tick_params(labelright=True)
        else:
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(True)
            ax.tick_params(labelleft=True)

        ax.spines["top"].set_visible(False)

