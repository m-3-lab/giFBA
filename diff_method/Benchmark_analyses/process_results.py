"""Preprocesses raw giFBA vs. dFBA benchmark JSON results into cached files.

Does the expensive part of ``benchmark_analysis_dfba_general.ipynb`` (loading
every result JSON, flattening the nested flux dicts, and computing the
equivalence metrics) once, and writes three files per experiment to
``{name}_data/``:

    model_names_{name}.csv   per-run model names, lists stored as JSON strings
    hyperparams_{name}.csv   per-run hyperparameters, run stats, and MAE/RPE
                             (list/dict cells stored as JSON strings)
    fluxes_{name}.pkl        common giFBA/dFBA flux frames + flux histogram

Usage:
    python process_results.py                  # every preset
    python process_results.py --config AH_BI   # one or more named presets
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

import benchmark_analysis_utils as bau

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "raw_data"

NESTED_COLS = ["rel_abund_list", "media"]
DROPPED_COLS = ["gifba_fluxes", "dfba_fluxes", "error", "status", "dfba_success", "dfba_message"]


def add_rel_abund_columns(df: pd.DataFrame, config_name: str) -> pd.DataFrame:
    """Replaces the `rel_abund` vectors with scalar summaries.

    The original vectors are kept in `rel_abund_list`. For "N_orgs" the
    min/max/mean/std columns are added and `rel_abund` becomes the minimum.

    Args:
        df: Loaded results table.
        config_name: Experiment preset name.

    Returns:
        `df` with the relative-abundance columns updated.
    """
    rel_abund_cols, rel_abund_metrics = bau.compute_rel_abund_metrics(df["rel_abund"], config_name)
    if config_name == "N_orgs":
        rel_abund_metrics = rel_abund_metrics.rename(
            columns=dict(zip(rel_abund_metrics.columns, rel_abund_cols))
        )
    df["rel_abund_list"] = df["rel_abund"].copy()
    if "std_rel_abund" in rel_abund_cols:
        rel_abund_metrics["std_rel_abund"] = rel_abund_metrics["std_rel_abund"].fillna(0.0)
        df = pd.concat([df, rel_abund_metrics], axis=1)
        df["rel_abund"] = df["min_rel_abund"]
    else:
        df["rel_abund"] = rel_abund_metrics
        df = df.rename(columns={"rel_abund": "rel_abund_org_0"})
    return df


def build_common_fluxes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flattens both flux columns and drops reactions inactive in every run.

    Args:
        df: Results table with `gifba_fluxes`/`dfba_fluxes` nested-dict columns.

    Returns:
        giFBA and dFBA flux frames with matching, sorted rows and columns.
    """
    gifba_fluxes_df = bau.parse_simulation_series(df["gifba_fluxes"])
    dfba_fluxes_df = bau.parse_simulation_series(df["dfba_fluxes"])

    all_nan_or_zero_gifba = gifba_fluxes_df.columns[(gifba_fluxes_df.fillna(0) == 0).all()]
    all_nan_or_zero_dfba = dfba_fluxes_df.columns[(dfba_fluxes_df.fillna(0) == 0).all()]
    inactive_in_both = set(all_nan_or_zero_gifba).intersection(all_nan_or_zero_dfba)

    gifba_common = gifba_fluxes_df.drop(columns=inactive_in_both, errors="ignore")
    dfba_common = dfba_fluxes_df.drop(columns=inactive_in_both, errors="ignore")
    return (
        gifba_common.sort_index(axis=0).sort_index(axis=1),
        dfba_common.sort_index(axis=0).sort_index(axis=1),
    )


def process_experiment(config: bau.ExperimentConfig) -> None:
    """Processes one experiment's raw results and writes its three output files.

    Args:
        config: Experiment preset to process.
    """
    results_dir = RAW_DATA_DIR / config.results_dir
    df = bau.load_and_filter_results(results_dir, config.max_time_s)
    
    print(f"[{config.name}] loaded {len(df)} runs from {results_dir}")

    df = add_rel_abund_columns(df, config.name)
    model_names = df.pop("model_names").sort_index()

    gifba_common, dfba_common = build_common_fluxes(df)
    eq_df, counts, bin_edges = bau.calc_equivalence_metrics(gifba_common, dfba_common, df["sum_N_rxns"])
    eq_df["MAE"] = (eq_df["Abs_Diff_Sum"] / eq_df["N_rxns"]).astype(float)
    eq_df["RPE"] = (eq_df["Sum_Rel_Error"] / eq_df["N_rxns"]).astype(float)

    hyperparams_df = df.drop(columns=DROPPED_COLS).sort_index().join(eq_df)
    hyperparams_df["RPE_abs"] = hyperparams_df["RPE"].abs()
    for col in bau.present(NESTED_COLS, hyperparams_df):
        hyperparams_df[col] = hyperparams_df[col].map(json.dumps)

    out_dir = BASE_DIR / f"{config.name}_data"
    out_dir.mkdir(exist_ok=True)
    model_names_path = out_dir / f"model_names_{config.name}.csv"
    hyperparams_path = out_dir / f"hyperparams_{config.name}.csv"
    fluxes_path = out_dir / f"fluxes_{config.name}.pkl"

    model_names.map(json.dumps).to_csv(model_names_path, index_label="run_id")
    hyperparams_df.to_csv(hyperparams_path, index_label="run_id")
    with fluxes_path.open("wb") as f:
        pickle.dump(
            {
                "gifba": gifba_common,
                "dfba": dfba_common,
                "flux_hist_counts": counts,
                "flux_hist_bin_edges": bin_edges,
            },
            f,
        )

    print(f"[{config.name}] wrote {model_names_path.name}, {hyperparams_path.name}, {fluxes_path.name} to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        nargs="+",
        choices=sorted(bau.PRESETS),
        default=sorted(bau.PRESETS),
        help="Experiment preset(s) to process (default: all).",
    )
    args = parser.parse_args()

    for name in args.config:
        process_experiment(bau.PRESETS[name])


if __name__ == "__main__":
    main()
