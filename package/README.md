# gifba

A [COBRApy](https://opencobra.github.io/cobrapy/) extension for iterative interaction
flux balance analysis (giFBA) of microbial communities: each organism is optimized
independently against a shared environmental pool, fluxes update the pool, and the
loop repeats to a fixed point (or periodic attractor).

## Installation

```bash
pip install "git+https://github.com/m-3-lab/giFBA.git@main#subdirectory=package"
```

Requires Python 3.10–3.13. Optional extras:

```bash
pip install "gifba[gurobi] @ git+https://github.com/m-3-lab/giFBA.git@main#subdirectory=package"  # licensed Gurobi solver via optlang
pip install "gifba[micom]  @ git+https://github.com/m-3-lab/giFBA.git@main#subdirectory=package"  # MICOM interop helpers in utils.py
```

For local development:

```bash
pip install -e "./package[test,gurobi,micom]"
python -m pytest package/tests
```

## Directory Tree

```
📦package
 ┣ 📂gifba
 ┃ ┣ 📂Toy_Models          10 bundled 2-organism JSON models (non-interacting, competition, cross-feeding, coupling, ...)
 ┃ ┣ 📂publication_utils   Reserved for plotting/comparison helpers (not yet implemented)
 ┃ ┣ __init__.py           Public exports: gifbaObject, CommunitySummary
 ┃ ┣ config.py             Package-wide constants
 ┃ ┣ gifba_object.py       gifbaObject: the fixed-point giFBA simulation engine
 ┃ ┣ summary.py            CommunitySummary: post-run uptake/secretion reporting
 ┃ ┗ utils.py              Input validation, toy-model loader, minimal-medium search, MICOM/cFBA interop
 ┣ 📂tests
 ┃ ┣ conftest.py           Shared fixtures (toy_models, toy_community, run_toy), pinned to GLPK
 ┃ ┣ golden_values.py      Measured reference fluxes for the 10 toy models
 ┃ ┣ test_toy_models.py    End-to-end regression tests over the toy models
 ┃ ┣ test_summary.py       Tests for CommunitySummary
 ┃ ┗ test_validators.py    Tests for the utils.check_* input validators
 ┗ pyproject.toml          Package metadata, dependencies, optional extras
```

## Quickstart

```python
import cobra as cb
from gifba import gifbaObject

model_a = cb.io.load_json_model("organism_a.json")
model_b = cb.io.load_json_model("organism_b.json")

community = gifbaObject([model_a, model_b], media={"EX_glc__D_e": -10}, rel_abund=[0.5, 0.5])
env_final, org_final = community.run_gifba(n_iterations=50, method="pfba")

print(community.summarize())
```

Or load one of the bundled toy models:

```python
from gifba import gifbaObject, utils

models, media = utils.load_simple_models("3_1_crossfeed")
community = gifbaObject(models, media)
env_final, org_final = community.run_gifba(n_iterations=25, method="pfba")
```

---

## API Reference

### `gifba.gifbaObject`

The core simulation object. Wraps a list of `cobra.Model` instances (deep-copied on
construction) and runs the giFBA fixed-point loop over a shared media.

#### `gifbaObject(models, media, rel_abund="equal", **kwargs)`

| Parameter | Type | Description |
|---|---|---|
| `models` | `cobra.Model \| list[cobra.Model]` | The organism model(s) in the community. Deep-copied internally. |
| `media` | `dict[str, float] \| "complete" \| None \| list` | Shared exchange media. Dict of exchange ID → flux (negative = uptake available). `"complete"`/`None` opens every exchange to -1000. A `[base_media, min_growth]` list derives a minimal medium via `utils.find_min_medium`. |
| `rel_abund` | `"equal" \| array-like` | Relative abundance per model. `"equal"` (default) splits evenly; otherwise a 1D array-like of length `num_models`, normalized to sum to 1. |
| `threshold` (kwarg) | `float` | Convergence tolerance. Default `1e-12`. |
| `oc_rounding` (kwarg) | `int` | Decimal places used when checking overconsumption. Default `config.ROUND` (6). |
| `oc_method` (kwarg) | `"optim" \| "newton"` | Overconsumption-correction algorithm. Default `"optim"`. |
| `community_id` (kwarg) | `str \| None` | Optional label for the community, used by `utils.prepare_compartmentalized_model*`. |
| `debug` (kwarg) | `bool` | Verbose per-iteration/per-rerun debug printing. Default `False`. |
| `verbose` (kwarg) | `bool` | Print convergence/overconsumption progress. Default `False`. |

Constructor attributes worth reading directly:

| Attribute | Description |
|---|---|
| `models` | List of the (copied) community models. |
| `media` | Resolved media dict after validation. |
| `num_models` | Number of organisms in the community. |
| `rel_abund` | Resolved relative abundance, shape `(num_models, 1)`. |
| `objective_rxns` | `{model_index: objective_reaction_id}` for each model's linear objective. |
| `simulation_count` | Running count of LP solves that cleared the minimum growth cutoff. |

#### `run_gifba(n_iterations, method, threshold=None, attractor_size=None, relaxation_ratio=None, fp_method=None, v=False, debug=False)`

Runs the iterative simulation to convergence (or `n_iterations`) and returns the
steady-state fluxes. The primary entry point for a simulation.

| Parameter | Type | Description |
|---|---|---|
| `n_iterations` | `int` | Maximum number of iterations to run. |
| `method` | `"pfba" \| "fba"` | Per-organism LP method each iteration. `"pfba"` (parsimonious) is recommended. |
| `threshold` | `float \| None` | Convergence tolerance; overrides the constructor default if given. |
| `attractor_size` | `float \| None` | Fraction of `n_iterations` averaged if convergence is never reached. Default `0.9`. |
| `relaxation_ratio` | `float \| None` | Mixing ratio for the relaxation fixed-point update. `1.0` = plain Picard iteration. Default `1.0`. |
| `fp_method` | `"picard" \| "relaxation" \| "anderson" \| None` | Fixed-point update rule. `"anderson"` is not yet implemented. Default `"relaxation"` (with ratio 1.0, equivalent to Picard). |
| `v` | `bool` | Verbose per-iteration output. |
| `debug` | `bool` | Extra per-iteration/rerun diagnostics (media & flux dumps). Not recommended for large communities. |

Returns `(env_final, org_final)`:
- `env_final: pd.Series` — steady-state (or attractor-averaged) media fluxes, indexed by exchange ID.
- `org_final: pd.DataFrame` — steady-state per-organism fluxes, indexed by model, columns are reaction IDs.

Also populates, after the run:

| Attribute | Description |
|---|---|
| `env_fluxes` | Full media flux history, indexed by `Iteration`, columns = exchange IDs. |
| `org_fluxes` | Full per-organism flux history, indexed by `(Model, Iteration)`, columns = reaction IDs. |
| `iter_converged` | Iteration index where convergence/periodicity was detected, or `None`. |
| `periodicity` | Length of the detected attractor cycle (1 = true fixed point), or `None`. |
| `exchange_ids`, `reaction_ids` | Unique exchange/reaction IDs across the whole community. |
| `exchange_to_metabolite_id`, `metabolite_id_to_name` | Lookup maps used for reporting. |
| `model_names` | `{model_index: model.name}`. |

#### `create_vars(m_vals=[1, 1])`

Allocates the `env_fluxes`/`org_fluxes` storage frames and resets simulation state.
Called automatically at the start of `run_gifba`; only call directly if you need a
freshly-initialized community without running it. `m_vals` is reserved for future
multi-run sampling and should be left at its default.

#### `summarize(iteration_shown=None)`

Returns a [`CommunitySummary`](#gifbacommunitysummary) built from the community's
current final fluxes. `iteration_shown` is cosmetic (labels the report) and defaults
to the last iteration run.

#### `average_periodicity()`

Averages `env_fluxes`/`org_fluxes` over the detected attractor period (or, absent
convergence, over the last `attractor_size` fraction of iterations, with a printed
warning). Called automatically at the end of `run_gifba`; returns `(env_flux_avg,
org_flux_avg)`.

`gifbaObject` also supports use as a context manager (`with gifbaObject(...) as c:`),
which is a no-op passthrough today.

---

### `gifba.CommunitySummary`

Built by `gifbaObject.summarize()`. Formats a community's final fluxes into
human-readable uptake/secretion tables, at both the whole-community and per-organism
level, with optional elemental-flux weighting.

#### `CommunitySummary(community, iteration_shown=None, element="C")`

| Parameter | Type | Description |
|---|---|---|
| `community` | `gifbaObject` | A community that has already been run (via `run_gifba`). |
| `iteration_shown` | `int \| None` | Iteration label shown in the report header. Defaults to the community's last iteration. |
| `element` | `str` | Chemical element (as in `cobra.Metabolite.elements`) used to compute `%`-of-flux breakdowns. Default `"C"` (carbon). |

| Attribute | Description |
|---|---|
| `flux` | Long-form per-organism flux table (`Model`, `Exchange` index), with `Flux`, `Metabolite`, `{element}-Number`, `{element}-Flux` columns. Zero fluxes dropped. |
| `total_flux` | Same, aggregated across the whole community (indexed by `Metabolite`). |
| `objective_rxns` | `{model_index: objective_reaction_id}`, copied from the community. |
| `objective_vals` | Final objective (growth) flux per model, in model order. |
| `objective_total` | Sum of all models' objective fluxes. |
| `method` | The FBA method (`"pfba"`/`"fba"`) the community was run with. |

#### `to_cytoscape()`

Returns `(edges_df, nodes_df)` ready to import into Cytoscape: edges have
`Source`/`Target`/`Type` (`"Uptake"`/`"Secretion"`)/`Value` columns; nodes have
`ID`/`Name`/`Type` (`"Organism"`/`"Metabolite"`) columns.

#### `to_string()`

Returns the full formatted text report (community + per-organism uptake/secretion,
with elemental-flux percentages). `str(summary)` and `repr(summary)` both call this;
in a Jupyter notebook, displaying a `CommunitySummary` renders the HTML/table version
instead (`_repr_html_`).

---

### `gifba.utils`

Validation helpers and standalone utilities. Most `check_*` functions are used
internally by `gifbaObject.__init__`/`run_gifba` but are safe to call directly.

| Function | Description |
|---|---|
| `load_simple_models(case)` | Loads one of the 10 bundled toy-model cases (e.g. `"3_1_crossfeed"`) from `gifba/Toy_Models`. Returns `(models, media)`. |
| `find_min_medium(community=None, models=None, base_media=None, min_growth=None)` | Derives a minimal medium (via `cobra.medium.minimal_medium`) that supports `min_growth` for every model, unioned with `base_media`. Pass either a `community` or explicit `models`/`base_media`/`min_growth`. Returns a media dict of negative (uptake) fluxes. |
| `check_models(models)` | Validates and deep-copies a `cobra.Model` or list of models. Raises `ValueError` on invalid input. |
| `check_media(community)` | Resolves `community.media` (`None`/`"complete"`/dict/minimal-medium list) into a plain flux dict. |
| `check_rel_abund(rel_abund, n_models)` | Validates/normalizes relative abundance into a `(n_models, 1)` array summing to 1. |
| `check_n_iterations(n_iterations)` | Coerces to a positive `int`, defaulting to 10. |
| `check_method(method)` | Validates the FBA method string (`"pfba"`/`"fba"`), defaulting to `"pfba"`. |
| `prep_micom_cfba(community_id, ids, paths, rel_abund=None)` | *Requires the `micom` extra.* Builds a compartmentalized cFBA `cobra.Model` from a MICOM community, plus the underlying `micom.Community` and objective coefficient dict. |
| `prepare_compartmentalized_model(community, rel_abund=None, obj_rxn_ids=None)` | Builds a single compartmentalized cFBA model (each organism in its own compartment, `e0` as the shared pool) from a `gifbaObject`, for comparison against giFBA. Returns `(comp_model, objective_reactions)`. |
| `prepare_compartmentalized_model_with_micom(gifba_community, model_paths, media, rel_abund=None, obj_rxn_ids=None)` | *Requires the `micom` extra.* Same idea as above but built from MICOM's own reaction set. Returns `(comp_model, objective_reactions)`. |

### `gifba.config`

Package-wide constants.

| Constant | Value | Description |
|---|---|---|
| `GROWTH_MIN_OBJ` | `0.01` | Minimum objective (growth) value for an organism's LP solution to be treated as "growing" and contribute flux each iteration. |
| `ROUND` | `6` | Default decimal precision used when checking overconsumption ratios (`gifbaObject`'s `oc_rounding` default). |

## Testing

```bash
python -m pytest package/tests               # full suite (~3s)
python -m pytest package/tests -k validators  # solver-free subset
```

Run from the repository root. `tests/conftest.py` pins the solver to GLPK for
reproducibility; `tests/golden_values.py` holds measured reference fluxes for the 10
toy models — treat changes there as deliberate, and explain in the commit message why
the numbers moved. See `CLAUDE.md` for full contribution conventions.
