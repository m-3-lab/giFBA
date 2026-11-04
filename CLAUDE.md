# Greedy Interaction Flux Balance Analysis (giFBA)

Python 3.10+ package for microbial community metabolic modeling: a fixed-point approach that optimizes each member model individually.

## Layout
- `package/gifba/`: source. Prefer the classes in `gifba_object.py` for per-community details.
- `Examples/1_toy_models/`: example usage on toy community models.
- `Examples/2_real_models/`: synthetic community interactions on real bacterial models.
- `Examples/2_real_models/4_dfba_benchmark/`: scripts, data/results and analysis comparing giFBA with a modified dynamic FBA. Figures go to `Examples/figures/real_model_figs/dfba_benchmark/`. Leave the paths in `vacc_scripts/VACC_*.py` alone (they run on the cluster), and keep benchmark-only dependencies out of `package/`.

## Finding code
Before any grep, `cat`, `find` or Read on source code, query graphify first: `graphify_find` for a named symbol, `query_graph` for "how/where does X work", `graphify_callers`/`graphify_callees`, `graphify_file_neighbors`, `graphify_tests_for`. Then read only the line range it points to (`offset`/`limit`), never the whole file; `gifba_object.py` alone is ~14k tokens. Pass the same instruction to any subagents. Fall back to targeted grep only when graphify returns nothing, is unavailable, or the code is uncommitted (the index is a snapshot), and say which fallback applied.

## Commands
Run from the repository root. The exception is the dFBA benchmark, which runs from its own directory in the `M3` env:

```bash
conda run -n M3 python -m pytest package/tests              # full suite, ~3s
conda run -n M3 python -m pytest package/tests -k validators # solver-free subset
conda run -n M3 python -m pytest package/tests -x -q         # stop on first failure
cd Examples/2_real_models/4_dfba_benchmark && conda run -n M3 python process_results.py
```

## Code Style
- Static type hints on every function and method.
- Google-style docstrings for all public modules, classes and functions.
- Use `# type: ignore` only when strictly necessary.
- 4-space indentation.
- Write like an experienced human developer: no conversational transitions, robotic summaries or textbook-style structural remarks.
- Comment only complex, non-obvious logic.

## Testing
- Tests are in `package/tests/`. CI (`.github/workflows/tests.yml`) runs the suite with GLPK on Python 3.10 and 3.13 for every push and PR.
- `golden_values.py` holds measured reference values for all ten toy models. Re-measure rather than hand-edit, and say in the commit message why the numbers moved.
- `conftest.py` pins the LP solver to GLPK. Keep the pin: the golden values were measured under GLPK, and CI has no Gurobi license.
- New tests cover behaviour that works today. Track known defects separately, not as `xfail`.

## Dependencies
- Prefer the standard library over new external dependencies.
- Required: cobrapy, numpy, pandas, scipy. The core `run_gifba` path needs nothing else and runs on the GLPK solver that ships with cobrapy.
- Optional extras: `gifba[gurobi]` (faster solver, needs a license) and `gifba[micom]` (only for the lazily imported `utils.prep_micom_cfba` and `utils.prepare_compartmentalized_model_with_micom`). Install them with `pip install -e "./package[test,gurobi,micom]"`.
