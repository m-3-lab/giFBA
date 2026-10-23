# Project Instructions: Greedy Interaction Flux Balance Analysis

## Overview
Python package for microbial community metabolic modeling, using fixed point problem approach and individual optimization of each microbial model.
Target Python Version: 3.10+

## Common Commands
Run all commands from the repository root.

## Target Analysis
Review `diff_method/` directory to aide analysis of the modified dynamic FBA and giFBA techniques. 
- Scripts for running the code on the UVM VACC are located in `diff_method/Benchmark_Scripts`
- Results for each set of benchmarks are stored in `diff_method/Benchmark_analyses`
- Code/notebooks for analysis and comparing results in `diff_method/Benchmark_analyses`


## Refactoring & Code Guidelines

### General Architecture
- Source code resides in `gifba/`.
- Prefer classes with `gifba_object.py` for understanding per community details
- Example Code Usage with Toy Community Metabolic Models resides in `Examples/1_toy_models/`
- Application to synthetic community interactions on real bacterial models are located in `/Examples/2_real_models/`
- Benchmark code (Scripts for running benchmark, data/results, and analysis code) used for comparison of giFBA to a modified dynamic FBA method are in `diff_method`

### Typing & Documentation
- **Static Typehinting:** Each method and function must have static typehinting 
- **Docstring Styles:** Use Google-style docstrings for all public modules, classes, and functions.
- **Error Suppression:** Do not suppress type checker errors with `# type: ignore` unless strictly necessary.
- **Indentation Style:** use 4-space/tab indentation
- **Human-First Tone:** Write code, comments, and documentation as if written by an experienced human developer. 
- **No AI Filler:** Completely avoid conversational transitions, robotic summaries, or textbook-style structural remarks.
- **Concise Comments:** Only write comments for complex, non-obvious logic. Never restate what the code clearly expresses.

### Testing & Verification Standards
Tests live in `package/tests/` and run in ~3s. Run from the repository root:

```bash
conda run -n gifba_test python -m pytest package/tests              # full suite
conda run -n gifba_test python -m pytest package/tests -k validators # solver-free subset
conda run -n gifba_test python -m pytest package/tests -x -q         # stop on first failure
```

CI (`.github/workflows/tests.yml`) runs the same suite on Python 3.10 and 3.13
with GLPK on every push and pull request.

- `tests/golden_values.py` holds measured reference values for all ten toy
  models. Treat a change there as a deliberate act: re-measure rather than
  hand-edit, and say in the commit message why the numbers moved.
- `tests/conftest.py` pins the LP solver to GLPK. The golden values were
  measured under GLPK, and CI has no Gurobi license, so do not remove the pin.
- New tests should cover behaviour that works today. Known defects are tracked
  separately rather than encoded as `xfail`.

### Dependencies
- Prefer standard library solutions where possible before adding new external dependencies.
- Required: cobrapy, numpy, pandas, scipy. The core `run_gifba` path needs
  nothing else and runs on the GLPK solver that cobrapy vendors.
- Optional extras: `gifba[gurobi]` (faster solver, needs a license) and
  `gifba[micom]` (only for `utils.prep_micom_cfba` and
  `utils.prepare_compartmentalized_model_with_micom`, both lazily imported).
  Install with `pip install -e "./package[test,gurobi,micom]"`.
