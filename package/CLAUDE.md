# Project Instructions: [Your Package Name]

## Overview
Python package for microbial community metabolic modeling, using fixed point problem approach and individual optimization of each microbial model.
Target Python Version: 3.10+

## Common Commands
Run all commands from the repository root.

## Refactoring & Code Guidelines

### General Architecture
- Source code resides in `gifba/`.
- Prefer classes with `gifba_object.py` for understanding per community details

### Typing & Documentation
- Each method and function must have static typehinting 
- Use Google-style docstrings for all public modules, classes, and functions.
- Do not suppress type checker errors with `# type: ignore` unless strictly necessary.

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
