# Installation

```bash
pip install "git+https://github.com/m-3-lab/giFBA.git@main#subdirectory=package"
```

Requires Python 3.10–3.13. Core dependencies (`cobra`, `numpy`, `pandas`, `scipy`)
are installed automatically and give you the full `run_gifba` path on cobra's
bundled GLPK solver.

## Optional extras

```bash
# Faster LP solver via optlang (needs a Gurobi license)
pip install "gifba[gurobi] @ git+https://github.com/m-3-lab/giFBA.git@main#subdirectory=package"

# MICOM interop: utils.prep_micom_cfba / utils.prepare_compartmentalized_model_with_micom
pip install "gifba[micom] @ git+https://github.com/m-3-lab/giFBA.git@main#subdirectory=package"
```

`micom` is imported lazily inside those two functions only, and `gurobipy` is never
imported by the package itself; it just lets optlang find Gurobi.

## Development install

Clone the repo and install editable with the test extras (add `gurobi,micom` if you
need them):

```bash
git clone https://github.com/m-3-lab/giFBA.git
cd giFBA
pip install -e "./package[test,gurobi,micom]"
python -m pytest package/tests
```

See [Contributing](contributing.md) for testing conventions.
