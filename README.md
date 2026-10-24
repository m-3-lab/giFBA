# giFBA: Greedy Interaction Flux Balance Analysis

giFBA is a [COBRApy](https://opencobra.github.io/cobrapy/) extension for simulating
metabolic exchange in microbial communities. Instead of solving one large
compartmentalized LP for the whole community (cFBA) or a single joint objective
(MICOM), giFBA iterates: each organism is optimized independently against a shared
environmental pool of metabolites, only secreted fluxes update that pool, and the
process repeats until the community reaches a fixed point (or a periodic attractor).

**Documentation:** <https://m-3-lab.github.io/giFBA/> (source in [`docs/`](docs/))

## Installation

```bash
pip install "git+https://github.com/m-3-lab/giFBA.git@main#subdirectory=package"
```

Requires Python 3.10–3.13. Optional extras and a development setup are covered in
[Installation](docs/installation.md).

## Quickstart

```python
import cobra as cb
from gifba import gifbaObject

model_a = cb.io.load_json_model("organism_a.json")
model_b = cb.io.load_json_model("organism_b.json")

community = gifbaObject(
    models=[model_a, model_b],
    media={"EX_glc__D_e": -10},   # negative flux = uptake available in the shared pool
    rel_abund=[0.5, 0.5],
)

env_final, org_final = community.run_gifba(n_iterations=50, method="pfba")
print(community.summarize())
```

See the [API Reference](docs/api.md), [Examples](docs/examples.md), and
[Contributing](docs/contributing.md) for more.

## AI Usage

Parts of this repository — including portions of the code, tests, and this
documentation — were drafted & implemented with the assistance of AI coding tools (Claude Code).
All AI-assisted contributions are reviewed by the project maintainers, and the
underlying scientific method, model validation, and numerical results (e.g. unit 
testing) are checked against expected biological/mathematical behavior rather 
than taken on faith from generated output.

## TODO

### Documentation

- [ ] Review the drafted docstrings in `utils.py` (all functions) and the module
      docstrings in `gifba_object.py`, `summary.py`, `utils.py` and `config.py`.
- [ ] Add static type hints to public signatures. Most types on the API page
      currently come from the docstrings rather than the code.
- [ ] Optionally convert the custom docstring sections in `gifba_object.py`
      ("State Modified", "Calls", "[Flow / Hierarchy]") to standard Google sections.
      They render as plain text blocks on the API page.

### Packaging and deployment

- [ ] `package/pyproject.toml`: the description still says "iterative interaction",
      and the author email is a placeholder.
- [ ] `.github/workflows/docs.yml`: switch the deploy trigger from `review_rs` back to
      `main` after merging, and consider adding `workflow_dispatch` for manual
      redeploys.
- [ ] GitHub Pages: set the source to the `gh-pages` branch at `/`.
