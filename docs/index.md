# giFBA: Greedy Interaction Flux Balance Analysis

giFBA is a [COBRApy](https://opencobra.github.io/cobrapy/) extension for simulating
metabolic exchange in microbial communities. Instead of solving one large
compartmentalized LP for the whole community (cFBA) or a single joint objective
(MICOM), giFBA iterates: each organism is optimized independently against a shared
environmental pool of metabolites, only secreted fluxes update that pool, and the
process repeats until the community reaches a fixed point (or a periodic attractor).

The [repository](https://github.com/m-3-lab/giFBA) contains the `gifba` Python package
(in `package/`) plus the notebooks, validation models, and analysis scripts used to
develop and benchmark it. The comparison against dynamic FBA is described in
[dFBA Benchmark](examples/dfba_benchmark.md).

## Quickstart

```bash
pip install "git+https://github.com/m-3-lab/giFBA.git@main#subdirectory=package"
```

See [Installation](installation.md) for optional extras and a development setup.

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

print(community.summarize())          # per-organism uptake/secretion report
edges, nodes = community.summarize().to_cytoscape()  # export for Cytoscape network viz
```

Or load one of the bundled toy models:

```python
from gifba import gifbaObject, utils

models, media = utils.load_simple_models("3_1_crossfeed")
community = gifbaObject(models, media)
env_final, org_final = community.run_gifba(n_iterations=25, method="pfba")
```

## Package overview

- **`gifba.gifbaObject`**: build a community from a list of `cobra.Model` objects
  and a shared media, then call `.run_gifba()` to simulate to a fixed point.
- **`gifba.CommunitySummary`**: returned by `gifbaObject.summarize()`. Formats
  per-organism and community-level uptake/secretion, and exports Cytoscape node/edge
  tables.
- **`gifba.utils`**: validation/coercion helpers used by `gifbaObject`, the bundled
  toy-model loader (`load_simple_models`), minimal-medium search
  (`find_min_medium`), and optional MICOM/compartmentalized-model interop.

Full details are in the [API Reference](api.md).
