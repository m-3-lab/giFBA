# gifba

A [COBRApy](https://opencobra.github.io/cobrapy/) extension for greedy interaction
flux balance analysis (giFBA) of microbial communities: each organism is optimized
independently against a shared environmental pool, fluxes update the pool, and the
loop repeats to a fixed point (or periodic attractor).

```bash
pip install "git+https://github.com/m-3-lab/giFBA.git@main#subdirectory=package"
```

Requires Python 3.10–3.13. Installation options, a quickstart, and the full API
reference are in the documentation: <https://m-3-lab.github.io/giFBA/>
