# Examples

The example notebooks live in the
[`Examples/`](https://github.com/m-3-lab/giFBA/tree/main/Examples) directory of the
repository. Figures and network diagrams they generate are saved to
`Examples/figures/`.

## Toy models

Walkthroughs of giFBA on the ten bundled two-organism toy models. Each case name
can also be loaded directly with `utils.load_simple_models(case)`.

| Interaction | Notebook |
|---|---|
| Non-interacting | [1_1_single](https://github.com/m-3-lab/giFBA/blob/main/Examples/1_toy_models/1_non_interacting/1_1_single.ipynb), [1_2_single](https://github.com/m-3-lab/giFBA/blob/main/Examples/1_toy_models/1_non_interacting/1_2_single.ipynb), [1_3_parallel](https://github.com/m-3-lab/giFBA/blob/main/Examples/1_toy_models/1_non_interacting/1_3_parallel.ipynb) |
| Competition | [2_1_competition](https://github.com/m-3-lab/giFBA/blob/main/Examples/1_toy_models/2_competition/2_1_competition.ipynb) |
| Cross-feeding | [3_1_crossfeed](https://github.com/m-3-lab/giFBA/blob/main/Examples/1_toy_models/3_crossfeeding/3_1_crossfeed.ipynb), [3_2_layered](https://github.com/m-3-lab/giFBA/blob/main/Examples/1_toy_models/3_crossfeeding/3_2_layered.ipynb) |
| Combined interactions | [4_1_crossfeed_competition](https://github.com/m-3-lab/giFBA/blob/main/Examples/1_toy_models/4_combined_interactions/4_1_crossfeed_competition.ipynb), [4_2_superfluous_crossfeed](https://github.com/m-3-lab/giFBA/blob/main/Examples/1_toy_models/4_combined_interactions/4_2_superfluous_crossfeed.ipynb) |
| Coupling | [5_1_coupling](https://github.com/m-3-lab/giFBA/blob/main/Examples/1_toy_models/5_coupling/5_1_coupling.ipynb), [5_2_dynamical](https://github.com/m-3-lab/giFBA/blob/main/Examples/1_toy_models/5_coupling/5_2_dynamical.ipynb) |

Bifurcation analyses for the coupling cases are in
[`5_coupling/bifurcation_analysis/`](https://github.com/m-3-lab/giFBA/tree/main/Examples/1_toy_models/5_coupling/bifurcation_analysis).

## Real models

Notebooks applying giFBA to AGORA2 communities:

- [2_1_ahallii_binfantis](https://github.com/m-3-lab/giFBA/blob/main/Examples/2_real_models/2_1_ahallii_binfantis.ipynb):
  *A. hallii* / *B. infantis* community
- [2_2_ahallii_binfantis_network_visualization](https://github.com/m-3-lab/giFBA/blob/main/Examples/2_real_models/2_2_ahallii_binfantis_network_visualization.ipynb):
  network visualization of the same community
- [3_cdiff_community](https://github.com/m-3-lab/giFBA/blob/main/Examples/2_real_models/3_cdiff_community.ipynb): *C. difficile* community
- [N_AGORA_Models](https://github.com/m-3-lab/giFBA/blob/main/Examples/2_real_models/N_AGORA_Models.ipynb)

The AGORA2 models are large binaries fetched separately and are not version
controlled.

## dFBA benchmark

A comparison of giFBA against a clamped-biomass dynamic FBA implementation. See
[dFBA Benchmark](examples/dfba_benchmark.md).
