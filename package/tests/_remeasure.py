"""Re-measure the golden values in ``golden_values.py``.

Run this only when a golden value has *intentionally* changed -- after fixing
one of the loader bugs, for example. It prints Python literals ready to paste
back into ``golden_values.py``; it does not edit anything itself, so the diff
stays under review.

    conda run -n gifba_test python package/tests/_remeasure.py
"""

import gifba
from gifba import utils

ITERS = 25
SOLVER = "glpk"
CASES = (
    "1_1_single", "1_2_single", "1_3_parallel", "2_1_competition",
    "3_1_crossfeed", "3_2_layered", "4_1_crossfeed_competition",
    "4_2_superfluous_crossfeed", "5_1_coupling", "5_2_dynamical",
)


def main():
    runs, envs = {}, {}
    for case in CASES:
        models, media = utils.load_simple_models(case)
        for model in models:
            model.solver = SOLVER
        community = gifba.gifbaObject(models, media)
        env_final, org_final = community.run_gifba(iters=ITERS, method="pfba")

        objectives = [
            round(float(org_final.loc[idx, rxn]), 12)
            for idx, rxn in community.objective_rxns.items()
        ]
        runs[case] = (
            community.iter_converged,
            community.periodicity,
            community.simulation_ct,
            objectives,
        )
        envs[case] = {k: round(float(v), 12) for k, v in env_final.items()}

    width = max(len(c) for c in CASES) + 3
    print("GOLDEN_RUNS = {")
    for case, value in runs.items():
        print(f"    {(repr(case) + ':'):<{width}} {value!r},")
    print("}")
    print()
    print("GOLDEN_ENV = {")
    for case, value in envs.items():
        print(f"    {(repr(case) + ':'):<{width}} {value!r},")
    print("}")


if __name__ == "__main__":
    main()
