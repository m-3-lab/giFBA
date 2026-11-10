"""Measured reference values for the bundled toy models.

Every number here was produced by running ``run_gifba(n_iterations=25, method="pfba")``
against the models shipped in ``gifba/Toy_Models`` with the LP solver pinned to
GLPK. They are *observed* values, not analytically derived ones, so they double
as a regression fence: if a refactor of ``gifba_object.py`` changes any of them,
the change was numerically visible and needs justifying.

Updating a value here is a deliberate act. Re-measure with
``tests/_remeasure.py`` rather than hand-editing.

In ``2_3_superfluous_crossfeed`` organism 2 does not grow (objective 0.0). That
is the behavior the case is built to show: organism 2 only eats C, and under
greedy pFBA organism 1 gains nothing by converting B to C, so it never
secretes any.
"""

# n_iterations used for every golden run; all ten cases converge well inside this.
GOLDEN_ITERS = 25

# case -> (iter_converged, periodicity, simulation_count, [objective per model])
GOLDEN_RUNS = {
    "1_1_single":                (2, 1, 3, [10.0]),
    "1_2_single":                (2, 1, 3, [20.0]),
    "1_3_parallel":              (2, 1, 6, [10.0, 10.0]),
    "2_1_crossfeed":             (3, 1, 8, [20.0, 10.0]),
    "2_2_layered":               (5, 1, 11, [20.0, 20.0]),
    "2_3_superfluous_crossfeed": (2, 1, 3, [10.0, 0.0]),
    "3_competition":             (2, 1, 12, [5.0, 5.0]),
    "4_1_crossfeed_competition": (3, 1, 16, [5.0, 10.0]),
    "4_2_coupling":              (3, 1, 13, [5.0, 5.0]),
    "4_4_dynamical":             (8, 2, 26, [13.75, 21.25]),
}

# case -> fixed point of the environment, in mmol/(gT * hr).
GOLDEN_ENV = {
    "1_1_single":                {"EX_A(e)": 10.0, "EX_Bio(e)": 10.0},
    "1_2_single":                {"EX_A(e)": 10.0, "EX_B(e)": 0.0, "EX_Bio(e)": 20.0},
    "1_3_parallel":              {"EX_A(e)": 10.0, "EX_B(e)": 10.0, "EX_Bio(e)": 20.0},
    "2_1_crossfeed":             {"EX_A(e)": 10.0, "EX_B(e)": 10.0, "EX_C(e)": 10.0,
                              "EX_Bio(e)": 30.0},
    "2_2_layered":               {"EX_A(e)": 10.0, "EX_B(e)": 10.0, "EX_C(e)": 10.0,
                                      "EX_D(e)": 10.0, "EX_Bio(e)": 40.0},
    "2_3_superfluous_crossfeed": {"EX_A(e)": 10.0, "EX_B(e)": 10.0, "EX_C(e)": 0.0,
                                  "EX_Bio(e)": 10.0},
    "3_competition":             {"EX_A(e)": 10.0, "EX_Bio(e)": 10.0},
    "4_1_crossfeed_competition": {"EX_A(e)": 10.0, "EX_B(e)": 5.0, "EX_Bio(e)": 15.0},

    "4_2_coupling":              {"EX_A(e)": 10.0, "EX_B(e)": 5.0, "EX_Bio(e)": 10.0},
    "4_4_dynamical":             {"EX_A(e)": 10.0, "EX_B(e)": 22.5, "EX_C(e)": 10.0,
                                  "EX_Bio(e)": 35.0},
}

# Every key accepted by utils.load_simple_models, in declaration order.
TOY_CASES = tuple(GOLDEN_RUNS)
