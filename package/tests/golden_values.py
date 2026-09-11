"""Measured reference values for the bundled toy models.

Every number here was produced by running ``run_gifba(n_iterations=25, method="pfba")``
against the models shipped in ``gifba/Toy_Models`` with the LP solver pinned to
GLPK. They are *observed* values, not analytically derived ones, so they double
as a regression fence: if a refactor of ``gifba_object.py`` changes any of them,
the change was numerically visible and needs justifying.

Updating a value here is a deliberate act. Re-measure with
``tests/_remeasure.py`` rather than hand-editing.

Two rows pin behavior that is arguably wrong, and will need re-measuring if
the underlying loader bugs are fixed:

* ``4_2_superfluous_crossfeed`` -- organism 2 does not grow (objective 0.0)
  because ``utils.load_simple_models`` has an unreachable ``elif`` that denies
  it ``EX_D(e)``.
* ``5_2_dynamical`` -- its medium never reaches the models. The loader supplies
  ``{"EX_A(e)": -10}`` but ``sim5_2_org{1,2}.json`` spell their exchanges
  ``Ex_A(e)`` with a lowercase x, so the key matches nothing and the case runs
  on an empty medium. The period-4 attractor below is therefore driven by a
  mass-generating loop in the models rather than by the intended medium.
"""

# n_iterations used for every golden run; all ten cases converge well inside this.
GOLDEN_ITERS = 25

# case -> (iter_converged, periodicity, simulation_count, [objective per model])
GOLDEN_RUNS = {
    "1_1_single":                (2, 1, 3, [10.0]),
    "1_2_single":                (2, 1, 3, [20.0]),
    "1_3_parallel":              (2, 1, 6, [10.0, 10.0]),
    "2_1_competition":           (2, 1, 12, [5.0, 5.0]),
    "3_1_crossfeed":             (3, 1, 8, [20.0, 10.0]),
    "3_2_layered":               (5, 1, 11, [20.0, 20.0]),
    "4_1_crossfeed_competition": (3, 1, 16, [5.0, 10.0]),
    "4_2_superfluous_crossfeed": (2, 1, 3, [10.0, 0.0]),
    "5_1_coupling":              (3, 1, 13, [5.0, 5.0]),
    "5_2_dynamical":             (8, 4, 19, [8.75, 16.25]),
}

# case -> fixed point of the environment, in mmol/(gT * hr).
# Note sim5_2's exchanges are spelled "Ex_" (lowercase x), unlike every other
# toy model's "EX_". That is the models' own spelling, not a typo here.
GOLDEN_ENV = {
    "1_1_single":                {"EX_A(e)": 10.0, "EX_Bio(e)": 10.0},
    "1_2_single":                {"EX_A(e)": 10.0, "EX_B(e)": 0.0, "EX_Bio(e)": 20.0},
    "1_3_parallel":              {"EX_A(e)": 10.0, "EX_B(e)": 10.0, "EX_Bio(e)": 20.0},
    "2_1_competition":           {"EX_A(e)": 10.0, "EX_Bio(e)": 10.0},
    "3_1_crossfeed":             {"EX_A(e)": 10.0, "EX_B(e)": 10.0, "EX_C(e)": 10.0,
                                  "EX_Bio(e)": 30.0},
    "3_2_layered":               {"EX_A(e)": 10.0, "EX_B(e)": 10.0, "EX_C(e)": 10.0,
                                  "EX_D(e)": 10.0, "EX_Bio(e)": 40.0},
    "4_1_crossfeed_competition": {"EX_A(e)": 10.0, "EX_B(e)": 5.0, "EX_Bio(e)": 15.0},
    "4_2_superfluous_crossfeed": {"EX_A(e)": 10.0, "EX_B(e)": 10.0, "EX_C(e)": 0.0,
                                  "EX_Bio(e)": 10.0},
    "5_1_coupling":              {"EX_A(e)": 10.0, "EX_B(e)": 5.0, "EX_Bio(e)": 10.0},
    "5_2_dynamical":             {"Ex_A(e)": 3.75, "Ex_B(e)": 12.5, "Ex_C(e)": 10.0,
                                  "EX_Bio(e)": 25.0},
}

# Every key accepted by utils.load_simple_models, in declaration order.
TOY_CASES = tuple(GOLDEN_RUNS)
