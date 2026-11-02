"""Package-wide constants."""

GROWTH_MIN_OBJ = 0.01
"""Minimum objective (growth) value for an organism's LP solution to be treated as
growing and contribute flux each iteration."""

ROUND = 6
"""Default decimal precision used when checking overconsumption ratios
(`gifbaObject`'s `oc_rounding` default)."""
