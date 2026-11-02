"""Plotting helpers shared by the toy-model notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

TRADEOFF_LABEL_GID = "micom_tradeoff_label"


def plot_micom_tradeoff_pts(
    micom_results: pd.DataFrame,
    iterpretation: str,
    tradeoffs: list[float],
    rel_abund: list[float],
    color: str,
    ms: float,
    lw: float,
    offsets: list[tuple[float, float]] | None = None,
) -> None:
    """Plot the MICOM tradeoff curve and label selected tradeoff points.

    Labels start at their point (plus `offsets`, if given). Call
    `adjust_tradeoff_labels` once the axis limits are final to move them clear of
    other markers.

    Args:
        micom_results (pd.DataFrame): MICOM results with `tradeoff`, `Org1_growth`
            and `Org2_growth` columns.
        iterpretation (str): `"relative"` plots growth as returned; `"specific"`
            divides by each organism's relative abundance.
        tradeoffs (list[float]): Tradeoff values to mark and label.
        rel_abund (list[float]): Relative abundance of each organism.
        color (str): Color for the curve, markers and labels.
        ms (float): Marker size for the labelled points.
        lw (float): Line width of the tradeoff curve.
        offsets (list[tuple[float, float]], optional): Per-label (x, y) nudges in
            data units. Defaults to no nudge.
    """
    x = micom_results["Org1_growth"].copy()
    y = micom_results["Org2_growth"].copy()
    if iterpretation == "specific":
        x /= rel_abund[0]
        y /= rel_abund[1]

    plt.plot(x, y, color=color, lw=lw)

    for idx, alpha_tradeoff in enumerate(tradeoffs):
        rows = micom_results[micom_results["tradeoff"] == alpha_tradeoff]
        posx = float(rows["Org1_growth"].iloc[0])
        posy = float(rows["Org2_growth"].iloc[0])
        if iterpretation == "specific":
            posx /= rel_abund[0]
            posy /= rel_abund[1]

        dx, dy = offsets[idx] if offsets is not None else (0.0, 0.0)
        text = plt.text(posx + dx, posy + dy,
                        rf"$\alpha$={alpha_tradeoff}",
                        fontsize=12,
                        zorder=5,
                        color=color)
        text.set_gid(TRADEOFF_LABEL_GID)
        # remember which point the label belongs to, in data coordinates
        text.anchor_xy = (posx, posy)

        plt.scatter(posx, posy, s=ms, c=color, marker="d", zorder=3)


def _marker_circles(ax: Axes) -> np.ndarray:
    """Approximate every marker and line on `ax` as circles in display pixels.

    Returns:
        np.ndarray: `(n, 3)` array of `x, y, radius`.
    """
    px_per_pt = ax.figure.dpi / 72
    circles = []

    for coll in ax.collections:
        offs = np.asarray(coll.get_offsets(), dtype=float)
        if offs.size == 0:
            continue
        pts = coll.get_offset_transform().transform(offs)
        sizes = np.broadcast_to(np.asarray(coll.get_sizes(), dtype=float), (len(pts),))
        # scatter `s` is marker area in pt^2, so its diameter is sqrt(s) points
        radii = np.sqrt(sizes) / 2 * px_per_pt
        circles.extend(np.column_stack([pts, radii]))

    for line in ax.lines:
        xy = np.asarray(line.get_xydata(), dtype=float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            continue
        pts = line.get_transform().transform(xy)

        if line.get_marker() not in (None, "None", "", " ") and line.get_markersize() > 0:
            r = line.get_markersize() / 2 * px_per_pt
            circles.extend((px, py, r) for px, py in pts)

        if line.get_linestyle() not in ("None", "", " ") and line.get_linewidth() > 0:
            r = line.get_linewidth() / 2 * px_per_pt + 1
            for p0, p1 in zip(pts[:-1], pts[1:]):
                n = max(int(np.hypot(*(p1 - p0)) / 2), 1)
                for t in np.linspace(0, 1, n + 1):
                    circles.append((*(p0 + t * (p1 - p0)), r))

    return np.array(circles, dtype=float).reshape(-1, 3)


def _circle_penetration(box: Bbox, circles: np.ndarray) -> float:
    """Sum of how far each circle reaches into `box`, in pixels."""
    if len(circles) == 0:
        return 0.0
    cx = np.clip(circles[:, 0], box.x0, box.x1)
    cy = np.clip(circles[:, 1], box.y0, box.y1)
    dist = np.hypot(circles[:, 0] - cx, circles[:, 1] - cy)
    return float(np.clip(circles[:, 2] - dist, 0, None).sum())


def adjust_tradeoff_labels(ax: Axes | None = None, pad_px: float = 3.0) -> None:
    """Move the MICOM tradeoff labels on `ax` clear of markers, lines and each other.

    Each label is tried at several distances and directions around its point and
    the least crowded spot is kept, preferring spots close to the point and never
    closer to another labelled point than its own. Call this after the axis limits
    are set, since placement works in display pixels.

    Args:
        ax (Axes, optional): Axes holding labels made by `plot_micom_tradeoff_pts`.
            Defaults to the current axes.
        pad_px (float): Clearance kept around each label, in pixels.
    """
    ax = ax if ax is not None else plt.gca()
    labels = [t for t in ax.texts if t.get_gid() == TRADEOFF_LABEL_GID]
    if not labels:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    circles = _marker_circles(ax)
    anchors = np.array([ax.transData.transform(t.anchor_xy) for t in labels])
    axes_box = ax.get_window_extent(renderer)

    angles = np.deg2rad(np.arange(0, 360, 22.5))
    radii = [8, 12, 18, 26, 36, 48, 62, 80]
    placed: list[Bbox] = []

    for idx, text in enumerate(labels):
        text.set_horizontalalignment("center")
        text.set_verticalalignment("center")
        extent = text.get_window_extent(renderer)
        w, h = extent.width + 2 * pad_px, extent.height + 2 * pad_px
        anchor = anchors[idx]
        others = np.delete(anchors, idx, axis=0)

        best_cost, best_center = np.inf, anchor
        for r in radii:
            for theta in angles:
                center = anchor + np.array([np.cos(theta) * (r + w / 2),
                                            np.sin(theta) * (r + h / 2)])
                box = Bbox.from_bounds(center[0] - w / 2, center[1] - h / 2, w, h)
                if (box.x0 < axes_box.x0 or box.x1 > axes_box.x1
                        or box.y0 < axes_box.y0 or box.y1 > axes_box.y1):
                    continue

                cost = 10 * _circle_penetration(box, circles) + 0.2 * r
                cost += sum(1e4 for other in placed if box.overlaps(other))
                own_dist = np.hypot(*(center - anchor))
                if len(others) and np.hypot(*(others - center).T).min() < own_dist:
                    cost += 200
                if cost < best_cost:
                    best_cost, best_center = cost, center

        text.set_position(ax.transData.inverted().transform(best_center))
        placed.append(Bbox.from_bounds(best_center[0] - w / 2, best_center[1] - h / 2, w, h))
