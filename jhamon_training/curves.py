"""Put each repetition's curve on a chosen abscissa.

Why this exists
---------------
Every repetition in the caches is stored linearly TIME-normalised to 101 nodes
(``utils.linear_tnorm``), and every SPM analysis in the project has been
computed on that axis. Time is not a shared reference between the two training
modalities. A Nordic descent accelerates throughout -- the participant resists
almost isometrically at first and then falls -- so the first half of its
duration covers only about a fifth of its range of motion. An isokinetic
repetition advances at a constant angular velocity, so its time axis is very
nearly its angle axis.

Measured on the group-mean trajectories, the Nordic angle-versus-time curve
departs from a straight line by up to 28.9 deg (37 % of the range of motion)
while the isokinetic one departs by 6.7 deg (8.6 %, essentially the
dynamometer's acceleration and deceleration ramps). The consequence is that the
same node denotes a different knee angle in each group -- at node 50 the Nordic
knee is at 76.7 deg and the isokinetic knee at 48.1 deg, and the gap peaks at
37.7 deg near node 80. A between-modality test computed on that axis compares
hamstring torque at one joint angle against hamstring torque at another, and
knee-flexor torque is strongly angle-dependent, so joint angle is confounded
with modality.

This module re-expresses the same stored curves on abscissae that are shared:

``time_axis()``
    The shipped axis: 0-100 % of the repetition's duration, 101 nodes. Kept so
    that the historical result can be reproduced exactly and reported as the
    temporal-patterning analysis it actually is.

``rom_percent_axis()``
    0-100 % of each repetition's OWN angular excursion, 101 nodes. The direct
    counterpart of the time axis -- same node count, no repetition discarded,
    the whole movement covered -- differing only in that the abscissa advances
    with the joint rather than with the clock. Because it truncates nothing,
    it isolates the effect of the abscissa from the effect of restricting the
    analysed range.

``knee_angle_axis()``
    Absolute knee flexion angle on a common grid (default 80 -> 30 deg). Node k
    is the same joint position in every repetition, every participant and both
    modalities, so a supra-threshold cluster can be read as an anatomical
    range. Repetitions whose excursion does not span the grid are dropped, so
    the grid is a coverage-versus-interpretability trade-off; ``coverage()``
    reports what a given grid costs.

Resolution caveat
-----------------
The caches store only the 101-point time-normalised curves, so the angle
registrations here are resampled from those rather than from raw signal. Where a
Nordic descent is fastest -- the last tenth of its duration -- consecutive time
nodes are about 3 deg apart, so angular resolution there is coarser than the
1 deg grid suggests. ``jhamon_training.curves`` is deliberately agnostic about
where the curves came from: re-deriving the caches at full sample rate and
feeding them through the same functions requires no change here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

REP_KEYS = ["par", "trses", "set", "rep"]

#: Repetitions with a smaller angular excursion than this (deg) are treated as
#: segmentation failures rather than movements, and are never registered. The
#: 5th percentile of a real repetition's range of motion is about 65 deg in both
#: groups, so this rejects only degenerate traces.
DEFAULT_MIN_ROM = 30.0


# --------------------------------------------------------------------------- #
# axes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Axis:
    """The abscissa a set of curves is expressed on.

    Attributes:
        kind: ``"time"``, ``"rom_percent"`` or ``"knee_angle"``.
        values: coordinate of each node, in ``unit``. For ``knee_angle`` this
            descends (85, 84, ... 25) because the movement descends; cluster
            endpoints are reported in these coordinates, so they read the way
            the movement runs.
        unit: suffix used when a cluster is formatted.
        label: axis label for figures.
    """

    kind: str
    values: np.ndarray
    unit: str
    label: str

    @property
    def n_nodes(self) -> int:
        return int(self.values.size)

    def coordinate(self, node: float) -> float:
        """Interpolate the axis coordinate of a (possibly fractional) node."""
        return float(np.interp(node, np.arange(self.n_nodes), self.values))


def time_axis(n_nodes: int = 101) -> Axis:
    """0-100 % of the repetition's duration -- the shipped abscissa."""
    return Axis("time", np.linspace(0.0, 100.0, n_nodes), "%", "Repetition duration (%)")


def rom_percent_axis(n_nodes: int = 101) -> Axis:
    """0-100 % of each repetition's own angular excursion."""
    return Axis(
        "rom_percent",
        np.linspace(0.0, 100.0, n_nodes),
        "%ROM",
        "Range of motion completed (%)",
    )


def knee_angle_axis(hi: float = 80.0, lo: float = 30.0, step: float = 1.0) -> Axis:
    """Absolute knee flexion angle, ``hi`` down to ``lo`` degrees.

    The default 80 -> 30 deg window is the widest grid that keeps essentially
    every repetition (97 % of Nordic and 100 % of isokinetic); widening it to
    88 -> 15 deg costs more than half the Nordic repetitions. Angles follow the
    convention used throughout the project: 0 deg is full extension.
    """
    if not hi > lo:
        raise ValueError(f"expected hi > lo, got hi={hi}, lo={lo}")
    return Axis(
        "knee_angle",
        np.arange(hi, lo - 1e-9, -abs(step)),
        "deg",
        "Knee flexion angle (deg)",
    )


# --------------------------------------------------------------------------- #
# curve sets
# --------------------------------------------------------------------------- #
@dataclass
class CurveSet:
    """One row per repetition, all expressed on a single :class:`Axis`.

    Attributes:
        meta: identifying columns (``par``, ``trses``, ``set``, ``rep``,
            ``tr_group``), one row per retained repetition.
        values: ``(n_repetitions, axis.n_nodes)`` array of registered curves.
        axis: the abscissa ``values`` is expressed on.
        var: the curve variable, e.g. ``"torque"``.
        dropped: per-group count of repetitions that could not be registered.
        n_input: per-group count of repetitions offered.
    """

    meta: pd.DataFrame
    values: np.ndarray
    axis: Axis
    var: str
    n_input: Dict[str, int] = field(default_factory=dict)
    dropped: Dict[str, int] = field(default_factory=dict)

    def groups(self) -> Sequence[str]:
        return sorted(self.meta["tr_group"].dropna().unique())

    def participant_means(self, group: str) -> np.ndarray:
        """One curve per participant: the mean of all their repetitions.

        This is the unit of analysis the manuscript reports for the
        between-modality contrast. Averaging within participant is what makes
        the rows independent; one row per participant x session would treat
        roughly fifteen curves from the same person as separate observations.
        """
        sub = self.meta["tr_group"] == group
        par = self.meta.loc[sub, "par"].to_numpy()
        vals = self.values[sub.to_numpy()]
        return np.vstack([vals[par == p].mean(axis=0) for p in sorted(set(par))])

    def cell_means(self, group: str) -> tuple[pd.DataFrame, np.ndarray]:
        """One curve per participant x session, for the progression contrasts."""
        sub = (self.meta["tr_group"] == group).to_numpy()
        m = self.meta[sub].reset_index(drop=True)
        frame = (
            pd.DataFrame(self.values[sub])
            .assign(par=m["par"].values, trses=m["trses"].values)
            .groupby(["par", "trses"])
            .mean()
        )
        return frame.index.to_frame(index=False), frame.to_numpy(float)

    def coverage(self) -> Dict[str, float]:
        """Fraction of the offered repetitions that survived registration."""
        return {
            g: 1.0 - self.dropped.get(g, 0) / n if n else float("nan")
            for g, n in self.n_input.items()
        }

    def select(self, mask: np.ndarray) -> "CurveSet":
        """A new CurveSet restricted to the repetitions where ``mask`` is True."""
        return CurveSet(
            meta=self.meta[mask].reset_index(drop=True),
            values=self.values[mask],
            axis=self.axis,
            var=self.var,
            n_input=dict(self.n_input),
            dropped=dict(self.dropped),
        )


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
def repetition_matrices(df: pd.DataFrame, var: str):
    """Pivot the long curve frame into per-repetition value and angle matrices.

    Returns ``(meta, values, angles)`` aligned row for row. Only repetitions
    that carry BOTH ``var`` and ``knee_ROM`` are returned, because a curve
    cannot be put on an angle axis without its own angle trace.
    """
    piv = (
        df[df["var"].isin([var, "knee_ROM"])]
        .groupby(REP_KEYS + ["var", "timepoint"], observed=True)["value"]
        .mean()
        .unstack("timepoint")
        .sort_index(axis=1)
    )
    if var not in piv.index.get_level_values("var"):
        raise ValueError(f"no {var!r} curves in this frame")
    if "knee_ROM" not in piv.index.get_level_values("var"):
        raise ValueError(
            "no 'knee_ROM' curves in this frame, so no curve can be angle-registered. "
            "This is the signature of the pre-2026-07 dame_manual_exclusions defect, "
            "which deleted every isokinetic angle trace."
        )
    val, ang = piv.xs(var, level="var"), piv.xs("knee_ROM", level="var")
    common = val.index.intersection(ang.index)
    val, ang = val.loc[common], ang.loc[common]

    meta = common.to_frame(index=False)
    groups = (
        df.drop_duplicates(REP_KEYS)
        .set_index(REP_KEYS)["tr_group"]
        .reindex(pd.MultiIndex.from_frame(meta[REP_KEYS]))
        .to_numpy()
    )
    meta["tr_group"] = groups
    return meta, val.to_numpy(float), ang.to_numpy(float)


def _monotone_descent(angle: np.ndarray) -> np.ndarray:
    """Force one angle trace to be non-increasing so it can be inverted.

    Registration needs angle(t) to be invertible. The stored traces already are,
    to within numerical wobble: the median repetition has 0.00 deg of upward
    travel and the 99th percentile has about 1 % of its range of motion, so a
    running minimum changes almost nothing while guaranteeing invertibility.
    Repetitions with a genuine reversal are rare enough that clipping them is
    preferable to discarding them, but ``backtrack_fraction`` below lets a
    caller audit the assumption.
    """
    return np.minimum.accumulate(angle)


def backtrack_fraction(angles: np.ndarray) -> np.ndarray:
    """Upward angular travel per repetition, as a fraction of its excursion.

    Diagnostic for the monotonicity assumption ``_monotone_descent`` makes.
    """
    d = np.diff(angles, axis=1)
    up = np.where(d > 0, d, 0.0).sum(axis=1)
    rom = np.abs(angles[:, 0] - angles[:, -1])
    return np.divide(up, rom, out=np.full_like(rom, np.nan), where=rom > 1)


def register(
    values: np.ndarray,
    angles: np.ndarray,
    axis: Axis,
    min_rom: float = DEFAULT_MIN_ROM,
) -> np.ndarray:
    """Resample every repetition onto ``axis``; unregistrable rows become NaN.

    Args:
        values: ``(n, 101)`` time-normalised curves.
        angles: ``(n, 101)`` time-normalised knee angle traces, same rows.
        axis: target abscissa.
        min_rom: reject repetitions whose excursion is smaller than this.

    Returns:
        ``(n, axis.n_nodes)``. A row is NaN when the repetition is degenerate
        or, for a ``knee_angle`` axis, does not span the grid.
    """
    if values.shape != angles.shape:
        raise ValueError(f"values {values.shape} and angles {angles.shape} differ")
    if axis.kind == "time":
        if values.shape[1] != axis.n_nodes:
            raise ValueError(
                f"time axis has {axis.n_nodes} nodes but the curves have {values.shape[1]}"
            )
        return values.copy()

    out = np.full((values.shape[0], axis.n_nodes), np.nan)
    for i in range(values.shape[0]):
        a = _monotone_descent(angles[i])
        rom = a[0] - a[-1]
        if not np.isfinite(rom) or rom < min_rom:
            continue
        if axis.kind == "rom_percent":
            # (a[0] - a) / rom increases monotonically 0 -> 1 along the descent,
            # which is exactly what np.interp needs as its sample abscissa.
            out[i] = np.interp(axis.values / 100.0, (a[0] - a) / rom, values[i])
        elif axis.kind == "knee_angle":
            if a[0] < axis.values[0] or a[-1] > axis.values[-1]:
                continue  # this repetition does not span the requested grid
            # np.interp needs an increasing sample abscissa, so flip the descent.
            out[i] = np.interp(axis.values, a[::-1], values[i][::-1])
        else:
            raise ValueError(f"unknown axis kind {axis.kind!r}")
    return out


def build_curveset(
    df: pd.DataFrame,
    var: str,
    axis: Axis,
    min_rom: float = DEFAULT_MIN_ROM,
    restrict_to: Optional[pd.MultiIndex] = None,
) -> CurveSet:
    """Register every repetition of ``var`` in ``df`` onto ``axis``.

    Args:
        df: long curve frame (``par``, ``trses``, ``set``, ``rep``, ``var``,
            ``timepoint``, ``value``, ``tr_group``).
        var: curve variable, e.g. ``"torque"`` or ``"knee_v"``.
        axis: target abscissa.
        min_rom: minimum angular excursion, in degrees.
        restrict_to: optional repetition index to keep. Pass the intersection
            across several axes when the axes must be compared on identical
            repetitions, so that a difference in the result cannot be a
            difference in the sample.

    Returns:
        A :class:`CurveSet` containing only successfully registered repetitions.
    """
    meta, values, angles = repetition_matrices(df, var)
    if restrict_to is not None:
        keep = pd.MultiIndex.from_frame(meta[REP_KEYS]).isin(restrict_to)
        meta, values, angles = meta[keep].reset_index(drop=True), values[keep], angles[keep]

    registered = register(values, angles, axis, min_rom=min_rom)
    ok = np.isfinite(registered).all(axis=1)

    n_input = meta["tr_group"].value_counts().to_dict()
    dropped = meta.loc[~ok, "tr_group"].value_counts().to_dict()
    return CurveSet(
        meta=meta[ok].reset_index(drop=True),
        values=registered[ok],
        axis=axis,
        var=var,
        n_input={k: int(v) for k, v in n_input.items()},
        dropped={k: int(dropped.get(k, 0)) for k in n_input},
    )


def common_repetitions(df: pd.DataFrame, var: str, axes: Sequence[Axis],
                       min_rom: float = DEFAULT_MIN_ROM) -> pd.MultiIndex:
    """Repetitions that register successfully on EVERY axis in ``axes``.

    Comparing abscissae is only meaningful when the comparison holds the sample
    fixed; otherwise a shifted cluster could just be a different set of
    repetitions. Feed the result back through ``build_curveset(restrict_to=...)``.
    """
    meta, values, angles = repetition_matrices(df, var)
    ok = np.ones(len(meta), bool)
    for axis in axes:
        ok &= np.isfinite(register(values, angles, axis, min_rom=min_rom)).all(axis=1)
    return pd.MultiIndex.from_frame(meta.loc[ok, REP_KEYS])


def select_top_repetitions(curves: CurveSet, n_reps: int = 3) -> CurveSet:
    """Keep the ``n_reps`` highest-peak repetitions per participant x session.

    Mirrors ``jhamon_training.select_top_repetitions``. The peak is taken on the
    registered curve, but note that the selection is deliberately made on the
    TIME-registered peak by the callers that need axis-invariant samples: a peak
    is a peak whatever the abscissa, yet an angle grid that truncates the ends
    can move it, and the progression contrasts must compare the same
    repetitions across axes.
    """
    tmp = curves.meta.assign(peak=np.nanmax(curves.values, axis=1))
    keep = (
        tmp.sort_values(["par", "trses", "peak"], ascending=[True, True, False])
        .groupby(["par", "trses"])
        .head(n_reps)
        .index
    )
    mask = np.zeros(len(tmp), bool)
    mask[keep] = True
    return curves.select(mask)
