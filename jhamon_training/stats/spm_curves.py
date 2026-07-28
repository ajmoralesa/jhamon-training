"""SPM on a :class:`~jhamon_training.curves.CurveSet`, whatever its abscissa.

The two SPM families the manuscript reports -- the between-modality contrast and
the session-by-session progression contrast -- are implemented here once, taking
the abscissa as data rather than as an assumption. ``stats/training.py`` and
``stats/spm.py`` hard-code the time axis and re-implement the same pivot, the
same balanced-design check and the same inference in slightly different ways;
this module replaces both for new work and lets the identical analysis be run on
knee angle simply by handing it a different :class:`~jhamon_training.curves.Axis`.

Cluster endpoints are reported in the axis's own units, so a cluster on a
``knee_angle`` axis reads as "83 -> 25 deg" and one on the time axis reads as
"72 -> 100 %". That is the whole point: on the time axis those two statements
are not interchangeable, because the two modalities reach a given percentage of
their duration at very different joint angles.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from jhamon_training.curves import Axis, CurveSet


# --------------------------------------------------------------------------- #
# clusters
# --------------------------------------------------------------------------- #
def extract_clusters(spmi, axis: Axis, alpha: Optional[float] = None) -> List[Dict[str, Any]]:
    """Supra-threshold clusters, with endpoints converted to axis coordinates.

    Args:
        spmi: an inferred ``spm1d`` object.
        axis: the abscissa the curves were expressed on.
        alpha: when given, keep only clusters whose cluster-level p is below it.
            ``spm1d`` already thresholds at the alpha passed to ``inference``,
            but a Bonferroni-corrected family needs the cluster-level p filtered
            at the corrected alpha too.

    Returns:
        Clusters ordered along the axis. ``direction`` is the sign of the test
        statistic inside the cluster; for a two-sample t with Nordic first,
        +1 means NH > IK.
    """
    out: List[Dict[str, Any]] = []
    for c in getattr(spmi, "clusters", []) or []:
        start_node, end_node = (float(v) for v in c.endpoints)
        lo = int(np.floor(start_node))
        hi = min(int(np.ceil(end_node)), axis.n_nodes - 1)
        z = np.asarray(spmi.z)[lo : hi + 1]
        if alpha is not None and float(c.P) >= alpha:
            continue
        out.append(
            {
                "start": round(axis.coordinate(start_node), 1),
                "end": round(axis.coordinate(min(end_node, axis.n_nodes - 1)), 1),
                "start_node": round(start_node, 1),
                "end_node": round(end_node, 1),
                "extent_nodes": round(end_node - start_node, 1),
                "p": float(c.P),
                "z_peak": round(float(np.nanmax(np.abs(z))), 3) if z.size else None,
                "direction": int(np.sign(np.nanmean(z))) if z.size else 0,
                "unit": axis.unit,
            }
        )
    return sorted(out, key=lambda d: d["start_node"])


def format_clusters(clusters: List[Dict[str, Any]], positive: str, negative: str) -> str:
    """One-line human-readable summary, for logs and figure captions."""
    if not clusters:
        return "no supra-threshold cluster"
    parts = []
    for c in clusters:
        who = positive if c["direction"] > 0 else negative
        p = "<0.001" if c["p"] < 0.001 else f"= {c['p']:.3f}"
        parts.append(f"{who} {c['start']:g}-{c['end']:g}{c['unit']} (p {p})")
    return " | ".join(parts)


# --------------------------------------------------------------------------- #
# between-modality contrast
# --------------------------------------------------------------------------- #
def between_groups(
    curves: CurveSet,
    groups: tuple[str, str] = ("NH", "IK"),
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Two-sample SPM{t}, one curve per participant, unequal variances.

    The first group is the positive direction, so with the default
    ``("NH", "IK")`` a positive cluster means Nordic exceeded isokinetic.
    Unequal variances are assumed because the two modalities differ markedly in
    between-participant spread -- the dynamometer imposes its own velocity on
    everyone, the Nordic descent does not -- and ``spm1d`` then estimates a
    single ReML effective degrees of freedom applied at every node.
    """
    import spm1d

    a, b = (curves.participant_means(g) for g in groups)
    t = spm1d.stats.ttest2(a, b, equal_var=False)
    ti = t.inference(alpha, two_tailed=True)
    clusters = extract_clusters(ti, curves.axis)
    return {
        "var": curves.var,
        "axis": curves.axis.kind,
        "axis_unit": curves.axis.unit,
        "n_nodes": curves.axis.n_nodes,
        "groups": list(groups),
        f"n_{groups[0].lower()}": int(a.shape[0]),
        f"n_{groups[1].lower()}": int(b.shape[0]),
        "n_reps": {g: int((curves.meta["tr_group"] == g).sum()) for g in groups},
        "coverage": {g: round(curves.coverage().get(g, float("nan")), 4) for g in groups},
        "alpha": alpha,
        "tstar": round(float(ti.zstar), 3),
        "df": round(float(np.ravel(ti.df)[-1]), 1),
        "fwhm": round(float(ti.fwhm), 2),
        "clusters": clusters,
        "summary": format_clusters(clusters, f"{groups[0]}>{groups[1]}", f"{groups[1]}>{groups[0]}"),
    }


# --------------------------------------------------------------------------- #
# within-group progression contrast
# --------------------------------------------------------------------------- #
def progression(
    curves: CurveSet,
    group: str,
    alpha: float = 0.05,
    reference: str = "tr_1",
    sessions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Reference session versus each later session, paired SPM{t}.

    ``stats/spm.py`` runs ``anova1rm`` here. With two levels a one-way
    repeated-measures ANOVA is a paired t-test (F = t^2), so the inference is
    identical; the t form is used because it carries direction and the F form
    does not.

    A participant contributes only when they have BOTH sessions, which is what
    keeps the design balanced. The number retained is reported per contrast
    because it varies -- Nordic participants missed sessions -- and a cluster
    from eight pairs should not be read like a cluster from thirteen.
    """
    import spm1d

    sessions = sessions or [f"tr_{k}" for k in range(2, 16)]
    index, values = curves.cell_means(group)

    out: Dict[str, Any] = {}
    for session in sessions:
        pair = index["trses"].isin([reference, session]).to_numpy()
        idx, vals = index[pair].reset_index(drop=True), values[pair]
        counts = idx.groupby("par")["trses"].nunique()
        participants = sorted(counts[counts == 2].index)
        if len(participants) < 3:
            out[session] = {"n": len(participants), "skipped": "fewer than 3 paired participants",
                            "clusters": []}
            continue

        pos = {(p, s): i for i, (p, s) in enumerate(zip(idx["par"], idx["trses"]))}
        before = np.vstack([vals[pos[(p, reference)]] for p in participants])
        after = np.vstack([vals[pos[(p, session)]] for p in participants])

        ti = spm1d.stats.ttest_paired(after, before).inference(alpha, two_tailed=True)
        clusters = extract_clusters(ti, curves.axis, alpha=alpha)
        out[session] = {
            "n": len(participants),
            "alpha": alpha,
            "tstar": round(float(ti.zstar), 3),
            "fwhm": round(float(ti.fwhm), 2),
            "clusters": clusters,
            "summary": format_clusters(clusters, f"{session}>{reference}", f"{reference}>{session}"),
        }
    return out


def progression_summary(per_session: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse the session-by-session contrasts into reportable scalars.

    ``first_sustained`` is the earliest session from which every later session is
    also significant, which is the quantity a reader wants when asking when a
    change became established rather than when it first flickered.
    """
    significant = [s for s, e in per_session.items() if e.get("clusters")]
    order = list(per_session.keys())
    sustained = None
    for s in significant:
        tail = order[order.index(s) :]
        if all(t in significant for t in tail):
            sustained = s
            break
    starts = [c["start"] for s in significant for c in per_session[s]["clusters"]]
    ends = [c["end"] for s in significant for c in per_session[s]["clusters"]]
    return {
        "n_sig": len(significant),
        "n_tested": len([s for s, e in per_session.items() if "skipped" not in e]),
        "sig_sessions": significant,
        "first_sustained": sustained,
        "cluster_start_min": round(min(starts), 1) if starts else None,
        "cluster_start_max": round(max(starts), 1) if starts else None,
        "cluster_end_min": round(min(ends), 1) if ends else None,
        "cluster_end_max": round(max(ends), 1) if ends else None,
    }
