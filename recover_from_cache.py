"""Regenerate the discrete outcomes + LMM stats from the surviving 2019 caches.

Written 2026-07-22 after the results folder went missing. The full pipeline in
``jhamon_training.py`` cannot run: ``RESULTS_TRAINING_PATH`` points at a
SynologyDrive folder that no longer exists, and ``/Volumes/AJMA`` (raw data) is
not mounted. The cached ``nht_results.pkl`` / ``ikt_results.pkl`` in
``~/Desktop/2019_jHamON/_RESULTS`` do survive, and everything downstream of raw
signal processing can be rebuilt from them.

Three defects stand between those caches and a working run; each is patched
here at the boundary rather than in the package, so the package stays untouched
until the sign-convention question below is settled:

1. The caches are keyed by real participant names (``01_Kevin``), not the
   ``jhamonNN`` pseudonyms the current code expects. Unmapped keys leave
   ``tr_group`` as NaN and ``frames.nht_todf`` dies on ``" ".join``.
2. IK ``knee_v_mean`` is negative in this cache (mean -20.2 deg/s), so
   ``filter_training_data``'s ``10 < knee_v_mean < 35`` window silently drops
   all ~9900 IK rows and the LMM collapses to NH-only.
3. Outputs must go somewhere that exists.

OPEN QUESTION (do not resolve by copying this script into the pipeline): whether
the IK sign fix belongs here or in ``calculate_ik_discrete_variables``. The
manuscript reports IK velocity as +20.0 deg/s, so the published run had it
positive somewhere upstream.

PROVENANCE WARNING: the NH cache is an older processing vintage than the run
behind the manuscript (6014 vs 5951 reps; 185.7 vs 187.4 J per rep). IK matches
almost exactly. Do not mix vintages across Results -- see REVISIONS.md.

Usage:
    source venv/bin/activate && python recover_from_cache.py [OUTDIR]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

CACHE_DIR = Path.home() / "Desktop" / "2019_jHamON" / "_RESULTS"
DEFAULT_OUT = Path.home() / "Desktop" / "2019_jHamON" / "_RESULTS_TRAINING"
DVS = ["mean_torque", "peak_torque", "knee_ROM", "angle_at_peak_torque", "knee_v_mean"]

# What the manuscript currently reports, for a provenance check on every run.
PUBLISHED = {
    "peak_torque": (221.3, 154.8),
    "angle_at_peak_torque": (41.2, 16.8),
    "knee_v_mean": (21.9, 20.0),
}


def pseudonymize(d: dict) -> dict:
    """Remap ``01_Kevin`` -> ``jhamon01``; leave already-pseudonymous keys alone."""
    out = {}
    for key, value in d.items():
        m = re.match(r"^(\d+)_", key)
        out[f"jhamon{m.group(1)}" if m else key] = value
    return out


def main(outdir: Path) -> None:
    from jhamon_training.data import frames
    from jhamon_training.data.utils import filter_training_data
    from jhamon_training.saveload import load_obj
    from jhamon_training.stats.analyses_discrete_vars import analyze_discrete_variables
    from jhamon_training.stats.mixed_lmm_analysis import run_mixed_lmm

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)

    nordict = pseudonymize(load_obj(name="nht_results.pkl", path=CACHE_DIR))
    ikdict = pseudonymize(load_obj(name="ikt_results.pkl", path=CACHE_DIR))
    print(f"loaded caches: {len(nordict)} NH, {len(ikdict)} IK participants")

    nordf = frames.nht_todf(my_dict=nordict)
    ikdf = frames.ikt_todf(my_dict=ikdict)
    nordf.to_feather(outdir / "nordf.feather")
    ikdf.to_feather(outdir / "ikdf.feather")

    results = analyze_discrete_variables(
        nordict=nordict,
        ikdf=ikdf,
        training_df=pd.concat([nordf, ikdf]),
        results_output_path=outdir,
    )
    training_disc = results["training_disc"].copy()

    is_ik = training_disc["tr_group"] == "IK"
    if (training_disc.loc[is_ik, "knee_v_mean"] < 0).any():
        print("IK knee_v_mean is negative in this cache -> taking absolute value")
        training_disc.loc[is_ik, "knee_v_mean"] = training_disc.loc[
            is_ik, "knee_v_mean"
        ].abs()

    training_disc = filter_training_data(training_disc)
    counts = training_disc.groupby("tr_group").size().to_dict()
    print(f"filtered rows per group: {counts}")
    if "IK" not in counts or "NH" not in counts:
        raise SystemExit("a whole group was filtered out -- refusing to fit the LMM")

    run_mixed_lmm(training_disc, json_path=outdir / "training_stats.json", write_json=True)

    stats = json.loads((outdir / "training_stats.json").read_text())
    print(f"\nLMM fitted on NH {stats['lmm_n_nh']} / IK {stats['lmm_n_ik']} participants")
    print(f"{'outcome':24s}{'NH':>9s}{'IK':>9s}{'published':>22s}")
    for dv in DVS:
        nh, ik = stats[f"lmm_{dv}_desc_nh_mean"], stats[f"lmm_{dv}_desc_ik_mean"]
        pub = PUBLISHED.get(dv)
        ref = f"{pub[0]} vs {pub[1]}" if pub else "-"
        print(f"{dv:24s}{nh:9.1f}{ik:9.1f}{ref:>22s}")
    print(f"\nwrote {outdir / 'training_stats.json'}")


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT))
