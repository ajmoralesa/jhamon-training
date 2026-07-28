#!/usr/bin/env python3
"""Render the axis comparison as one self-contained HTML page.

Reads ``axis_comparison.json`` and the two figures written by
``compare_axes.py`` and produces a single file with the images inlined as data
URIs, so it can be mailed to a co-author or opened from anywhere without
carrying a folder of assets alongside it.

Usage:
    source venv/bin/activate && python build_axis_report.py [--input-dir DIR]
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

AXIS_LABEL = {
    "time": "Time (% of duration)",
    "rom_percent": "Angle (% of own ROM)",
    "knee_angle": "Knee angle (deg)",
}
AXIS_NOTE = {
    "time": "the shipped axis",
    "rom_percent": "same node count, nothing truncated",
    "knee_angle": "anatomically interpretable",
}


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def esc(x: Any) -> str:
    return html.escape(str(x))


def fmt_p(p: float) -> str:
    return "&lt; 0.001" if p < 0.001 else f"= {p:.3f}"


def cluster_chips(clusters: List[Dict[str, Any]], pos: str, neg: str) -> str:
    if not clusters:
        return '<span class="chip chip-null">no supra-threshold cluster</span>'
    out = []
    for c in clusters:
        who = pos if c["direction"] > 0 else neg
        cls = "chip-pos" if c["direction"] > 0 else "chip-neg"
        out.append(
            f'<span class="chip {cls}"><b>{esc(who)}</b> {c["start"]:g}→{c["end"]:g}'
            f'{esc(c["unit"])} <span class="p">p {fmt_p(c["p"])}</span></span>'
        )
    return " ".join(out)


def between_table(entry: Dict[str, Any], var_label: str) -> str:
    rows = []
    for axis in ("time", "rom_percent", "knee_angle"):
        r = entry[axis]
        highlight = ' class="row-key"' if axis == "knee_angle" else ""
        rows.append(
            f"<tr{highlight}><td><b>{esc(AXIS_LABEL[axis])}</b>"
            f'<div class="sub">{esc(AXIS_NOTE[axis])}</div></td>'
            f'<td class="num">{r["tstar"]:.3f}</td>'
            f'<td class="num">{r["df"]:.1f}</td>'
            f'<td class="num">{r["fwhm"]:.1f}</td>'
            f'<td class="num">{r["n_nodes"]}</td>'
            f'<td>{cluster_chips(r["clusters"], "NH>IK", "IK>NH")}</td></tr>'
        )
    return f"""
<table>
  <caption>{esc(var_label)} — NH vs IK, one curve per participant (13 vs 14), α = 0.05 two-tailed</caption>
  <thead><tr><th>Abscissa</th><th class="num">t*</th><th class="num">df</th>
  <th class="num">FWHM</th><th class="num">nodes</th><th>Supra-threshold clusters</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""


def grid_table(grids: Dict[str, Any], var_label: str) -> str:
    rows = []
    for name, r in grids.items():
        rows.append(
            f"<tr><td><b>{esc(name)} deg</b></td>"
            f'<td class="num">{100 * r["coverage_nh"]:.0f}%</td>'
            f'<td class="num">{100 * r["coverage_ik"]:.0f}%</td>'
            f'<td class="num">{r["tstar"]:.3f}</td>'
            f'<td>{cluster_chips(r["clusters"], "NH>IK", "IK>NH")}</td></tr>'
        )
    return f"""
<table>
  <caption>{esc(var_label)} — does the angle-axis result depend on where the grid is cut?</caption>
  <thead><tr><th>Grid</th><th class="num">NH kept</th><th class="num">IK kept</th>
  <th class="num">t*</th><th>Supra-threshold clusters</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""


def progression_table(prog: Dict[str, Any], group: str, alpha_tag: str, alpha_label: str) -> str:
    rows = []
    for axis in ("time", "rom_percent", "knee_angle"):
        e = prog[f"{group}_{alpha_tag}_{axis}"]
        sessions = ", ".join(s.replace("tr_", "") for s in e["sig_sessions"]) or "—"
        sustained = e["first_sustained"].replace("tr_", "session ") if e["first_sustained"] else "none"
        tr15 = e["per_session"]["tr_15"]
        rows.append(
            f"<tr><td><b>{esc(AXIS_LABEL[axis])}</b></td>"
            f'<td class="num">{e["n_sig"]}/{e["n_tested"]}</td>'
            f'<td class="num">{esc(sustained)}</td>'
            f'<td class="sessions">{esc(sessions)}</td>'
            f'<td>{cluster_chips(tr15.get("clusters", []), "s15>s1", "s1>s15")}</td></tr>'
        )
    return f"""
<table>
  <caption>{esc(group)} torque progression — session 1 vs each later session, paired SPM{{t}}, {esc(alpha_label)}</caption>
  <thead><tr><th>Abscissa</th><th class="num">sig.</th><th class="num">first sustained</th>
  <th>significant sessions</th><th>session 1 vs 15</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""


def matched_table(matched: Dict[str, Any]) -> str:
    angles = matched["angles"]
    picks = [80, 70, 60, 50, 40, 30]
    idx = [min(range(len(angles)), key=lambda i: abs(angles[i] - a)) for a in picks]
    head = "".join(f'<th class="num">{a}°</th>' for a in picks)
    body = ""
    for label, unit in (("torque", "N·m"), ("velocity", "deg·s⁻¹")):
        e = matched[label]
        for who, key in (("NH", "NH"), ("IK", "IK")):
            cells = "".join(f'<td class="num">{e[key]["mean"][i]:.1f}</td>' for i in idx)
            body += f'<tr><td>{esc(label)} ({esc(unit)})</td><td><b>{who}</b></td>{cells}</tr>'
        cells = "".join(f'<td class="num diff">{e["difference"][i]:+.1f}</td>' for i in idx)
        body += f'<tr class="row-key"><td>{esc(label)}</td><td><b>NH − IK</b></td>{cells}</tr>'
    return f"""
<table>
  <caption>Group means at matched joint angles — only quotable once the abscissa is shared</caption>
  <thead><tr><th>Variable</th><th>Group</th>{head}</tr></thead>
  <tbody>{body}</tbody>
</table>"""


def mapping_table(mapping: Dict[str, Any]) -> str:
    nodes = list(range(0, 101, 10))
    head = "".join(f'<th class="num">{n}</th>' for n in nodes)
    rows = ""
    for g in ("NH", "IK"):
        cells = "".join(f'<td class="num">{mapping[g]["angle_at_node"][str(n)] if str(n) in mapping[g]["angle_at_node"] else mapping[g]["angle_at_node"][n]:.1f}</td>' for n in nodes)
        rows += f"<tr><td><b>{g}</b> knee angle (deg)</td>{cells}</tr>"
    gap = mapping["angle_gap_at_same_node"]
    cells = "".join(
        f'<td class="num diff">{(gap[str(n)] if str(n) in gap else gap[n]):+.1f}</td>' for n in nodes
    )
    rows += f'<tr class="row-key"><td><b>NH − IK</b> mismatch (deg)</td>{cells}</tr>'
    return f"""
<table>
  <caption>What the same time node means in each group</caption>
  <thead><tr><th>Node (% of duration)</th>{head}</tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def build(data: Dict[str, Any], fig_dir: Path) -> str:
    mapping = data["mapping"]
    between_none = data["between_groups"]["none"]
    between_corr = data["between_groups"]["corrected"]
    excl = data["exclusions"]
    grid = data["primary_grid"]

    t_time = between_none["torque"]["time"]
    t_angle = between_none["torque"]["knee_angle"]
    t_romp = between_none["torque"]["rom_percent"]
    v_time = between_none["velocity"]["time"]
    v_angle = between_none["velocity"]["knee_angle"]

    css = """
:root{--bg:#ffffff;--fg:#1a1a1a;--muted:#5c5c5c;--line:#e2e2e2;--card:#fafafa;
--pos:#15803d;--posbg:#dcfce7;--neg:#7e22ce;--negbg:#f3e8ff;--warn:#b45309;--warnbg:#fef3c7;
--key:#f0f7ff;--accent:#1d4ed8;}
@media (prefers-color-scheme:dark){:root{--bg:#111315;--fg:#e8e8e8;--muted:#a0a0a0;--line:#2c2f33;
--card:#191c1f;--pos:#4ade80;--posbg:#14351f;--neg:#c084fc;--negbg:#2c1a3d;--warn:#fbbf24;
--warnbg:#3a2c08;--key:#111e33;--accent:#7ea6ff;}}
:root[data-theme=dark]{--bg:#111315;--fg:#e8e8e8;--muted:#a0a0a0;--line:#2c2f33;--card:#191c1f;
--pos:#4ade80;--posbg:#14351f;--neg:#c084fc;--negbg:#2c1a3d;--warn:#fbbf24;--warnbg:#3a2c08;
--key:#111e33;--accent:#7ea6ff;}
:root[data-theme=light]{--bg:#ffffff;--fg:#1a1a1a;--muted:#5c5c5c;--line:#e2e2e2;--card:#fafafa;
--pos:#15803d;--posbg:#dcfce7;--neg:#7e22ce;--negbg:#f3e8ff;--warn:#b45309;--warnbg:#fef3c7;
--key:#f0f7ff;--accent:#1d4ed8;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;}
.wrap{max-width:1080px;margin:0 auto;padding:2.5rem 1.25rem 5rem}
h1{font-size:1.9rem;line-height:1.25;margin:0 0 .4rem;letter-spacing:-.02em}
h2{font-size:1.3rem;margin:3rem 0 .6rem;padding-bottom:.35rem;border-bottom:2px solid var(--line);
letter-spacing:-.01em}
h3{font-size:1.05rem;margin:2rem 0 .5rem;color:var(--fg)}
.lede{color:var(--muted);font-size:1.05rem;margin:0 0 .5rem}
.meta{color:var(--muted);font-size:.85rem;margin-bottom:2rem}
p{margin:.8rem 0}
figure{margin:1.5rem 0}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
figcaption{color:var(--muted);font-size:.86rem;margin-top:.5rem}
.tablewrap{overflow-x:auto;margin:1.2rem 0}
table{border-collapse:collapse;width:100%;font-size:.88rem;min-width:600px}
caption{text-align:left;font-weight:600;padding-bottom:.5rem;font-size:.92rem}
th,td{padding:.5rem .6rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
thead th{font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);
border-bottom:2px solid var(--line);white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.diff{font-weight:600}
tr.row-key{background:var(--key)}
.sub{color:var(--muted);font-size:.78rem;font-weight:400}
.sessions{font-variant-numeric:tabular-nums;font-size:.84rem;color:var(--muted)}
.chip{display:inline-block;padding:.12rem .5rem;border-radius:999px;font-size:.8rem;
white-space:nowrap;margin:.1rem .15rem .1rem 0}
.chip-pos{background:var(--posbg);color:var(--pos)}
.chip-neg{background:var(--negbg);color:var(--neg)}
.chip-null{color:var(--muted);font-style:italic}
.chip .p{opacity:.75;font-size:.75rem}
.callout{background:var(--card);border-left:4px solid var(--accent);padding:.9rem 1.1rem;
border-radius:0 8px 8px 0;margin:1.3rem 0}
.callout.warn{border-left-color:var(--warn);background:var(--warnbg)}
.callout h4{margin:0 0 .4rem;font-size:.95rem}
.callout p{margin:.4rem 0}
.verdict{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:.9rem;margin:1.4rem 0}
.vcard{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1rem}
.vcard .label{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.vcard .value{font-size:1.5rem;font-weight:650;margin:.25rem 0;letter-spacing:-.02em}
.vcard .note{font-size:.83rem;color:var(--muted)}
code{background:var(--card);padding:.1rem .35rem;border-radius:4px;font-size:.85em;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
ul,ol{padding-left:1.3rem}
li{margin:.35rem 0}
hr{border:0;border-top:1px solid var(--line);margin:2.5rem 0}
"""

    verdict = f"""
<div class="verdict">
  <div class="vcard"><div class="label">Torque, time axis</div>
    <div class="value">2 clusters</div>
    <div class="note">IK&gt;NH {t_time['clusters'][0]['start']:g}–{t_time['clusters'][0]['end']:g}%
    and NH&gt;IK {t_time['clusters'][1]['start']:g}–{t_time['clusters'][1]['end']:g}% — a biphasic story</div></div>
  <div class="vcard"><div class="label">Torque, knee-angle axis</div>
    <div class="value">1 cluster</div>
    <div class="note">NH&gt;IK across the whole {grid['hi']:g}–{grid['lo']:g}° range, p &lt; 0.001.
    The "IK stronger early" cluster is gone</div></div>
  <div class="vcard"><div class="label">Angle mismatch at the same node</div>
    <div class="value">{mapping['max_angle_gap_deg']:.1f}°</div>
    <div class="note">peak difference in knee angle between the groups at an identical
    time node (node {mapping['max_angle_gap_node']})</div></div>
  <div class="vcard"><div class="label">Velocity crossover</div>
    <div class="value">~62°</div>
    <div class="note">moves from ≈75% of duration to 62° of flexion — roughly a third of the
    way through the range, not the final fifth</div></div>
</div>"""

    return f"""<title>jHamON — SPM abscissa: time versus knee angle</title>
<style>{css}</style>
<div class="wrap">
<h1>Should the SPM analyses be normalised by time or by knee angle?</h1>
<p class="lede">Both SPM families re-run on three abscissae over identical repetitions.
The between-modality torque conclusion changes qualitatively; the progression conclusion
mostly does not.</p>
<p class="meta">jHamON training study · NH vs IK · branch <code>angle-normalised-spm</code> ·
generated by <code>compare_axes.py</code> from <code>axis_comparison.json</code></p>

{verdict}

<h2>1. Why the axis matters here</h2>
<p>Every curve in the caches is linearly <b>time</b>-normalised to 101 nodes, and every SPM
result in the project is computed on that axis. Time is only a fair common reference if both
modalities advance through their range of motion at the same rate. They do not — and that is
by design, not by accident.</p>
<p>A Nordic descent is an accelerating fall: the participant resists almost isometrically at
first, then gives way. An isokinetic repetition is held at constant angular velocity by the
dynamometer. Measured on the group-mean trajectories, the Nordic angle-versus-time curve
departs from a straight line by <b>{mapping['NH']['max_deviation_deg']:.1f}°</b>
({mapping['NH']['max_deviation_pct_rom']:.1f}% of its range of motion); the isokinetic one by
<b>{mapping['IK']['max_deviation_deg']:.1f}°</b> ({mapping['IK']['max_deviation_pct_rom']:.1f}%,
essentially the dynamometer's acceleration and deceleration ramps).</p>

<figure>
  <img src="{data_uri(fig_dir / 'axis_mapping.png')}" alt="Knee angle against repetition duration for each group, and the angle mismatch between them">
  <figcaption><b>Left:</b> how knee angle advances with time in each group; dashed lines show
  what a constant-rate movement would look like. <b>Right:</b> the difference in knee angle
  between the groups at the same time node — the quantity that gets confounded with modality.</figcaption>
</figure>

<div class="tablewrap">{mapping_table(mapping)}</div>

<div class="callout warn">
  <h4>The consequence</h4>
  <p>At the midpoint of the repetition the Nordic knee is at
  {mapping['NH']['angle_at_node'].get('50', mapping['NH']['angle_at_node'].get(50)):.1f}° and the
  isokinetic knee at {mapping['IK']['angle_at_node'].get('50', mapping['IK']['angle_at_node'].get(50)):.1f}°.
  Nordic participants have travelled about a fifth of their range; isokinetic participants
  about half. Knee-flexor torque is strongly angle-dependent, so a between-modality test on
  the time axis compares torque at one joint angle against torque at another and reports the
  difference as an effect of modality.</p>
</div>

<h2>2. What changes: between-modality contrast</h2>
<p>All three abscissae are run on the <b>same repetitions</b> — the intersection of those
registrable on every axis ({between_none['torque']['_n_shared_repetitions']:,} for torque) — so a moved cluster cannot be a
changed sample. <code>rom_percent</code> is the control: it has the same node count as the time
axis, truncates nothing and drops no repetition, so any difference against <code>time</code>
isolates the abscissa from every other change.</p>

<figure>
  <img src="{data_uri(fig_dir / 'axis_comparison.png')}" alt="Group mean torque and velocity curves on the time axis and the knee-angle axis, with SPM clusters shaded">
  <figcaption>Identical repetitions, two abscissae. Shading marks supra-threshold SPM{{t}}
  clusters (green NH&gt;IK, purple IK&gt;NH); bands are ±1 SD between participants.</figcaption>
</figure>

<div class="tablewrap">{between_table(between_none['torque'], 'Torque')}</div>

<div class="callout">
  <h4>The biphasic torque result does not survive</h4>
  <p>On the time axis the curves cross at about 42% and produce two clusters — read as
  "isokinetic torque is higher early, Nordic torque higher late". On the knee-angle axis there
  is a single cluster: <b>NH &gt; IK across the entire {grid['hi']:g}–{grid['lo']:g}° range</b>,
  p &lt; 0.001. The Nordic curve is never below the isokinetic one at a matched joint angle.</p>
  <p>The control confirms this is re-alignment and not truncation: on
  <code>rom_percent</code>, which keeps the whole movement, the IK&gt;NH cluster shrinks from
  {t_time['clusters'][0]['end']:g}% of the curve to just {t_romp['clusters'][0]['end']:g}%, and
  the NH&gt;IK cluster expands to
  {t_romp['clusters'][1]['start']:g}–{t_romp['clusters'][1]['end']:g}%.</p>
  <p>Mechanistically the artefact is easy to state: over the first quarter of the repetition
  a Nordic participant has barely moved and is still in the <em>onset transient</em> of the
  torque trace, while the dynamometer has already swept ~21° and reached its plateau. The time
  axis lines those two phases up and calls the difference an effect of modality.</p>
</div>

<div class="tablewrap">{between_table(between_none['velocity'], 'Knee angular velocity')}</div>

<p>For velocity the direction of the conclusion survives but its <b>location</b> moves
substantially. On the time axis the crossover sits at roughly 75% of the duration, which reads
as "the Nordic group is faster only in the final fifth". On the angle axis the crossover is at
about 62° of flexion — roughly a third of the way through the range — so the Nordic group
exceeds the prescribed isokinetic speed over most of the movement. That bears directly on the
velocity-matching claim the design rests on.</p>

<div class="tablewrap">{matched_table(data['matched_angle_values'])}</div>

<h2>3. Robustness</h2>
<h3>Does the grid choice drive it?</h3>
<p>The absolute-angle grid necessarily discards repetitions that do not span it, so the
window is a coverage-versus-interpretability trade-off. The primary
{grid['hi']:g}–{grid['lo']:g}° window keeps 97% of Nordic and 100% of isokinetic repetitions.
The torque conclusion is identical on every window tried, including ones that keep only 42%
of the Nordic repetitions.</p>
<div class="tablewrap">{grid_table(data['grid_sensitivity']['torque'], 'Torque')}</div>
<div class="tablewrap">{grid_table(data['grid_sensitivity']['velocity'], 'Knee angular velocity')}</div>

<h3>Does the exclusion rule drive it?</h3>
<p>No. With no exclusions at all and with the corrected repetition-level exclusions, the
between-modality clusters agree to within a few tenths of a percent:</p>
<ul>
  <li>Torque, knee angle — no exclusions: {esc(t_angle['summary'])}; corrected exclusions:
      {esc(between_corr['torque']['knee_angle']['summary'])}</li>
  <li>Velocity, knee angle — no exclusions: {esc(v_angle['summary'])}; corrected exclusions:
      {esc(between_corr['velocity']['knee_angle']['summary'])}</li>
</ul>

<h3>Is the angle trace invertible?</h3>
<p>Angle normalisation requires inverting angle(<i>t</i>), so the trace must not reverse.
It effectively never does: the median repetition has
{data['monotonicity']['NH']['median']:.4f} of its range in upward travel (NH) and
{data['monotonicity']['IK']['median']:.4f} (IK), and only
{data['monotonicity']['NH']['pct_above_10pct']:.2f}% of Nordic and
{data['monotonicity']['IK']['pct_above_10pct']:.2f}% of isokinetic repetitions exceed 10%.</p>

<h2>4. What changes: within-group progression</h2>
<p>Here the answer is more mixed, and the honest summary is that <b>the axis matters much less
than it does for the between-modality contrast</b>.</p>

<div class="tablewrap">{progression_table(data['progression'], 'NH', 'unc', 'α = 0.05 uncorrected')}</div>
<div class="tablewrap">{progression_table(data['progression'], 'IK', 'unc', 'α = 0.05 uncorrected')}</div>
<div class="tablewrap">{progression_table(data['progression'], 'NH', 'bonf', 'α = 0.05/14 Bonferroni')}</div>
<div class="tablewrap">{progression_table(data['progression'], 'IK', 'bonf', 'α = 0.05/14 Bonferroni')}</div>

<div class="callout">
  <h4>Reading these</h4>
  <p><b>Nordic:</b> the count of significant sessions and the onset of a sustained effect are
  the same on time and on angle (9/14, sustained from session 10). What changes is
  <em>where</em> the effect is located and how interpretable that location is: the time axis
  puts the session-1-vs-15 cluster at 85–97% of the duration, the angle axis puts it at
  62–35° of flexion — a statement about the joint rather than about the clock.</p>
  <p><b>Isokinetic:</b> time and <code>rom_percent</code> agree exactly (13/14), which is
  precisely what should happen when the dynamometer fixes the tempo and the two axes are
  near-affine. The drop to 7/14 on the absolute-angle grid is mostly <b>truncation</b>: the
  time-axis clusters extend to 100%, i.e. into the terminal ramp that the
  {grid['hi']:g}–{grid['lo']:g}° window excludes. This is a real cost of the absolute grid and
  should be reported as such rather than presented as a weaker effect.</p>
  <p><b>Both groups, Bonferroni:</b> the correction the Methods promise leaves very little
  standing on any axis (NH 3/14 → 2/14, IK 8/14 → 3/14). That is a finding about the
  correction, not about the abscissa.</p>
</div>

<h2>5. Two code defects found on the way</h2>
<p>Both are in <code>jhamon_training/data/utils.py:dame_manual_exclusions</code> and both are
corrected on this branch, with the original behaviour retained behind
<code>legacy=True</code> so published numbers stay reproducible.</p>
<ol>
  <li><b>Exclusions were scoped to a variable, not to a repetition.</b> The filters collect the
  offending rows' <code>all_labels</code>, but <code>frames.py</code> builds that label from
  <code>par trses set rep var tr_group</code> — the variable name is part of it. Only the
  flagged variable's rows were dropped, so a Nordic repetition rejected for an impossible
  starting knee angle kept contributing its torque curve. On the current caches that meant
  12 827 Nordic <code>knee_ROM</code> rows removed against 2 727 <code>torque</code> rows:
  three different effective samples depending on which variable an analysis read.</li>
  <li><b>The isokinetic angle threshold predated the sign convention.</b>
  <code>IKfilt_knee2</code> tested <code>knee_ROM</code> at node 100 for <code>&gt; -70</code>,
  written when isokinetic angles were stored negative. <code>frames.ikt_todf</code> now emits
  <code>90 - abs(value)</code> (≈ +11.7° at the end), so every value passes the test: the
  filter matched <b>100% of isokinetic repetitions</b> (9 932 of 9 932) and deleted the entire
  isokinetic angle dataset. Defect 1 concealed it — isokinetic torque and velocity survived
  untouched, so nothing downstream failed. It only surfaces when an analysis needs isokinetic
  <code>knee_ROM</code>, which angle normalisation does.</li>
</ol>
<p>A third criterion, <code>knee_ROM</code> at node 50 <code>&gt; 50</code> for the isokinetic
group, was deliberately <b>not</b> carried over. It was also written for the old convention,
under which it was inert and has never excluded a repetition in any published run. Under the
current convention it would sit at the exact midpoint of a normal sweep
(a linear 88.9 → 11.7° descent passes 50.3° halfway) and discard 9.1% of ordinary
repetitions — reinstating it would invent a behaviour rather than restore one.</p>
<p>Net effect of the correction: Nordic {excl['baseline_repetitions']['NH']:,} →
{excl['corrected']['repetitions']['NH']:,} repetitions, isokinetic
{excl['baseline_repetitions']['IK']:,} → {excl['corrected']['repetitions']['IK']:,}, applied
consistently across every variable.</p>

<h2>6. Recommendation</h2>
<ul>
  <li><b>Between modalities: make knee angle primary.</b> The claim is about the mechanical
  stimulus delivered to the hamstrings, hamstring torque is angle-dependent, and the two
  modalities traverse the range at deliberately different rates. Report the
  {grid['hi']:g}–{grid['lo']:g}° window (97%/100% coverage) with <code>rom_percent</code> as
  the sensitivity check.</li>
  <li><b>Keep the time axis as a supplementary result</b>, framed as what it is — a
  description of the temporal profile. It remains the right axis for anything about rate of
  force development or tempo.</li>
  <li><b>Progression: report on angle for interpretability</b>, but state plainly that the
  conclusion is axis-insensitive for the Nordic group and that the isokinetic drop on the
  absolute grid is truncation. Do not present it as a change in findings.</li>
  <li><b>The Results sentence quoting "isokinetic torque ≈59% higher over the first 28%"
  must go.</b> It is an artefact of comparing a torque onset transient against an established
  plateau.</li>
  <li><b>The Methods paragraph needs rewriting.</b> It currently concedes that "matched
  percentages do not correspond to identical knee angles" and then reports the angular ranges
  alongside each result. Mapping cluster boundaries to angles after the fact does not repair
  the inference — the test itself was computed at mismatched angles.</li>
</ul>

<div class="callout warn">
  <h4>Outstanding: angular resolution</h4>
  <p>The caches store only the 101-point time-normalised curves, so the angle registrations
  here are resampled from those rather than from raw signal. Where a Nordic descent is fastest
  — the last tenth of its duration — consecutive time nodes are about 3° apart, so angular
  resolution near full extension is coarser than the 1° grid implies. Everything above is
  robust to the grid, but a definitive version should re-derive the curves from
  <code>/Volumes/AJMA</code> at full sample rate. <code>jhamon_training.curves</code> is
  agnostic about provenance, so that requires no change to the analysis code.</p>
</div>
</div>"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input-dir", default="reports/axis_comparison")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    in_dir = Path(args.input_dir).expanduser()
    data = json.loads((in_dir / "axis_comparison.json").read_text())
    out = Path(args.output) if args.output else in_dir / "axis_comparison.html"
    out.write_text(build(data, in_dir / "figures"), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, self-contained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
