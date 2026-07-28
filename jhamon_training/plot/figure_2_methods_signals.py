"""Figure 2 (Methods) — representative repetition signals + movement photos.

Manuscript Figure 2, laid out as a 2x2 grid (one row per modality):

    A  NH photo   B  NH signals vs time
    C  IK photo   D  IK signals vs time

This is a *methods* illustration: it shows the raw signals recorded during one
representative repetition (knee torque, knee flexion angle, knee angular
velocity vs. time) and the analysed eccentric window (onset -> offset). It does
NOT annotate computed outcomes (mean/peak torque, work, etc.) or group results
— those belong in the results figures. A single shared colour legend serves
both signal panels.

Design notes
------------
* A single (individual) representative repetition is used, not a group average:
  the panel shows the real signal on a real time axis (seconds); averaging
  across time-normalised repetitions (done for the *results* curves) would
  smear the trace.
* Knee angle is plotted as flexion angle, 0 deg = full extension (manuscript
  convention). The IK dynamometer channel is recorded relative to the 90 deg
  flexed start, so it is mapped via ``IK_START_FLEXION_DEG + recorded``.
* Academic styling: Helvetica, a colour-blind-safe (Okabe-Ito-derived) palette,
  thin outward ticks, no top spine, subtle grid — set in ``ACADEMIC_RC``.
* ``load_nh_example`` / ``load_ik_example`` still compute the discrete outcomes
  (returned under ``"vars"``) for reference / figure captions, even though they
  are no longer drawn on the figure.

Data (bundled in ``plot/_example_data/`` for reproducibility without the volume):
* ``NH_repetition_data_example.npy`` — per-rep ``D`` matrix (jhamon01/tr_14).
  Cols: 0 t(s), 1 force ch1(N), 2 force(N), 3 knee angle(rad), 4 hip angle(rad),
  5 knee ang. vel(rad/s), 6 hip ang. vel, 7-8 accel, 9-20 marker (x,y). NH torque
  = force * lever_arm (lever arm from the shank markers, as in analizame_curvas).
* ``IK_repetition_data_example.npy`` — 3 x N at 256 Hz (jhamon29/tr_12), ~400
  samples padding each side. Rows: 0 torque(N.m), 1 ang. vel(deg/s), 2 position(deg).
* ``NH_img.png`` / ``IK_img.png`` — movement photos (swap for final versions).

Run as a script to regenerate ``reports/Figure_2_methods_signals.{png,pdf}``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from jhamon_training.signal.filters import butter_low_filter
from jhamon_training.signal.mech import _detect_onset

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
NH_SIGNAL = "torque"  # {"torque", "force"} — plot NH torque (N.m) by default
NH_FS = 100  # Hz
IK_FS = 256  # Hz
IK_PAD = 400  # samples of baseline padding stored on each side of the IK rep
IK_START_FLEXION_DEG = 90.0  # knee flexion at the IK start (0 deg = extension)

# Shared axis limits for cross-panel comparability.
TORQUE_LIM = (0, 210)
ANGLE_LIM = (0, 100)

# --------------------------------------------------------------------------- #
# Academic style: typography, colour-blind-safe palette, minimal chrome
# --------------------------------------------------------------------------- #
INK = "#242424"        # text, spines, ticks
GRID = "#E4E4E4"       # subtle grid
WINDOW = "#9AA0A6"     # neutral shading of the analysed repetition window

# This figure is the one panel in the paper that colours something other than
# group: three signal channels sharing a time base, each keyed to its own axis.
# The hues are therefore chosen to sit clear of the two group hues in
# ``plot/paper/style.py`` (NH burgundy #A62B36, IK teal #008B85), so that a
# reader does not carry "teal = IK" out of Figure 3 and into a velocity trace
# here -- the more so because panel B is the NH exercise and panel D the IK one.
#
# The velocity channel used to be a bluish green (#0E7C61), which sat at a
# normal-vision deltaE of only 6.1 from the IK teal: the same colour, to a
# reader. Plum moves it to 22.7. The triad is also slightly safer internally
# than the one it replaces (worst all-pairs CVD deltaE 9.0 vs 8.2, normal 17.4
# vs 17.1), and the worst signal-vs-group pair is now 12.0 (orange vs burgundy)
# rather than 6.1.
C_TORQUE = "#1A6BB5"   # blue  — torque / force
C_ANGLE = "#C4571B"    # orange — knee angle
C_VEL = "#8E4585"      # plum   — knee angular velocity

ACADEMIC_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlepad": 8,
    "axes.labelsize": 9.5,
    "axes.labelpad": 3,
    "axes.linewidth": 0.8,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.axisbelow": True,
    "text.color": INK,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 400,
}

_DATA_DIR = Path(__file__).resolve().parent / "_example_data"
_VOLUME_DIR = Path("/Volumes/jHamON/_RESULTS")


# --------------------------------------------------------------------------- #
# Data loading + variable computation
# --------------------------------------------------------------------------- #
def _resolve(filename: str) -> Path:
    """Return the bundled example file, falling back to the raw-data volume."""
    local = _DATA_DIR / filename
    if local.exists():
        return local
    volume = _VOLUME_DIR / filename
    if volume.exists():
        return volume
    raise FileNotFoundError(f"Could not find {filename} in {_DATA_DIR} or {_VOLUME_DIR}.")


def load_nh_example() -> dict:
    """Load the NH example repetition; compute reported variables + window.

    Mirrors the segmentation logic of ``data.nordic.analizame_curvas``.
    """
    D = np.load(_resolve("NH_repetition_data_example.npy"))
    time = D[:, 0]
    force = D[:, 2]                            # load-cell force (N)
    knee_angle = D[:, 3] * 180 / np.pi         # deg flexion (0 = extension)
    knee_vel = np.abs(D[:, 5]) * 180 / np.pi   # deg/s, |angular velocity|

    # Lever arm from the shank markers (marker4 -> marker6) at the first frame.
    shank_length = float(np.hypot(*(D[0, [19, 20]] - D[0, [15, 16]])))
    lever_arm = shank_length - 0.05
    torque = force * lever_arm                 # N.m

    # Repetition window: onset (last neg. dF/dt before threshold) -> peak velocity.
    force_filt = butter_low_filter(force, fs=NH_FS, cut_hz=3, order=2)
    dif_force = np.diff(force_filt)
    onset_1 = _detect_onset(
        force_filt, max(force_filt) * 0.25, n_above=80, n_below=0, show=False
    )[0, 0]
    onset_precise = dif_force[: onset_1 + 5] < 0
    force_onset = 0 if np.where(onset_precise)[0].size == 0 else np.where(onset_precise)[0][-1]
    if force[force_onset] > max(force_filt) * 0.25:
        force_onset = np.where(onset_precise)[0][-2]
    force_offset = _detect_onset(
        force_filt, max(force_filt) * 0.3, n_above=80, n_below=0, show=False
    )[0, 1]
    offset_idx = D[force_onset : force_offset + 25, 5].argmax() + force_onset

    rep = slice(int(force_onset), int(offset_idx))
    seg = D[rep, :]
    peak_idx = int(force_onset) + int(seg[:, 2].argmax())
    knee_rom = abs(seg[0, 3] - seg[-1, 3]) * 180 / np.pi
    work = float(np.abs(np.trapezoid(seg[:, 2] * lever_arm, seg[:, 3])))

    signal = torque if NH_SIGNAL == "torque" else force
    return {
        "time": time, "signal": signal,
        "signal_label": "Knee torque (N·m)" if NH_SIGNAL == "torque" else "Force (N)",
        "knee_angle": knee_angle, "knee_vel": knee_vel,
        "rep": rep, "work_torque": torque,
        "idx": {"onset": int(force_onset), "offset": int(offset_idx), "peak": peak_idx},
        "vars": {
            "tor_mean": seg[:, 2].mean() * lever_arm,
            "tor_peak": seg[:, 2].max() * lever_arm,
            "angle_at_peak": abs(knee_angle[peak_idx]),
            "knee_ROM": knee_rom,
            "v_mean": float(np.abs(seg[:, 5]).mean() * 180 / np.pi),
            "work": work,
        },
    }


def load_ik_example() -> dict:
    """Load the IK example repetition; compute reported variables + window."""
    IK = np.load(_resolve("IK_repetition_data_example.npy"))
    torque, velocity, position = IK[0], IK[1], IK[2]
    n = IK.shape[1]
    time = np.arange(n) / IK_FS

    knee_angle = IK_START_FLEXION_DEG + position   # deg flexion (0 = extension)
    knee_vel = -velocity                           # positive during eccentric

    onset, offset = IK_PAD, n - IK_PAD
    rep = slice(onset, offset)
    peak_idx = onset + int(torque[rep].argmax())
    knee_rom = abs(knee_angle[offset] - knee_angle[onset])
    work = float(np.abs(np.trapezoid(torque[rep], np.deg2rad(position[rep]))))

    return {
        "time": time, "signal": torque, "signal_label": "Knee torque (N·m)",
        "knee_angle": knee_angle, "knee_vel": knee_vel,
        "rep": rep, "work_torque": torque,
        "idx": {"onset": onset, "offset": offset, "peak": peak_idx},
        "vars": {
            "tor_mean": torque[rep].mean(),
            "tor_peak": torque[rep].max(),
            "angle_at_peak": knee_angle[peak_idx],
            "knee_ROM": knee_rom,
            "v_mean": float(np.abs(knee_vel[rep]).mean()),
            "work": work,
        },
    }


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def _make_spine_invisible(ax) -> None:
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _triple_axes(host):
    """Two offset right-hand twin axes on ``host`` (knee angle, knee velocity)."""
    par_angle = host.twinx()
    par_vel = host.twinx()
    par_vel.spines["right"].set_position(("axes", 1.16))
    _make_spine_invisible(par_vel)
    par_vel.spines["right"].set_visible(True)
    return par_angle, par_vel


def _panel_letter(ax, letter, *, xy=(-0.13, 1.04)):
    ax.text(*xy, letter, transform=ax.transAxes, fontsize=14, fontweight="bold",
            va="bottom", ha="right", color=INK)


def _plot_signal_panel(host, data, letter, title):
    """Signal-vs-time panel with three colour-coded axes (methods illustration).

    Shows only the raw signals and the analysed eccentric window (onset/offset);
    no computed-outcome annotations — those belong in the results figures.
    Returns the (torque, angle, velocity) line handles for a shared legend.
    """
    par_angle, par_vel = _triple_axes(host)
    t, idx = data["time"], data["idx"]

    host.axvspan(t[idx["onset"]], t[idx["offset"]], color=WINDOW, alpha=0.12, zorder=0,
                 lw=0)
    host.set_axisbelow(True)
    host.grid(axis="y", color=GRID, lw=0.7, zorder=0)

    (l_tor,) = host.plot(t, data["signal"], color=C_TORQUE, label="Knee torque")
    (l_ang,) = par_angle.plot(t, data["knee_angle"], color=C_ANGLE, label="Knee angle")
    (l_vel,) = par_vel.plot(t, data["knee_vel"], color=C_VEL,
                            label="Knee angular velocity")

    for key in ("onset", "offset"):
        host.axvline(t[idx[key]], color=INK, ls=(0, (4, 3)), lw=0.9, zorder=1)
        host.annotate(key, xy=(t[idx[key]], 0.5), xycoords=("data", "axes fraction"),
                      xytext=(-3, 0), textcoords="offset points", rotation=90,
                      va="center", ha="right", fontsize=7.5, color=INK)

    host.set_xlabel("Time (s)")
    _style_signal_axes(host, par_angle, par_vel, data)
    host.set_title(title, loc="left")
    _panel_letter(host, letter)
    return l_tor, l_ang, l_vel


def _style_signal_axes(host, par_angle, par_vel, data):
    host.set_ylabel(data["signal_label"], color=C_TORQUE)
    par_angle.set_ylabel("Knee flexion angle (°)", color=C_ANGLE)
    par_vel.set_ylabel("Knee angular velocity (°/s)", color=C_VEL)

    host.set_ylim(*TORQUE_LIM)
    par_angle.set_ylim(*ANGLE_LIM)
    host.set_xlim(data["time"][0], data["time"][-1])
    host.spines["top"].set_visible(False)
    par_angle.spines["top"].set_visible(False)

    host.tick_params(axis="y", colors=C_TORQUE)
    par_angle.tick_params(axis="y", colors=C_ANGLE)
    par_vel.tick_params(axis="y", colors=C_VEL)
    for ax, c in ((host, C_TORQUE), (par_angle, C_ANGLE), (par_vel, C_VEL)):
        ax.spines["right"].set_color(c)
        ax.spines["left"].set_color(C_TORQUE)


def _plot_photo(ax, filename, letter):
    try:
        ax.imshow(mpimg.imread(_resolve(filename)))
    except FileNotFoundError:
        ax.text(0.5, 0.5, f"[{filename}]\nplaceholder", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#9AA0A6")
        ax.set_facecolor("#F5F5F5")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    _panel_letter(ax, letter, xy=(0.0, 1.0))


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #
def _shared_legend(fig, handles):
    """One colour legend for the whole figure (top centre)."""
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=10, handlelength=1.7,
               columnspacing=2.4, borderaxespad=0.0)


def make_figure(save_dir: Path | None = None, *, with_photos: bool = True):
    """Build Figure 2 (methods illustration). Returns the Figure.

    ``with_photos`` toggles the movement-photo column (2x2) vs. signals only.
    """
    nh, ik = load_nh_example(), load_ik_example()

    with mpl.rc_context(ACADEMIC_RC):
        if with_photos:
            fig = plt.figure(figsize=(12.5, 10.0))
            gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.18],
                                  wspace=0.22, hspace=0.26)
            _plot_photo(fig.add_subplot(gs[0, 0]), "NH_img.png", "A")
            handles = _plot_signal_panel(fig.add_subplot(gs[0, 1]), nh, "B",
                                         "Nordic hamstring exercise (NH)")
            _plot_photo(fig.add_subplot(gs[1, 0]), "IK_img.png", "C")
            _plot_signal_panel(fig.add_subplot(gs[1, 1]), ik, "D",
                               "Isokinetic eccentric (IK)")
            fig.subplots_adjust(left=0.015, right=0.88, top=0.92, bottom=0.05)
        else:
            fig = plt.figure(figsize=(8.6, 9.2))
            handles = _plot_signal_panel(fig.add_subplot(2, 1, 1), nh, "A",
                                         "Nordic hamstring exercise (NH)")
            _plot_signal_panel(fig.add_subplot(2, 1, 2), ik, "B",
                               "Isokinetic eccentric (IK)")
            fig.subplots_adjust(left=0.10, right=0.82, top=0.91, bottom=0.07, hspace=0.32)

        _shared_legend(fig, handles)
        fig.set_facecolor("white")
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            for ext in ("png", "pdf"):
                out = save_dir / f"Figure_2_methods_signals.{ext}"
                fig.savefig(out, bbox_inches="tight", facecolor="white")
                print(f"Saved {out}")
    return fig


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    make_figure(save_dir=repo_root / "reports")
    plt.show()
