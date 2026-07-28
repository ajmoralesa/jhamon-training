"""Publication figures for the NH-vs-IK training standardization paper.

One module per manuscript figure, all sharing ``style.py``. The numbering here
is the manuscript's numbering, and it is the numbering ``jhamon_results.qmd``
calls out in the Results text:

    Figure 1  training programme: sessions, sets, repetitions      (Methods)
    Figure 2  measurement set-up and representative signals        (Methods)
    Figure 3  modality comparison: torque and velocity + SPM{t}    (Results)
    Figure 4  repetitions and mechanical work                      (Results)
    Figure 5  standardization validity: Bland-Altman               (Results)
    Figure 6  training progression and discrete outcomes           (Results)

``make_paper_figures.py`` at the repository root draws all six from the cached
signal-processing results in one command.
"""

from . import style  # noqa: F401
