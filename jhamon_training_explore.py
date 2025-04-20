# This script generates results of the functional data recorded during the
# Nordic Hamstring and Isokinetic training sessions.

import matplotlib.pyplot as plt
from pathlib import Path
from jhamon_training import check_result_file
from jhamon_training.data import frames
import os
from jhamon_training.data.utils import dame_manual_exclusions
from jhamon_training.saveload import save_obj
import jhamon_training.stats.training as spmtr
import jhamon_training.stats.spm as spm
import pandas as pd
import numpy as np
from scipy import integrate  # Add import for scipy.integrate

pathtodata = Path("/Volumes/AJMA/")
results_output_path = Path.home() / "Desktop" / "_RESULTS_TRAINING"

# NORDIC training sessions
nordict = check_result_file(
    pathtodata,
    results_output_path,
    participant_id="jhamon01",
    res_file="nht_results.pkl",
)
nordf = frames.nht_todf(my_dict=nordict)


# Explore the data - Plot all repetitions
plt.figure(figsize=(12, 8))

# Counter for repetitions with knee_ROM data
found_count = 0

# Iterate through all training sessions, sets, and repetitions
for tr_key, tr_data in nordict["jhamon01"].items():
    for set_key, set_data in tr_data.items():
        for rep_key, rep_data in set_data.items():
            for rep_idx, rep_values in rep_data[1].items():
                if isinstance(rep_values, dict) and "knee_ROM" in rep_values:
                    found_count += 1
                    knee_ROM = rep_values["knee_ROM"]
                    # Use indices as x-axis
                    indices = np.arange(len(knee_ROM))
                    # Plot with a label to identify the repetition
                    plt.plot(
                        indices,
                        knee_ROM,
                        alpha=0.7,
                        label=f"{tr_key}-{set_key}-{rep_key}-{rep_idx}",
                    )

print(f"Found {found_count} repetitions with knee_ROM data")

plt.xlabel("Index")
plt.ylabel("Knee Angle (ROM)")
plt.title("Knee Angle for All Repetitions")
plt.grid(True, linestyle="--", alpha=0.7)
# Add legend if needed (comment out if there are too many repetitions)
# plt.legend(loc='best', fontsize='small')
# plt.tight_layout()
plt.show()
