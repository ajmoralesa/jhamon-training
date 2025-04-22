import pandas as pd
import numpy as np
from scipy import integrate
from typing import Dict, Any


def calculate_ik_discrete_variables(ikdf: pd.DataFrame) -> pd.DataFrame:
    """Calculates discrete variables (work, mean/peak torque, angle at peak torque)
    for each repetition in the IK dataframe.

    Args:
        ikdf: DataFrame containing filtered IK training data.

    Returns:
        DataFrame with discrete variables per repetition.
    """
    # Filter necessary data
    iktor = ikdf[ikdf["var"] == "torque"].copy()
    ikrom = ikdf[ikdf["var"] == "knee_ROM"].copy()
    ikvel = ikdf[ikdf["var"] == "knee_v"].copy()  # Add knee velocity filter

    # --- Calculate Work ---
    tor_wide = iktor[["timepoint", "all_labels", "value"]].pivot(
        index="timepoint", columns="all_labels", values="value"
    )
    rom_wide = ikrom[["timepoint", "all_labels", "value"]].pivot(
        index="timepoint", columns="all_labels", values="value"
    )

    # Calculate work using scipy.integrate.trapezoid
    aa = np.abs(
        integrate.trapezoid(tor_wide.values, x=np.deg2rad(rom_wide.values), axis=0)
    )

    # Explicitly create a MultiIndex for clarity and type checking
    iktor_index_df = iktor[iktor["timepoint"] == 0][
        ["par", "trses", "set", "rep"]
    ].drop_duplicates()
    ikwork_index = pd.MultiIndex.from_frame(iktor_index_df)

    # Create DataFrame with work values
    ikwork = pd.DataFrame(aa, index=ikwork_index, columns=["work"])

    # --- Calculate other discrete variables ---
    # Group torque data by repetition
    grouped_torque = iktor.groupby(["par", "trses", "set", "rep"])

    # Group velocity data by repetition
    grouped_velocity = ikvel.groupby(["par", "trses", "set", "rep"])

    # Calculate mean velocity
    mean_velocity = grouped_velocity["value"].mean().rename("knee_v_mean")

    # 1. Mean Torque
    mean_torque = grouped_torque["value"].mean().rename("mean_torque")

    # 2. Peak Torque and Timepoint of Peak Torque
    # Find the original DataFrame index corresponding to the max torque value within each group
    idx_peak_torque = grouped_torque["value"].idxmax()
    # Retrieve the peak torque value and the timepoint it occurred at using the original index
    peak_torque_info = iktor.loc[idx_peak_torque, ["value", "timepoint"]].rename(
        columns={"value": "peak_torque"}
    )
    # Set the index of peak_torque_info to match the group keys for joining
    peak_torque_info.index = pd.MultiIndex.from_tuples(
        grouped_torque.groups.keys(), names=["par", "trses", "set", "rep"]
    )

    # 4. Angle at Peak Torque
    # Create a lookup index using the timepoints at peak torque
    ikrom_indexed = ikrom.set_index(["par", "trses", "set", "rep", "timepoint"])

    # Create multi-index tuples for lookup
    lookup_index_tuples = []

    # Get the timepoints and convert the index to a list of tuples
    timepoints = peak_torque_info["timepoint"].values
    group_keys = list(map(tuple, peak_torque_info.index.tolist()))

    # Create lookup tuples by combining each group key with its timepoint
    for i, group_key in enumerate(group_keys):
        try:
            timepoint = int(timepoints[i])
            # Ensure group_key is a tuple of 4 elements (par, trses, set, rep)
            if len(group_key) == 4:
                lookup_index_tuples.append((*group_key, timepoint))
            else:
                print(
                    f"Warning: Expected group_key of length 4, got {len(group_key)}: {group_key}"
                )
        except (TypeError, ValueError, IndexError) as e:
            print(f"Warning: Could not create lookup index for {group_key}: {e}")

    if lookup_index_tuples:
        lookup_multi_index = pd.MultiIndex.from_tuples(
            lookup_index_tuples, names=["par", "trses", "set", "rep", "timepoint"]
        )

        # Look up the angle values
        try:
            angle_at_peak = ikrom_indexed.loc[
                ikrom_indexed.index.intersection(lookup_multi_index), "value"
            ]
            angle_at_peak = angle_at_peak.rename("angle_at_peak_torque")
            # Reset index to match the other DataFrames (drop timepoint)
            angle_at_peak.index = angle_at_peak.index.droplevel("timepoint")
        except Exception as e:
            print(f"Warning: Error looking up angles at peak torque: {e}")
            angle_at_peak = pd.Series(
                np.nan, index=peak_torque_info.index, name="angle_at_peak_torque"
            )
    else:
        print("Warning: No valid lookup indices created for angle at peak torque.")
        angle_at_peak = pd.Series(
            np.nan, index=peak_torque_info.index, name="angle_at_peak_torque"
        )

    # Calculate knee_ROM for each repetition
    knee_ROM = (
        ikrom.groupby(["par", "trses", "set", "rep"])["value"]
        .agg(lambda x: x.max() - x.min())
        .rename("knee_ROM")
    )

    # --- Combine all discrete variables ---
    # Start with ikwork
    ik_discrete = ikwork.copy()

    # Join the other calculated variables
    ik_discrete = ik_discrete.join(mean_torque, how="left")
    ik_discrete = ik_discrete.join(peak_torque_info["peak_torque"], how="left")
    ik_discrete = ik_discrete.join(angle_at_peak, how="left")
    ik_discrete = ik_discrete.join(mean_velocity, how="left")
    ik_discrete = ik_discrete.join(knee_ROM, how="left")

    # Reset index to make par, trses, set, rep regular columns
    ik_discrete = ik_discrete.reset_index()

    # Add tr_group column to identify IK data
    ik_discrete["tr_group"] = "IK"

    print(f"Calculated IK discrete variables. Shape: {ik_discrete.shape}")
    return ik_discrete
