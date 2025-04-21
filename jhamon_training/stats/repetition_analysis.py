import pandas as pd


def compare_nh_ik_repetitions(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates IK group average repetitions per set/session and compares
    them with the prescribed NH repetitions.

    Args:
        training_df: DataFrame containing the combined training data,
                     including 'tr_group', 'par', 'trses', 'set', 'rep'.

    Returns:
        DataFrame comparing NH prescribed reps vs. IK average reps per set/session.
    """

    # --- IK Group Repetition Analysis ---

    # Filter for IK group and select relevant columns for unique repetition identification
    ik_reps_df = training_df[training_df["tr_group"] == "IK"][
        ["par", "trses", "set", "rep"]
    ].drop_duplicates()

    # Count repetitions per participant per set per session
    grouped_size_participant = ik_reps_df.groupby(["trses", "set", "par"]).size()
    reps_per_participant_set_session = grouped_size_participant.to_frame(  # type: ignore
        name="reps_count"
    ).reset_index()

    # Calculate average and std dev of repetitions per set per session across participants
    avg_reps_per_set_session = (
        reps_per_participant_set_session.groupby(["trses", "set"])["reps_count"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # --- Compare NH vs IK Repetitions per Set/Session ---

    # 1. Create NH prescribed reps DataFrame (logic from figure1_plot.py)
    tr_sessions_list = (
        [f"tr_{1}"] * 3
        + [f"tr_{i}" for i in range(2, 6) for _ in range(4)]
        + [f"tr_{i}" for i in range(6, 12) for _ in range(5)]
        + [f"tr_{i}" for i in range(12, 16) for _ in range(6)]
    )
    sets_list = (
        [f"set_{i}" for i in range(1, 4)]
        + [f"set_{i}" for i in range(1, 5)] * 4
        + [f"set_{i}" for i in range(1, 6)] * 6
        + [f"set_{i}" for i in range(1, 7)] * 4
    )
    reps_list = [5] * 19 + [6] * 15 + [8] * 39

    nh_reps_prescribed_df = pd.DataFrame(
        {"tr_session": tr_sessions_list, "set_str": sets_list, "NH_reps": reps_list}
    )

    # 2. Clean NH Reps DataFrame
    nh_reps_prescribed_df["trses"] = (
        nh_reps_prescribed_df["tr_session"].str.extract(r"(\d+)").astype(int)
    )
    nh_reps_prescribed_df["set"] = (
        nh_reps_prescribed_df["set_str"].str.extract(r"(\d+)").astype(int)
    )
    nh_reps_prescribed_df = nh_reps_prescribed_df[["trses", "set", "NH_reps"]]

    # 3. Prepare IK Reps DataFrame
    ik_reps_avg_df = avg_reps_per_set_session.rename(
        columns={"mean": "IK_mean_reps", "std": "IK_std_reps"}
    )

    # 4. Merge DataFrames
    # --- Debugging: Check columns and types before merge ---
    # print("\n--- NH Prescribed Reps DF Info ---")
    # nh_reps_prescribed_df.info()
    # print(nh_reps_prescribed_df.head())

    # print("\n--- IK Average Reps DF Info ---")
    # Attempt str.extract approach on IK data as requested
    try:
        # print("Attempting str.extract on IK data...")
        ik_reps_avg_df["trses"] = (
            ik_reps_avg_df["trses"].str.extract(r"(\d+)").astype(int)
        )
        ik_reps_avg_df["set"] = ik_reps_avg_df["set"].str.extract(r"(\d+)").astype(int)
        # print("Successfully extracted/cast IK merge columns.")
    except (KeyError, AttributeError, TypeError) as e:
        # print(f"Error during str.extract/casting on IK data: {type(e).__name__}: {e}")
        # print("Attempting direct casting to int instead...")
        try:
            ik_reps_avg_df["trses"] = ik_reps_avg_df["trses"].astype(int)
            ik_reps_avg_df["set"] = ik_reps_avg_df["set"].astype(int)
            # print("Successfully cast IK merge columns directly to int.")
        except (KeyError, TypeError) as e2:
            print(f"Merge column casting failed in IK data: {type(e2).__name__}: {e2}")
            print("Columns available in ik_reps_avg_df:", ik_reps_avg_df.columns)
            # Raise error or return partial data if casting fails? For now, let merge attempt proceed.

    # ik_reps_avg_df.info()
    # print(ik_reps_avg_df.head())
    # print("------------------------------------\n")
    # --- End Debugging ---

    combined_reps_df = pd.merge(
        nh_reps_prescribed_df,
        ik_reps_avg_df,
        on=["trses", "set"],
        how="left",  # Keep all NH prescribed entries
    )

    # Calculate the difference between NH prescribed and IK average reps
    combined_reps_df["reps_diff"] = (
        combined_reps_df["NH_reps"] - combined_reps_df["IK_mean_reps"]
    )

    return combined_reps_df
