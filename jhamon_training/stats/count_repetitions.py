import pandas as pd
from pathlib import Path
from jhamon_training import check_result_file
from jhamon_training.data import frames
from jhamon_training.data.utils import dame_manual_exclusions, filter_training_data
from jhamon_training.stats.analyses_discrete_vars import analyze_discrete_variables


def main():
    # Set up paths
    pathtodata = Path("/Volumes/AJMA/")
    results_output_path = Path.home() / "Desktop" / "_RESULTS_TRAINING"

    # Load data
    nordict = check_result_file(
        pathtodata, results_output_path, res_file="nht_results.pkl"
    )
    ikdict = check_result_file(
        pathtodata, results_output_path, res_file="ikt_results.pkl"
    )

    # Create dataframes
    nordf = frames.nht_todf(my_dict=nordict)
    ikdf = frames.ikt_todf(my_dict=ikdict)
    training_df = pd.concat([nordf, ikdf])
    training_dfilt = dame_manual_exclusions(training_df)

    # Get discrete analysis results
    discrete_analysis_results = analyze_discrete_variables(
        nordict=nordict,
        ikdf=ikdf,
        training_df=training_df,
        results_output_path=results_output_path,
    )
    training_disc = discrete_analysis_results["training_disc"]
    training_disc = filter_training_data(training_disc)

    # Count repetitions by group
    nh_reps = len(training_disc[training_disc["tr_group"] == "NH"]["rep"].unique())
    ik_reps = len(training_disc[training_disc["tr_group"] == "IK"]["rep"].unique())
    total_reps = nh_reps + ik_reps

    # Calculate prescribed repetitions
    prescribed_reps_per_participant = 497  # Total prescribed reps per participant
    nh_participants = len(
        training_disc[training_disc["tr_group"] == "NH"]["par"].unique()
    )
    prescribed_total_nh = prescribed_reps_per_participant * nh_participants

    # Calculate percentage of analyzed repetitions
    nh_percentage = (nh_reps / prescribed_total_nh) * 100

    print(f"NH repetitions: {nh_reps}")
    print(f"IK repetitions: {ik_reps}")
    print(f"Total repetitions: {total_reps}")
    print(f"Number of NH participants: {nh_participants}")
    print(f"Total prescribed NH repetitions: {prescribed_total_nh}")
    print(f"Percentage of NH repetitions analyzed: {nh_percentage:.1f}%")


if __name__ == "__main__":
    main()
