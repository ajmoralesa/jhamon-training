import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from jhamon_training import check_result_file
from jhamon_training.data import frames
from jhamon_training.data.utils import dame_manual_exclusions, filter_training_data
from jhamon_training.stats.analyses_discrete_vars import analyze_discrete_variables


def load_participant_data(
    pathtodata: Path, results_output_path: Path, participant: str
) -> pd.DataFrame:
    """Load data for a specific participant.

    Args:
        pathtodata: Path to the raw data directory
        results_output_path: Path to the results directory
        participant: Participant ID to analyze

    Returns:
        DataFrame containing the participant's data
    """
    # Load Nordic training sessions
    nordict = check_result_file(
        pathtodata, results_output_path, res_file="nht_results.pkl"
    )

    # Load or generate Nordic DataFrame
    nordf_file = results_output_path / "nordf.feather"
    if nordf_file.exists():
        print(f"Loading cached Nordic DataFrame from {nordf_file}")
        nordf = pd.read_feather(nordf_file)
    else:
        print("Generating Nordic DataFrame...")
        nordf = frames.nht_todf(my_dict=nordict)
        print(f"Saving Nordic DataFrame to {nordf_file}")
        nordf.to_feather(nordf_file)

    # Load IK training sessions
    ikdict = check_result_file(
        pathtodata, results_output_path, res_file="ikt_results.pkl"
    )

    # Load or generate IK DataFrame
    ikdf_file = results_output_path / "ikdf.feather"
    if ikdf_file.exists():
        print(f"Loading cached IK DataFrame from {ikdf_file}")
        ikdf = pd.read_feather(ikdf_file)
    else:
        print("Generating IK DataFrame...")
        ikdf = frames.ikt_todf(my_dict=ikdict)
        print(f"Saving IK DataFrame to {ikdf_file}")
        ikdf.to_feather(ikdf_file)

    # Merge both datasets
    training_df = pd.concat([nordf, ikdf])

    # Filter for the specific participant
    participant_data = training_df[training_df["par"] == participant].copy()

    return participant_data


def inspect_data_structure(
    df: pd.DataFrame, participant: str, session: str, set_num: int
):
    """Inspect the structure of the data for debugging.

    Args:
        df: DataFrame containing the participant's data
        participant: Participant ID
        session: Training session
        set_num: Set number
    """
    print("\nData Structure Inspection:")
    print(f"Total rows: {len(df)}")
    print("\nColumns:", df.columns.tolist())
    print("\nUnique values in 'var' column:", df["var"].unique())

    # Filter for specific session and set
    session_data = df[(df["trses"] == session) & (df["set"] == f"set_{set_num}")].copy()

    print(f"\nData for session {session}, set {set_num}:")
    print(f"Number of rows: {len(session_data)}")
    print("\nSample of the data:")
    print(session_data.head())

    # Check data for each variable type
    for var in ["torque", "knee_ROM", "knee_v"]:
        var_data = session_data[session_data["var"] == var]
        print(f"\n{var} data:")
        print(f"Number of rows: {len(var_data)}")
        if len(var_data) > 0:
            print("Sample values:", var_data["value"].head().tolist())


def plot_repetition_data(
    df: pd.DataFrame,
    participant: str,
    session: str,
    set_num: int,
    output_path: Path,
) -> None:
    """Plot raw data for a specific repetition.

    Args:
        df: DataFrame containing the participant's data
        participant: Participant ID
        session: Training session (e.g., 'tr_1')
        set_num: Set number
        output_path: Path where the plot will be saved
    """
    # First inspect the data structure
    inspect_data_structure(df, participant, session, set_num)

    # Filter data for the specific session and set
    session_data = df[(df["trses"] == session) & (df["set"] == f"set_{set_num}")].copy()

    # Get unique repetitions
    reps = sorted(session_data["rep"].unique())
    print(f"\nRepetitions found: {reps}")

    # Set up the figure
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.figsize": (12, 8),
        }
    )

    # Create figure with subplots for each repetition
    fig, axes = plt.subplots(len(reps), 1, sharex=True)
    if len(reps) == 1:
        axes = [axes]

    # Plot each repetition
    for i, rep in enumerate(reps):
        rep_data = session_data[session_data["rep"] == rep]

        # Get data for each variable
        torque_data = rep_data[rep_data["var"] == "torque"]
        rom_data = rep_data[rep_data["var"] == "knee_ROM"]
        velocity_data = rep_data[rep_data["var"] == "knee_v"]

        print(f"\nRepetition {rep} data:")
        print(f"Time points: {len(rep_data['timepoint'].unique())}")
        print(f"ROM points: {len(rom_data)}")
        print(f"Torque points: {len(torque_data)}")
        print(f"Velocity points: {len(velocity_data)}")

        if len(torque_data) > 0 and len(rom_data) > 0:
            # Use timepoint as x-axis
            timepoints = sorted(rep_data["timepoint"].unique())

            # Plot ROM and torque
            ax = axes[i]
            ax.plot(
                timepoints, rom_data["value"].values, label="Knee ROM", color="#1f77b4"
            )
            ax.set_ylabel("ROM (degrees)")

            # Create a second y-axis for torque
            ax2 = ax.twinx()
            ax2.plot(
                timepoints, torque_data["value"].values, label="Torque", color="#ff7f0e"
            )
            ax2.set_ylabel("Torque (Nm)")

            # Add title and grid
            ax.set_title(f"Repetition {rep}")
            ax.grid(True, alpha=0.3)

            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            print(f"Missing data for repetition {rep}")

    # Set common x-label
    axes[-1].set_xlabel("Time (s)")

    # Add main title
    plt.suptitle(f"Participant {participant} - Session {session} - Set {set_num}")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path / f"raw_data_{participant}_{session}_set{set_num}.png")
    plt.close()


def plot_participant_comparison(
    nh_df: pd.DataFrame,
    ik_df: pd.DataFrame,
    nh_participant: str,
    ik_participant: str,
    session: str,
    set_num: int,
    output_path: Path,
) -> None:
    """Plot NH and IK participant data side by side.

    Args:
        nh_df: DataFrame containing the NH participant's data
        ik_df: DataFrame containing the IK participant's data
        nh_participant: NH participant ID
        ik_participant: IK participant ID
        session: Training session (e.g., 'tr_1')
        set_num: Set number
        output_path: Path where the plot will be saved
    """
    # Set up the figure
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.figsize": (15, 6),
        }
    )

    # Create figure with two subplots side by side
    fig, (ax_nh, ax_ik) = plt.subplots(1, 2, sharey=True)

    # Initialize twin axes
    ax_nh_twin = ax_nh.twinx()
    ax_ik_twin = ax_ik.twinx()

    # Plot NH participant data
    nh_session_data = nh_df[
        (nh_df["trses"] == session) & (nh_df["set"] == f"set_{set_num}")
    ].copy()

    nh_reps = sorted(nh_session_data["rep"].unique())
    print(f"\nNH Repetitions found: {nh_reps}")

    # Plot all NH repetitions
    for rep in nh_reps:
        rep_data = nh_session_data[nh_session_data["rep"] == rep]
        torque_data = rep_data[rep_data["var"] == "torque"]
        rom_data = rep_data[rep_data["var"] == "knee_ROM"]

        if len(torque_data) > 0 and len(rom_data) > 0:
            timepoints = sorted(rep_data["timepoint"].unique())
            ax_nh.plot(
                timepoints,
                rom_data["value"].values,
                label=f"ROM Rep {rep}",
                color="#1f77b4",
                alpha=0.7,
            )
            ax_nh_twin.plot(
                timepoints,
                torque_data["value"].values,
                label=f"Torque Rep {rep}",
                color="#ff7f0e",
                alpha=0.7,
            )

    # Plot IK participant data
    ik_session_data = ik_df[
        (ik_df["trses"] == session) & (ik_df["set"] == f"set_{set_num}")
    ].copy()

    ik_reps = sorted(ik_session_data["rep"].unique())
    print(f"\nIK Repetitions found: {ik_reps}")

    # Plot all IK repetitions
    for rep in ik_reps:
        rep_data = ik_session_data[ik_session_data["rep"] == rep]
        torque_data = rep_data[rep_data["var"] == "torque"]
        rom_data = rep_data[rep_data["var"] == "knee_ROM"]

        if len(torque_data) > 0 and len(rom_data) > 0:
            timepoints = sorted(rep_data["timepoint"].unique())
            ax_ik.plot(
                timepoints,
                rom_data["value"].values,
                label=f"ROM Rep {rep}",
                color="#1f77b4",
                alpha=0.7,
            )
            ax_ik_twin.plot(
                timepoints,
                torque_data["value"].values,
                label=f"Torque Rep {rep}",
                color="#ff7f0e",
                alpha=0.7,
            )

    # Set labels and titles
    ax_nh.set_title(f"NH Participant: {nh_participant}")
    ax_ik.set_title(f"IK Participant: {ik_participant}")

    ax_nh.set_xlabel("Time Point")
    ax_ik.set_xlabel("Time Point")

    ax_nh.set_ylabel("ROM (degrees)")
    ax_ik.set_ylabel("ROM (degrees)")

    # Add legends
    lines_nh, labels_nh = ax_nh.get_legend_handles_labels()
    lines_nh_twin, labels_nh_twin = ax_nh_twin.get_legend_handles_labels()
    ax_nh.legend(
        lines_nh + lines_nh_twin,
        labels_nh + labels_nh_twin,
        loc="upper right",
        title="NH Repetitions",
    )

    lines_ik, labels_ik = ax_ik.get_legend_handles_labels()
    lines_ik_twin, labels_ik_twin = ax_ik_twin.get_legend_handles_labels()
    ax_ik.legend(
        lines_ik + lines_ik_twin,
        labels_ik + labels_ik_twin,
        loc="upper right",
        title="IK Repetitions",
    )

    # Add grid
    ax_nh.grid(True, alpha=0.3)
    ax_ik.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        output_path
        / f"comparison_{nh_participant}_{ik_participant}_{session}_set{set_num}.png"
    )
    plt.close()


def main():
    # Set up paths
    pathtodata = Path("/Volumes/AJMA/")
    results_output_path = Path.home() / "Desktop" / "_RESULTS_TRAINING"
    figures_path = results_output_path / "figures"
    figures_path.mkdir(exist_ok=True)

    # Choose participants and session to analyze
    nh_participant = "jhamon01"  # Example NH participant
    ik_participant = "jhamon18"  # Example IK participant
    session = "tr_1"  # Example session
    set_num = 1  # Example set number

    # Load participant data
    nh_df = load_participant_data(pathtodata, results_output_path, nh_participant)
    ik_df = load_participant_data(pathtodata, results_output_path, ik_participant)

    # Plot the comparison
    plot_participant_comparison(
        nh_df, ik_df, nh_participant, ik_participant, session, set_num, figures_path
    )

    # Print some basic information
    print(f"\nAnalyzing data:")
    print(f"NH Participant {nh_participant}:")
    print(f"  Total sessions: {len(nh_df['trses'].unique())}")
    print(
        f"  Sets in session {session}: {len(nh_df[nh_df['trses'] == session]['set'].unique())}"
    )
    print(
        f"  Repetitions in set {set_num}: {len(nh_df[(nh_df['trses'] == session) & (nh_df['set'] == f'set_{set_num}')]['rep'].unique())}"
    )

    print(f"\nIK Participant {ik_participant}:")
    print(f"  Total sessions: {len(ik_df['trses'].unique())}")
    print(
        f"  Sets in session {session}: {len(ik_df[ik_df['trses'] == session]['set'].unique())}"
    )
    print(
        f"  Repetitions in set {set_num}: {len(ik_df[(ik_df['trses'] == session) & (ik_df['set'] == f'set_{set_num}')]['rep'].unique())}"
    )


if __name__ == "__main__":
    main()
