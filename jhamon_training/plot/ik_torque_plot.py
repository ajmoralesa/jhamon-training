import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def plot_ik_torque_curves(ikdict: dict, output_path: Path) -> None:
    """
    Generates and saves a plot of Isokinetic training torque curves by session.

    Args:
        ikdict: Dictionary containing the Isokinetic training data.
        output_path: Path object where the plot image will be saved.
    """
    fig, ax = plt.subplots(figsize=(10, 6))  # Use a slightly larger figure

    # Store all torque data grouped by session type (e.g., 'tr_1', 'tr_2')
    all_session_torques = {}
    if ikdict:  # Check if the dictionary is not empty
        for p in ikdict:
            for tr in ikdict[p]:
                for s in ikdict[p][tr]:
                    for r in ikdict[p][tr][s]:
                        torque = ikdict[p][tr][s][r].get("torque")
                        if (
                            torque is not None and len(torque) > 0
                        ):  # Check if torque data exists and is not empty
                            all_session_torques.setdefault(tr, []).append(torque)

    if not all_session_torques:
        print("No IK torque data found to plot.")
        plt.close(fig)  # Close the empty figure
        return

    # Get unique session names and assign colors using a colormap
    # Custom sort key function to extract number and sort numerically
    sort_key = lambda s: int(s.split("_")[1])
    sessions = sorted(list(all_session_torques.keys()), key=sort_key)
    # Using viridis colormap, but others like 'plasma', 'magma', 'cividis' or 'tab10' could be used
    cmap = plt.get_cmap("cividis")  # Get colormap object
    colors = {session: cmap(i / len(sessions)) for i, session in enumerate(sessions)}

    # Plot individual curves with alpha and calculate/plot averages
    for session_name in sessions:
        torque_list = all_session_torques[session_name]
        color = colors[session_name]

        if not torque_list:
            continue  # Skip if no data for this session

        # Plot individual torque curves with alpha
        for torque in torque_list:
            ax.plot(
                torque, color=color, alpha=0.02, zorder=1
            )  # Lower alpha further, explicit zorder

        # Calculate and plot the average torque curve for the session
        # Assumes all torque arrays in torque_list have the same length
        try:
            # Convert list of arrays to a 2D NumPy array for averaging
            torque_array = np.array(torque_list)
            # Check if all arrays had the same length implicitly via conversion success
            if torque_array.ndim == 2:
                mean_torque = np.mean(torque_array, axis=0)
                # Increase linewidth slightly and set higher zorder
                ax.plot(
                    mean_torque,
                    color=color,
                    linewidth=3,
                    label=f"{session_name} Avg",
                    zorder=3,
                )
            else:
                print(
                    f"Warning: Could not calculate average for session {session_name} due to inconsistent data shapes."
                )
                # Optionally: Plot the first rep curve as a placeholder for the legend if needed
                # ax.plot([], [], color=color, linewidth=3, label=f'{session_name} (Avg N/A)', zorder=3)

        except ValueError:
            # Handle cases where arrays might have different lengths explicitly
            print(
                f"Warning: Skipping average calculation for session {session_name} due to varying lengths or invalid data."
            )
            # Optionally: Plot the first rep curve as a placeholder for the legend if needed
            # ax.plot([], [], color=color, linewidth=3, label=f'{session_name} (Avg N/A)', zorder=3)

    # Add labels, title, legend, and grid
    ax.set_xlabel("Timepoints")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title("Isokinetic Training Torque Curves by Session")
    ax.legend(title="Session Averages")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()  # Adjust layout to prevent labels overlapping

    # Ensure the output directory exists
    os.makedirs(output_path.parent, exist_ok=True)
    # Save the figure
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"IK torque plot saved to: {output_path}")
    plt.close(fig)  # Close the figure to free memory
