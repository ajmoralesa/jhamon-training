import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors

# 1. Create the 'repes' data
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

repes_df = pd.DataFrame(
    {"tr_session": tr_sessions_list, "set": sets_list, "reps": reps_list}
)

# Ensure 'tr_session' is categorical and ordered like R's factor
session_order = [f"tr_{i}" for i in range(1, 16)]
repes_df["tr_session"] = pd.Categorical(
    repes_df["tr_session"], categories=session_order, ordered=True
)

# 2. Create the 'protocol' data
protocol_df = pd.DataFrame(
    {
        "tr_session": session_order,
        "labs": (
            ["3 sets x 5 reps"]
            + ["4 sets x 5 reps"] * 4
            + ["5 sets x 6 reps"] * 3
            + ["5 sets x 8 reps"] * 3
            + ["6 sets x 8 reps"] * 4
        ),
    }
)
protocol_df["tr_session"] = pd.Categorical(
    protocol_df["tr_session"], categories=session_order, ordered=True
)


# 3. Aggregate data for plotting
plot_data_df = (
    repes_df.groupby("tr_session", observed=False)["reps"].sum().reset_index()
)
plot_data_df = plot_data_df.rename(columns={"reps": "totalrep"})

# Join protocol labels
plot_data_df = pd.merge(plot_data_df, protocol_df, on="tr_session")

# 4. Define weekly boundaries and merge week data early
week_mapping_df = pd.DataFrame(
    {
        "tr_session": session_order,
        "week": ([1] + [i for i in range(2, 9) for _ in range(2)]),
    }
)
week_mapping_df["tr_session"] = pd.Categorical(
    week_mapping_df["tr_session"], categories=session_order, ordered=True
)

# Get numeric codes (0-based index) for sessions
week_mapping_df["session_num"] = week_mapping_df["tr_session"].cat.codes

# Join week data to plot_data_df HERE
plot_data_df = pd.merge(
    plot_data_df, week_mapping_df[["tr_session", "week"]], on="tr_session"
)

# Calculate numeric boundaries for rectangles (y-axis after flip)
rect_data_df = (
    week_mapping_df.groupby("week")["session_num"].agg(["min", "max"]).reset_index()
)
# Adjust by 0.5 for full bar coverage, map to ymin/ymax for horizontal span
rect_data_df["ymin"] = rect_data_df["min"] - 0.5
rect_data_df["ymax"] = rect_data_df["max"] + 0.5

# Add alternating fill identifier
rect_data_df["fill_color_idx"] = rect_data_df["week"] % 2

# Important: Create a reversed week mapping for the visual layout
# This maps the original week numbers (1-8) to their display position (8-1)
# First, sort by week to ensure correct ordering
rect_data_df = rect_data_df.sort_values("week")
# Then create the reversed mapping
max_week = rect_data_df["week"].max()
rect_data_df["reversed_week"] = max_week - rect_data_df["week"] + 1
# Now sort by the reversed week to get correct visual ordering (Week 1 at bottom)
rect_data_df = rect_data_df.sort_values("reversed_week")

# Also reverse the ymin/ymax values to match the reversed session ordering
n_sessions = len(session_order)
rect_data_df["display_ymin"] = n_sessions - rect_data_df["ymax"]
rect_data_df["display_ymax"] = n_sessions - rect_data_df["ymin"]

# Create a mapping from week to categorical positions
week_to_sessions = week_mapping_df.groupby("week")["tr_session"].apply(list).to_dict()

# Create a mapping from week to display positions
week_to_positions = {}
for week, sessions in week_to_sessions.items():
    # Convert categorical sessions to numeric positions in the reversed order
    positions = [len(session_order) - 1 - session_order.index(s) for s in sessions]
    week_to_positions[week] = positions

# Get maximum repetitions for label positioning
max_reps = plot_data_df["totalrep"].max()
# Extend the plot width to accommodate labels and avoid overlap
plot_width = max_reps * 1.25  # Add 40% extra width

# 5. Create the plot
sns.set_theme(style="white", context="notebook")
fig, ax = plt.subplots(figsize=(7, 8))  # Adjust figure size

# --- Color mapping for bars based on week ---
# Choose a colormap (e.g., 'viridis', 'Blues', 'YlGnBu', 'cividis')
cmap = plt.get_cmap("cividis")
# Normalize week numbers to 0-1 for the colormap
norm = mcolors.Normalize(plot_data_df["week"].min(), plot_data_df["week"].max())
# Create a dictionary mapping session to color
color_dict = {
    session: cmap(norm(week))
    for session, week in plot_data_df[["tr_session", "week"]].values
}

# Reverse session order for plotting bottom-to-top
reversed_session_order = session_order[::-1]
# Create the list of colors in the reversed plotting order
bar_colors = [color_dict[session] for session in reversed_session_order]
# ---------------------------------------------

# Define background colors
bg_colors = ["#B0B0B0", "#C8C8C8"]  # Darker grey tones

# Add background rectangles using axhspan (since plot will be horizontal)
for week, positions in week_to_positions.items():
    # Calculate min and max y positions for this week
    ymin = min(positions) - 0.5
    ymax = max(positions) + 0.5

    ax.axhspan(
        ymin,
        ymax,
        facecolor=bg_colors[int(week % 2)],
        alpha=0.5,
        # Extend the span to the right (x-direction)
        xmin=0,
        xmax=1,  # This uses normalized coordinates (0-1)
        zorder=1,
    )

    # Add week label directly at the average position
    y_pos = sum(positions) / len(positions)
    ax.text(
        max_reps * 1.15,  # Position further right to avoid overlap
        y_pos,
        f"Week {week}",
        va="center",
        ha="right",
        color="black",
        fontsize=9,
        zorder=2,
    )

# Add bars (use uniform grey color instead of week-based colors)
sns.barplot(
    y="tr_session",
    x="totalrep",
    data=plot_data_df,
    order=reversed_session_order,  # Plot in reversed order
    color="grey",  # Use uniform grey for all bars
    ax=ax,
    zorder=3,  # Bars on top
)

# Add bar labels (protocol labs) INSIDE bars
# Use numpy arange for y-positions corresponding to REVERSED categorical levels
y_positions = np.arange(len(reversed_session_order))
label_map = plot_data_df.set_index("tr_session")["labs"].to_dict()

for i, session in enumerate(reversed_session_order):  # Iterate in plot order
    if session in label_map:  # Check if session exists in data
        label_text = label_map[session]
        ax.text(
            0.5,  # Position label slightly inside from the left edge
            y_positions[i],
            label_text,
            va="center",
            ha="left",  # Align to the left
            color="white",  # Make label white for contrast
            fontsize=10,  # Increased font size (was 8)
            fontweight="bold",  # Make text bold for better visibility
            zorder=4,  # Labels on top of bars
        )

# Style the plot
ax.set_xlabel("Repetitions")
ax.set_ylabel("Training session")

# --- Customize Y-axis ----
# Set ticks to correspond to the number of sessions
ax.set_yticks(np.arange(len(session_order)))
# Set labels to be numeric 1-15 (in reversed order to match plot)
ax.set_yticklabels([str(i) for i in range(len(session_order), 0, -1)])
# -------------------------

ax.grid(False)  # Remove grid lines like theme_bw() + theme(panel.grid...)
sns.despine(ax=ax)  # Remove top and right spines

# Adjust x-axis limits to make room for week labels
ax.set_xlim(0, plot_width)  # Extend x-axis to avoid overlap

# Ensure layout is tight
plt.tight_layout()

# Create reports directory if it doesn't exist
import os

if not os.path.exists("./reports"):
    os.makedirs("./reports")

# Save the plot
plt.savefig("./reports/Figure_1_python.png", dpi=300)  # Save with good resolution

# Optional: Display the plot
# plt.show()

print("Plot saved to ./reports/Figure_1_python.png")
