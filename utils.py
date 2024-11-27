import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def plot_comparison_with_baseline(
    availability,
    df_learning,
    baseline_learning,
    accuracies=None,
    algorithm="Q-learning",
):
    """
    Compare the given availability and accuracies with the baseline using one plot with multiple y-axes.

    Args:
        availability (float): Teacher availability level to filter the data (as a fraction, e.g., 0.8 for 80%).
        df_learning (pd.DataFrame): DataFrame containing Q-learning or SARSA results.
        baseline_learning (tuple): Baseline values (avg reward, success rate, avg learning speed) for Q-learning or SARSA.
        accuracies (list, optional): List of accuracies to filter by. If None, all accuracies are used.
        algorithm (str, optional): The algorithm type to use for ('Q-learning' or 'SARSA').

    Raises:
        ValueError: If the provided algorithm is not 'Q-learning' or 'SARSA'.
    """
    # Check if the algorithm is Q-learning or SARSA
    if algorithm == "Q-learning" or algorithm == "SARSA":
        df = df_learning
        baseline_reward, baseline_success_rate, baseline_learning_speed = (
            baseline_learning
        )
    else:
        # If the algorithm is neither Q-learning nor SARSA, raise an error
        raise ValueError("Algorithm must be either 'Q-learning' or 'SARSA'")

    # If accuracies are not provided, use all available accuracies in the data
    if accuracies is None:
        accuracies = df["Accuracy"].unique()

    # Filter the DataFrame for the provided availability and the given accuracies
    # We use .copy() to avoid potential warnings about chained assignments
    selected_data = df.loc[
        (df["Availability"] == availability) & (df["Accuracy"].isin(accuracies))
    ].copy()

    # Create a figure and the first axis (ax1) for Avg Reward
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Set figure size to (10, 6)

    # Plot Avg Reward on y-axis 1
    ax1.set_xlabel("Accuracy")  # Set x-axis label
    ax1.set_ylabel(
        "Avg Reward", color="tab:blue"
    )  # Set y-axis label for Avg Reward with blue color
    (line1,) = ax1.plot(
        selected_data["Accuracy"],  # X-axis values: Accuracy
        selected_data["Avg Reward"],  # Y-axis values: Avg Reward
        marker="o",  # Use circle markers for data points
        markersize=8,  # Set marker size
        color="tab:blue",  # Set line color to blue
        linestyle="solid",  # Solid line style
        label=f"Avg Reward (Availability: {availability}, {algorithm})",  # Line label for legend
    )
    # Plot the baseline Avg Reward as a horizontal dashed line
    line2 = ax1.axhline(
        baseline_reward,  # Baseline Avg Reward (constant)
        color="tab:blue",  # Blue line
        linestyle="dashed",  # Dashed line style
        linewidth=2,  # Line width
        label=f"Baseline Avg Reward ({algorithm})",  # Label for the baseline
    )
    ax1.tick_params(
        axis="y", labelcolor="tab:blue"
    )  # Set tick parameters for y-axis 1 (blue color)

    # Ensure that x-ticks are based on the provided accuracy values
    plt.xticks(
        selected_data["Accuracy"]
    )  # Only show x-ticks for the provided accuracy values

    # Create a second y-axis (ax2) for Success Rate
    ax2 = ax1.twinx()  # Create a second axis that shares the same x-axis as ax1
    ax2.set_ylabel(
        "Success Rate (%)", color="tab:green"
    )  # Set y-axis label for Success Rate (green)
    (line3,) = ax2.plot(
        selected_data["Accuracy"],  # X-axis values: Accuracy
        selected_data["Success Rate (%)"],  # Y-axis values: Success Rate
        marker="s",  # Use square markers for data points
        markersize=8,  # Set marker size
        color="tab:green",  # Green color for Success Rate
        linestyle="solid",  # Solid line style
        label=f"Success Rate (Availability: {availability}, {algorithm})",  # Line label for legend
    )
    # Plot the baseline Success Rate as a dotted line
    line4 = ax2.axhline(
        baseline_success_rate,  # Baseline Success Rate (constant)
        color="tab:green",  # Green color for Success Rate
        linestyle="dotted",  # Dotted line style
        linewidth=2,  # Line width
        label=f"Baseline Success Rate ({algorithm})",  # Label for the baseline
    )
    ax2.tick_params(
        axis="y", labelcolor="tab:green"
    )  # Set tick parameters for y-axis 2 (green color)

    # Create a third y-axis (ax3) for Avg Learning Speed
    ax3 = ax1.twinx()  # Create a third axis that shares the same x-axis as ax1
    ax3.spines["right"].set_position(
        ("outward", 60)
    )  # Offset the third axis outward by 60 pixels
    ax3.set_ylabel(
        "Avg Learning Speed", color="tab:orange"
    )  # Set y-axis label for Learning Speed (orange)
    (line5,) = ax3.plot(
        selected_data["Accuracy"],  # X-axis values: Accuracy
        selected_data["Avg Learning Speed"],  # Y-axis values: Avg Learning Speed
        marker="^",  # Use triangle markers for data points
        markersize=8,  # Set marker size
        color="tab:orange",  # Orange color for Avg Learning Speed
        linestyle="solid",  # Solid line style
        label=f"Learning Speed (Availability: {availability}, {algorithm})",  # Line label for legend
    )
    # Plot the baseline Avg Learning Speed as a dashdot line
    line6 = ax3.axhline(
        baseline_learning_speed,  # Baseline Avg Learning Speed (constant)
        color="tab:orange",  # Orange color for Learning Speed
        linestyle="dashdot",  # Dashdot line style
        linewidth=2,  # Line width
        label=f"Baseline Learning Speed ({algorithm})",  # Label for the baseline
    )
    ax3.tick_params(
        axis="y", labelcolor="tab:orange"
    )  # Set tick parameters for y-axis 3 (orange color)

    # Add a title to the plot
    fig.suptitle(
        f"Comparison with Baseline for {availability * 100}% Availability ({algorithm})",  # Title includes availability percentage and algorithm
        fontsize=14,  # Set font size for the title
    )
    ax1.grid(True)  # Enable grid for the plot

    # Custom legend elements
    custom_lines = [
        Line2D(
            [0],
            [0],
            color="tab:blue",
            linestyle="solid",
            marker="o",
            label="Avg Reward",
        ),  # Avg Reward line
        Line2D(
            [0], [0], color="tab:blue", linestyle="dashed", label="Baseline Avg Reward"
        ),  # Baseline Avg Reward
        Line2D(
            [0],
            [0],
            color="tab:green",
            linestyle="solid",
            marker="s",
            label="Success Rate",
        ),  # Success Rate line
        Line2D(
            [0],
            [0],
            color="tab:green",
            linestyle="dotted",
            label="Baseline Success Rate",
        ),  # Baseline Success Rate
        Line2D(
            [0],
            [0],
            color="tab:orange",
            linestyle="solid",
            marker="^",
            label="Learning Speed",
        ),  # Learning Speed line
        Line2D(
            [0],
            [0],
            color="tab:orange",
            linestyle="dashdot",
            label="Baseline Learning Speed",
        ),  # Baseline Learning Speed
    ]

    # Create a custom legend with the desired layout (2 rows, 3 columns)
    fig.legend(
        custom_lines,
        [
            "Avg Reward",
            "Baseline Avg Reward",  # First row: Avg Reward and Baseline
            "Success Rate",
            "Baseline Success Rate",  # Second row: Success Rate and Baseline
            "Learning Speed",
            "Baseline Learning Speed",  # Third row: Learning Speed and Baseline
        ],
        loc="upper center",  # Position the legend at the upper center
        bbox_to_anchor=(
            0.5,
            -0.05,
        ),  # Adjust the legend's position closer to the x-axis
        ncol=3,  # Arrange the legend into 3 columns
        frameon=False,  # Disable the box frame around the legend
    )

    # Adjust the layout to fit the legend and remove extra space
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Display the plot
    plt.show()
