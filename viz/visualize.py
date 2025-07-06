import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_heatmap(df):
    avg_runs_array = np.array(df['avg_runs'])
    avg_runs_matrix = avg_runs_array.reshape((16, 32))
    group_array = np.array(df['num_imamiya'])
    group_matrix = group_array.reshape((16, 32))

    fig, ax = plt.subplots(figsize=(12, 6))
    heatmap = sns.heatmap(avg_runs_matrix, annot=True, fmt=".2f", cmap="viridis", cbar=True, ax=ax)
    ax.set_title("Average Runs per Lineup (9-Inning Game)", fontsize=14)
    ax.set_xlabel("Lineup Index (Column)", fontsize=12)
    ax.set_ylabel("Lineup Index (Row)", fontsize=12)

    rows, cols = group_matrix.shape
    for i in range(rows):
        for j in range(cols - 1):
            if group_matrix[i, j] != group_matrix[i, j+1]:
                ax.plot([j+1, j+1], [i, i+1], color='black', lw=2)
    for i in range(rows - 1):
        for j in range(cols):
            if group_matrix[i, j] != group_matrix[i+1, j]:
                ax.plot([j, j+1], [i+1, i+1], color='black', lw=2)

    plt.tight_layout()
    plt.show()
