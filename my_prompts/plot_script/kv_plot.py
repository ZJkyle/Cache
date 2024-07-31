import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Parameters
layers = 32
dim = 1024
seq = 5065
total_lines_per_layer = 1 * dim * seq

# Load the total data
file_path = "my_prompts/output_kv/wiki_value_float.csv"  # Update this with the correct file path
total_data = pd.read_csv(file_path)

# Loop through each layer
for layer in range(layers):
    offset = layer * total_lines_per_layer
    data = total_data.iloc[offset : offset + total_lines_per_layer]
    data["value"] = data["value"].astype(np.uint16).view(np.float16)

    L = data["layer"].values
    X = data["channel"].values
    Y = data["sequence"].values
    Z = data["value"].values

    print(f"Layer {layer} - Min Value: {Z.min()}, Max Value: {Z.max()}")

    # Normalize Z values to 0-1 range
    Z_normalized = (Z - Z.min()) / (Z.max() - Z.min()) * 1

    # Create a grid
    X_unique = np.unique(X)
    Y_unique = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(X_unique, Y_unique)
    Z_grid = np.zeros_like(X_grid, dtype=float)

    # Fill the grid with normalized Z values
    for i in range(seq):
        for j in range(dim):
            idx = i * dim + j
            Z_grid[i, j] = Z_normalized[idx]

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap="coolwarm")

    # Add a color bar which maps values to colors
    mappable = plt.cm.ScalarMappable(cmap="coolwarm")
    mappable.set_array(Z_normalized)
    fig.colorbar(
        mappable,
        ax=ax,
        shrink=0.5,
        aspect=5,
        label="Normalized Value (0-1)",
        location="left",
    )

    # Label the axes
    ax.set_xlabel("Channel")
    ax.set_ylabel("Sequence")
    ax.set_zlabel("Normalized Value")

    # Set title
    ax.set_title(f"Layer {layer}")

    # Set the Z-axis limit
    ax.set_zlim([0, 1])  # Ensuring the z-axis starts from 0

    # Save the plot
    filename = f"my_prompts/output_png/value/value_layer_{layer}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print("Done: ", filename)
