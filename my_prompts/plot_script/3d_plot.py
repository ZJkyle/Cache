import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Function to plot each layer
def plot_layer(layer_data, layer_number):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = layer_data["embedding"]
    y = layer_data["sequence_id"]
    z = layer_data["value"]
    c = layer_data["value"]

    # Normalize the color mapping
    norm = plt.Normalize(vmin=min(c), vmax=max(c))
    colors = plt.cm.coolwarm(norm(c))

    # Plot lines along the z-axis
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [y[i], y[i]], [0, z[i]], color=colors[i])

    ax.set_xlabel("Embedding")
    ax.set_ylabel("Sequence ID")
    ax.set_zlabel("Value")
    ax.set_title(f"Layer {layer_number}")

    # Adding color bar
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("Magnitude")

    plt.savefig(f"layer_{layer_number}.png")
    plt.close()


# Read the CSV file without header and assign column names
file_path = "my_prompts/output_kv/rope_key_float.csv"  # 替換為實際的文件路徑
data = pd.read_csv(
    file_path, header=None, names=["layer", "embedding", "sequence_id", "value"]
)

# Group by layer and plot
layers = data["layer"].unique()
for layer in layers:
    layer_data = data[data["layer"] == layer]
    plot_layer(layer_data, layer)
