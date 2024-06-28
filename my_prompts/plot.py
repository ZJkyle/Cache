# import pandas as pd
# import matplotlib.pyplot as plt
# import os
#
# # Load data
# data = pd.read_csv(
#     "./output_kv/tmp_k.csv", header=None, names=["Embedding", "T_L_ID", "Value"]
# )
#
# # Convert columns to appropriate types, handling errors
# data["Embedding"] = pd.to_numeric(data["Embedding"], errors="coerce")
# data["T_L_ID"] = pd.to_numeric(data["T_L_ID"], errors="coerce")
# data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
#
# # Drop rows with invalid data
# data.dropna(inplace=True)
#
# # Sort data by Embedding to ensure lines are drawn correctly
# data.sort_values(by=["Embedding"], inplace=True)
#
# # Calculate the first quartile value
# first_quartile_value = data["Value"].quantile(0.25)
#
# # Filter data to only include values greater than the first quartile
# filtered_data = data[data["Value"] > first_quartile_value]
#
# # Create output directory for plots
# output_dir = "./T_L_ID_plots"
# os.makedirs(output_dir, exist_ok=True)
#
# # Plot for each T_L_ID
# for t_l_id, group_data in filtered_data.groupby("T_L_ID"):
#     plt.figure()
#     plt.scatter(group_data["Embedding"], group_data["Value"], s=1)
#     plt.xlabel("Embedding")
#     plt.ylabel("Value")
#     plt.title(f"T_L_ID: {t_l_id}")
#     plt.savefig(os.path.join(output_dir, f"T_L_ID_{t_l_id}.png"))
#     plt.close()
#
# print(f"Plots saved in directory: {output_dir}")
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from matplotlib.colors import Normalize, to_rgba

# Load data and convert columns to appropriate types
data = pd.read_csv(
    "./output_kv/tmp_k.csv",
    header=None,
    names=["Embedding", "T_L_ID", "Value"],
    dtype={"Embedding": "int", "T_L_ID": "int", "Value": "float32"},
)

# Drop rows with invalid data
data.dropna(inplace=True)

# Sort data by Embedding to ensure lines are drawn correctly
data.sort_values(by=["Embedding"], inplace=True)

# Calculate the first quartile value
first_quartile_value = data["Value"].quantile(0.25)

# Filter data to only include values greater than the first quartile
filtered_data = data[data["Value"] > first_quartile_value]

# Create output directory for plots
output_dir = "."
os.makedirs(output_dir, exist_ok=True)

# Create a color map
color_map = plt.get_cmap("viridis")

# Normalize Embedding values for color mapping
embedding_norm = Normalize(
    vmin=filtered_data["Embedding"].min(), vmax=filtered_data["Embedding"].max()
)

# Normalize Value for alpha transparency
value_norm = Normalize(
    vmin=filtered_data["Value"].min(), vmax=filtered_data["Value"].max()
)

# Plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for embedding, group_data in filtered_data.groupby("Embedding"):
    t_l_ids = group_data["T_L_ID"]
    values = group_data["Value"]
    norm_value = embedding_norm(embedding)
    color = color_map(norm_value)

    for t_l_id, value in zip(t_l_ids, values):
        alpha = value_norm(value)
        line_color = to_rgba(color, alpha=alpha)  # Adjust transparency based on value
        ax.plot([embedding, embedding], [t_l_id, t_l_id], [0, value], color=line_color)

# Set labels
ax.set_xlabel("Embedding")
ax.set_ylabel("T_L_ID")
ax.set_zlabel("Value")

# Save plot to file
plt.savefig(os.path.join(output_dir, "3d_plot.png"))

# Show plot
plt.show()

print(f"Plot saved in directory: {output_dir}")
