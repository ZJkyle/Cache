import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load the CSV file
df = pd.read_csv("my_prompts/output_kv/wiki_key_float.csv", nrows=32768)

# Step 2: Convert 'value' from uint16 to fp16
df["value"] = df["value"].astype(np.float16)

# Step 3: Normalize 'value' to range [-1, 1]
max_val = df["value"].max()
min_val = df["value"].min()
df["normalized_value"] = 2 * ((df["value"] - min_val) / (max_val - min_val)) - 1

# Step 4: Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot each point as a line from zero to its value
for idx, row in df.iterrows():
    ax.plot(
        [row["channel"], row["channel"]],
        [row["sequence"], row["sequence"]],
        [0, row["normalized_value"]],
        color=plt.cm.coolwarm((row["normalized_value"] + 1) / 2),
        marker="_",
    )

# Color bar setup with -1 to 1 mapping using the coolwarm colormap
norm = plt.Normalize(-1, 1)
cmap = plt.cm.coolwarm
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Add colorbar to the plot, fixing the error by explicitly specifying the axes
cbar = plt.colorbar(sm, ax=ax, pad=0.1)
cbar.set_label("Normalized Value")

# Labeling Axes
ax.set_xlabel("Channel")
ax.set_ylabel("Sequence")
ax.set_zlabel("Normalized Value")

# Save the plot as a PNG file
plt.savefig("my_prompts/outputpng/3d_plot_lines.png", dpi=300)
