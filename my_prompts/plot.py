import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
data = pd.read_csv(
    "./output_kv/tmp_k.csv", header=None, names=["Embedding", "T_L_ID", "Value"]
)

# Convert columns to appropriate types, handling errors
data["Embedding"] = pd.to_numeric(data["Embedding"], errors="coerce")
data["T_L_ID"] = pd.to_numeric(data["T_L_ID"], errors="coerce")
data["Value"] = pd.to_numeric(data["Value"], errors="coerce")

# Drop rows with invalid data
data.dropna(inplace=True)

# Check the first few rows of the dataframe to ensure correct loading
print(data.head())

# Sort data by Embedding to ensure lines are drawn correctly
data.sort_values(by=["Embedding"], inplace=True)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Line plot
ax.plot(data["Embedding"], data["T_L_ID"], data["Value"], color="b")

# Set labels
ax.set_xlabel("Embedding")
ax.set_ylabel("T_L_ID")
ax.set_zlabel("Value")

# Save plot to file
plt.savefig("3d_line_plot.png")
