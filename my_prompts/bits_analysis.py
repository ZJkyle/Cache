import pandas as pd
import numpy as np

file_path = "my_prompts/dump_bits.csv"
with open(file_path, "r") as file:
    data = file.read()

numbers = [float(num) for num in data.replace("\n", ",").split(",") if num.strip()]

series = pd.Series(numbers)

mean_value = series.mean() / 2048
max_value = series.max() / 2048
min_value = series.min() / 2048


print(f"Mean: {mean_value}")
print(f"Max: {max_value}")
print(f"Min: {min_value}")
