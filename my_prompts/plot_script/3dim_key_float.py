import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# 读取CSV文件
df_float = pd.read_csv(
    "my_prompts/output_kv/key_float.csv",
    header=None,
    names=["layer", "embedding", "sequence", "value"],
)


# 计算每个维度下的分布并计算熵值
def calculate_entropy(df, group_by):
    dist = (
        df.groupby(group_by)["value"]
        .apply(lambda x: x.value_counts(normalize=True))
        .unstack(fill_value=0)
    )
    return dist.apply(lambda x: entropy(x, base=2), axis=1)


entropy_layer = calculate_entropy(df_float, ["layer"])
entropy_embedding = calculate_entropy(df_float, ["embedding"])
entropy_sequence = calculate_entropy(df_float, ["sequence"])

# 创建一个DataFrame来存储所有的熵值
entropy_df = pd.DataFrame(
    {
        "Layer": entropy_layer,
        "Embedding": entropy_embedding,
        "Sequence": entropy_sequence,
    }
)

# 打印熵值结果以检查
print(entropy_df)

# 绘制每个维度的熵值柱状图
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# Layer熵值图
axs[0].bar(entropy_layer.index, entropy_layer.values, color="blue")
axs[0].set_title("Entropy of Layer Float Values")
axs[0].set_ylabel("Entropy (bits per element)")
axs[0].set_xlabel("Layer Index")

# Embedding熵值图
axs[1].bar(entropy_embedding.index, entropy_embedding.values, color="orange")
axs[1].set_title("Entropy of Embedding Float Values")
axs[1].set_ylabel("Entropy (bits per element)")
axs[1].set_xlabel("Embedding Index")

# Sequence熵值图
axs[2].bar(entropy_sequence.index, entropy_sequence.values, color="green")
axs[2].set_title("Entropy of Sequence Float Values")
axs[2].set_ylabel("Entropy (bits per element)")
axs[2].set_xlabel("Sequence Index")

plt.tight_layout()

# 保存图像
plt.savefig("my_prompts/outputpng/entropy_float_comparison.png")
plt.close()

# 绘制每个维度的原始数值分布图
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# Layer原始数值分布图
axs[0].scatter(df_float["layer"], df_float["value"], color="blue", alpha=0.7)
axs[0].set_title("Value Distribution by Layer")
axs[0].set_ylabel("Value")
axs[0].set_xlabel("Layer Index")

# Embedding原始数值分布图
axs[1].scatter(df_float["embedding"], df_float["value"], color="orange", alpha=0.7)
axs[1].set_title("Value Distribution by Embedding")
axs[1].set_ylabel("Value")
axs[1].set_xlabel("Embedding Index")

# Sequence原始数值分布图
axs[2].scatter(df_float["sequence"], df_float["value"], color="green", alpha=0.7)
axs[2].set_title("Value Distribution by Sequence")
axs[2].set_ylabel("Value")
axs[2].set_xlabel("Sequence Index")

plt.tight_layout()

# 保存图像
plt.savefig("my_prompts/outputpng/value_distribution_float_comparison.png")
plt.close()

# 额外输出一张只有第一条 sequence, layer, embedding 合在一起的图
first_sequence_layer_embedding = df_float[
    (df_float["layer"] == df_float["layer"].min())
    & (df_float["embedding"] == df_float["embedding"].min())
    & (df_float["sequence"] == df_float["sequence"].min())
]

plt.figure(figsize=(12, 6))
plt.bar(
    ["Layer", "Embedding", "Sequence"],
    [entropy_layer.iloc[0], entropy_embedding.iloc[0], entropy_sequence.iloc[0]],
    color=["blue", "orange", "green"],
)
plt.title("Entropy for First Layer, Embedding, and Sequence")
plt.ylabel("Entropy (bits per element)")
plt.xlabel("Dimension")

# 保存图像
plt.tight_layout()
plt.savefig("my_prompts/outputpng/first_layer_embedding_sequence_entropy.png")
plt.close()
