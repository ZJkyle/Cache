import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

data = np.loadtxt("./my_prompts/output_kv/uint8_kv.txt", dtype=int)

j = data[:, 0]
qs = data[:, 1]

# 按每128行分组
group_size = 64
groups = np.arange(len(j)) // group_size

# 归一化量值以映射到颜色和透明度
norm_qs = (qs - np.min(qs)) / (np.max(qs) - np.min(qs))

# 使用颜色映射
colors = cm.coolwarm(norm_qs)
alpha = norm_qs  # 透明度和颜色强度成正比

# 准备数据用于绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 绘制沿z轴的直线
for group in np.unique(groups):
    mask = groups == group
    xs = j[mask]
    ys = np.full_like(xs, group)
    zs = qs[mask]
    color = colors[mask]
    for x, y, z, c, a in zip(xs, ys, zs, color, alpha[mask]):
        ax.plot([x, x], [y, y], [0, z], color=c, alpha=a)

ax.set_xlabel("j (Element in qk)")
ax.set_ylabel("Group")
ax.set_zlabel("Quantized Value")

# 添加颜色条
norm = plt.Normalize(np.min(qs), np.max(qs))
sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.1)
cbar.set_label("Quantized Value")

# 保存图像
plt.savefig("./my_prompts/outputpng/quantized_values_plot_colored.png")
plt.show()
