import csv
import numpy as np


# 讀取 CSV 文件
def read_csv(filename):
    data = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row])
    return data


# 對 value 值進行簡單線性縮放
def scale_values(data, group_by):
    # 將數據轉換為 NumPy 數組
    data = np.array(data)

    # 分組的索引位置
    group_by_index = {"embedding": 1, "sequence": 2}

    # 確定要分組的索引
    group_index = group_by_index[group_by]

    # 根據 group_by 維度分組
    groups = {}
    for row in data:
        group_key = row[group_index]
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(row)

    # 縮放每個群體的 value 值
    scaled_data = []
    for group_key in groups:
        group_data = np.array(groups[group_key])
        values = group_data[:, 3]

        # 計算最大值和最小值
        min_val = np.min(values)
        max_val = np.max(values)

        # 線性縮放到 [0, 1] 範圍內
        scaled_values = (values - min_val) / (max_val - min_val)

        # 反縮放
        unscaled_values = scaled_values * (max_val - min_val) + min_val

        # 將縮放後的值存入結果中
        scaled_data.extend(
            zip(group_data[:, 0], group_data[:, 1], group_data[:, 2], unscaled_values)
        )

    return scaled_data


# 主程序
if __name__ == "__main__":
    filename = "your_file.csv"  # 替換為你的 CSV 文件路徑
    data = read_csv(filename)

    # 簡單線性縮放 embedding 維度
    embedding_scaled = scale_values(data, "embedding")

    # 簡單線性縮放 sequence 維度
    sequence_scaled = scale_values(data, "sequence")

    # 輸出縮放後的結果
    print("Embedding Scaled Data:")
    for row in embedding_scaled:
        print(row)

    print("\nSequence Scaled Data:")
    for row in sequence_scaled:
        print(row)
