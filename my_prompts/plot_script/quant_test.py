import csv
import numpy as np
import gzip
import io


# 讀取 CSV 文件
def read_csv(filename):
    data = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(x) for x in row])
    return data


# 非對稱的線性量化
def linear_quantize(data, group_index, n_levels=128):
    groups = {}
    for row in data:
        key = row[group_index]
        if key not in groups:
            groups[key] = []
        groups[key].append(row[3])  # value 位於索引3

    quantized_results = []

    for key, values in groups.items():
        values = np.array(values)
        min_val = np.min(values)
        max_val = np.max(values)
        range_val = max_val - min_val

        # 線性量化
        quantized_values = np.round(((values - min_val) / range_val) * (n_levels - 1))

        quantized_results.extend(quantized_values)

    return np.array(quantized_results, dtype=np.int32)


# 使用 gzip 進行壓縮並計算壓縮率
def compress_data_with_gzip(data):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
        f.write(data)
    compressed_data = buffer.getvalue()
    compression_ratio = len(data) / len(compressed_data)
    return compression_ratio


# 分析量化與壓縮
def analyze_quantization_and_compression(data):
    # 量化
    quantized_embedding = linear_quantize(data, 1)
    quantized_sequence = linear_quantize(data, 2)

    # 壓縮
    embedding_compressed_ratio = compress_data_with_gzip(quantized_embedding.tobytes())
    sequence_compressed_ratio = compress_data_with_gzip(quantized_sequence.tobytes())

    print(
        f"Compression Ratio for Quantized Embedding: {embedding_compressed_ratio:.2f}"
    )
    print(f"Compression Ratio for Quantized Sequence: {sequence_compressed_ratio:.2f}")


# 主程序
if __name__ == "__main__":
    filename = "my_prompts/output_kv/tmp.csv"  # 更換為你的 CSV 檔案路徑
    data = read_csv(filename)

    analyze_quantization_and_compression(data)
