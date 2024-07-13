import csv
import numpy as np
import heapq


class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def read_csv(filename):
    data = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(x) for x in row])
    return data


def linear_quantize(data, group_index, n_levels=128):
    groups = {}
    for row in data:
        key = row[group_index]
        if key not in groups:
            groups[key] = []
        groups[key].append(row[3])  # value 位于索引3

    quantized_results = []
    for key, values in groups.items():
        values = np.array(values)
        min_val = np.min(values)
        max_val = np.max(values)
        range_val = max_val - min_val

        # 线性量化
        quantized_values = np.round(((values - min_val) / range_val) * (n_levels - 1))
        quantized_results.extend(quantized_values)

    return np.array(quantized_results, dtype=np.int32)


def build_huffman_tree(char_freq):
    heap = []
    for char, freq in char_freq.items():
        heapq.heappush(heap, Node(char, freq))

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]


def generate_codes(node, prefix="", code={}):
    if node.char is not None:
        code[node.char] = prefix
    else:
        generate_codes(node.left, prefix + "0", code)
        generate_codes(node.right, prefix + "1", code)
    return code


def huffman_encode(data, codes):
    return "".join(codes[char] for char in data)


def calculate_compression_ratio(original_data, encoded_data):
    original_size = len(original_data) * 32  # each quantized value is an int32
    encoded_size = len(encoded_data)  # number of bits in encoded data
    return original_size / encoded_size


def analyze_quantization_and_compression(data):
    quantized_data = linear_quantize(
        data, 1
    )  # Assuming quantizing on the first group index
    compression_ratios = []

    for i in range(0, len(quantized_data), 128):
        chunk = quantized_data[i : i + 128]
        if len(chunk) < 128:
            continue  # Skip the last chunk if it's not full

        char_freq = {char: list(chunk).count(char) for char in set(chunk)}
        huffman_tree = build_huffman_tree(char_freq)
        huffman_codes = generate_codes(huffman_tree)
        encoded = huffman_encode(chunk, huffman_codes)
        compression_ratio = calculate_compression_ratio(chunk, encoded)
        compression_ratios.append(compression_ratio)

    average_compression_ratio = np.mean(compression_ratios)
    print(f"Average Compression Ratio: {average_compression_ratio:.2f}")


if __name__ == "__main__":
    filename = "my_prompts/output_kv/tmp.csv"  # replace with your CSV file path
    data = read_csv(filename)
    analyze_quantization_and_compression(data)
