import numpy as np
import os
from collections import Counter
from bitstring import BitStream, BitArray

data = np.loadtxt("./my_prompts/output_kv/uint8_kv.txt", dtype=int)

values = data[:, 1]


def delta_encode(data):
    return np.diff(data, prepend=data[0])


def delta_decode(data):
    return np.cumsum(data)


# 分组
group_size = 1024
groups = [values[i : i + group_size] for i in range(0, len(values), group_size)]

# 对每组数据进行差分编码
delta_encoded_groups = [delta_encode(group) for group in groups]


# 算术编码相关函数
class ArithmeticEncoder:
    def __init__(self, precision=32):
        self.precision = precision
        self.half = 1 << (self.precision - 1)
        self.quarter = 1 << (self.precision - 2)
        self.three_quarters = self.half + self.quarter
        self.high = (1 << self.precision) - 1
        self.low = 0
        self.follow_bits = 0
        self.output = BitStream()

    def update(self, symbol, cum_freq):
        total = cum_freq[-1]
        range_ = self.high - self.low + 1
        self.high = self.low + (range_ * cum_freq[symbol + 1]) // total - 1
        self.low = self.low + (range_ * cum_freq[symbol]) // total

        while True:
            if self.high < self.half:
                self._bit_plus_follow(0)
            elif self.low >= self.half:
                self._bit_plus_follow(1)
                self.low -= self.half
                self.high -= self.half
            elif self.low >= self.quarter and self.high < self.three_quarters:
                self.follow_bits += 1
                self.low -= self.quarter
                self.high -= self.quarter
            else:
                break
            self.high = (self.high << 1) + 1
            self.low = self.low << 1

    def _bit_plus_follow(self, bit):
        self.output.append(f"uint:1={bit}")
        for _ in range(self.follow_bits):
            self.output.append(f"uint:1={1 - bit}")
        self.follow_bits = 0

    def finish(self):
        self.follow_bits += 1
        if self.low < self.quarter:
            self._bit_plus_follow(0)
        else:
            self._bit_plus_follow(1)
        # 确保位数是8的倍数
        remaining_bits = (8 - len(self.output) % 8) % 8
        if remaining_bits > 0:
            self.output.append(f"uint:{remaining_bits}=0")


def arithmetic_encoding(data):
    frequency = Counter(data)
    symbols = sorted(frequency.keys())
    cum_freq = [0]
    for symbol in symbols:
        cum_freq.append(cum_freq[-1] + frequency[symbol])

    encoder = ArithmeticEncoder()
    for symbol in data:
        encoder.update(symbols.index(symbol), cum_freq)
    encoder.finish()
    return encoder.output.bytes, symbols, cum_freq


def arithmetic_decoding(encoded_data, symbols, cum_freq, original_size):
    decoder = BitStream(encoded_data)
    low = 0
    high = (1 << 32) - 1
    value = int(decoder.read("uint:32"))
    total = cum_freq[-1]
    decoded_data = []

    for _ in range(original_size):
        range_ = high - low + 1
        cum_value = ((value - low + 1) * total - 1) // range_
        symbol_index = (
            next(i for i, freq in enumerate(cum_freq) if freq > cum_value) - 1
        )
        symbol = symbols[symbol_index]
        decoded_data.append(symbol)
        high = low + (range_ * cum_freq[symbol_index + 1]) // total - 1
        low = low + (range_ * cum_freq[symbol_index]) // total

        while True:
            if high < (1 << 31):
                pass
            elif low >= (1 << 31):
                value -= 1 << 31
                low -= 1 << 31
                high -= 1 << 31
            elif low >= (1 << 30) and high < (3 << 30):
                value -= 1 << 30
                low -= 1 << 30
                high -= 1 << 30
            else:
                break
            low = low << 1
            high = (high << 1) + 1
            if decoder.pos < decoder.len:
                value = (value << 1) + int(decoder.read("uint:1"))
            else:
                value = value << 1

    return np.array(decoded_data)


# 压缩每组数据
compressed_groups = []
symbols_list = []
cum_freq_list = []
for group in delta_encoded_groups:
    encoded_data, symbols, cum_freq = arithmetic_encoding(group)
    compressed_groups.append(encoded_data)
    symbols_list.append(symbols)
    cum_freq_list.append(cum_freq)

# 保存压缩数据到文件
compressed_file_path = "./my_prompts/output_kv/compressed_data.bin"
with open(compressed_file_path, "wb") as f:
    for encoded_data in compressed_groups:
        f.write(encoded_data)

# 显示压缩效果
original_size = len(values) * 8  # 每个值用8位表示
compressed_size = os.path.getsize(compressed_file_path) * 8
compression_ratio = compressed_size / original_size
print(f"Original size: {original_size} bits")
print(f"Compressed size: {compressed_size} bits")
print(f"Compression ratio: {compression_ratio:.2f}")

# 示例解压缩
first_compressed_group = compressed_groups[0]
decoded_group = arithmetic_decoding(
    first_compressed_group, symbols_list[0], cum_freq_list[0], len(groups[0])
)
decoded_group = delta_decode(decoded_group)
print(f"Decoded first group (original vs decoded):")
print(groups[0][:10])
print(decoded_group[:10])
