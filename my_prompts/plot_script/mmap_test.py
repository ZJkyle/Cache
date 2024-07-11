import mmap
import struct
import numpy as np
import os

# 假設你的 mmap 文件名為 'layer_0_k.dat'
file_name = "kvcache_engine/mmap_data/layer_0_k.dat"

# 獲取當前工作目錄
current_dir = os.getcwd()
file_path = os.path.join(current_dir, file_name)

print(f"Current working directory: {current_dir}")
print(f"File path: {file_path}")

# 檢查文件是否存在
if not os.path.exists(file_path):
    print(f"File {file_path} does not exist.")
else:
    # 打開文件
    with open(file_path, "r+b") as f:
        # 創建 mmap 對象
        mm = mmap.mmap(f.fileno(), 0)

        # 計算 float 的個數
        num_floats = len(mm) // struct.calcsize("f")

        # 使用 struct 模組解碼 float 陣列
        float_array = struct.unpack("f" * num_floats, mm[:])

        # 如果需要使用 numpy 數組
        float_array_np = np.array(float_array)

        # 關閉 mmap 對象
        mm.close()
        # 轉換為 numpy 數組
        float_array_np = np.array(float_array)
        np.set_printoptions(threshold=np.inf)
        # 打印前 1024 個 float
        print(float_array_np[:16385])
