# perplexity eval
## run /root/datasets/pg19/test/.txt for 50 files 
## ./build/bin/llama-long-context-perplexity -f ../datasets/pg19/test/.txt -m ../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-F16.gguf -t 12
#!/bin/bash

# 定義模型路徑和執行檔
MODEL_PATH="../../../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
BIN_PATH="./../../build/bin/llama-long-context-perplexity"
DATASET_PATH="../../../datasets/pg19/test/"
THREADS=12

# 確認路徑是否存在
if [ ! -d "$DATASET_PATH" ]; then
  echo "資料夾 $DATASET_PATH 不存在！"
  exit 1
fi

# 取得前50個 .txt 檔案
FILES=($(ls "$DATASET_PATH"*.txt | head -n 50))

# 逐一處理每個檔案
for FILE in "${FILES[@]}"; do
  echo "正在處理檔案: $FILE"
  $BIN_PATH -f "$FILE" -m "$MODEL_PATH" -t "$THREADS"
  if [ $? -ne 0 ]; then
    echo "執行 $FILE 評估時發生錯誤！"
    exit 1
  fi
done

echo "所有檔案處理完成。"
