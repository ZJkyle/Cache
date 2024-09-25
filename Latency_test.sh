#!/bin/bash

# 設定輸出 csv 的基礎名稱
csv_file_base="/root/Cache/experiment/Latency/Context_Size_and_Tokens_per_Second"
csv_file="${csv_file_base}.csv"

# 檢查是否存在同名檔案，若存在則加上數字後綴
counter=1
while [[ -f "$csv_file" ]]; do
  csv_file="${csv_file_base}_${counter}.csv"
  ((counter++))
done

# 定義輸入提示
prompt="What is Large Language Model?"

# 清除或初始化 csv 檔案，並寫入表頭
echo "Context Size,Eval Time Line" > "$csv_file"

# 輸入次數
read -p "請輸入要執行的次數: " run_count

# 進行多次執行，根據用戶輸入的次數
counter=1
while [[ $counter -le $run_count ]]; do
  # 設定新的 log 檔名稱 (依照 context_size 命名)
  context_size=$((2**counter))
  new_log_file="/root/Cache/experiment/Latency/Latency_test_${context_size}.log"
  
  # 執行 llama-cli 指令，將終端機輸出和錯誤輸出記錄到新的 log 檔案
  ./build/bin/llama-cli -m ../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -p "$prompt" -n -2 -t 12 -c $context_size --keep -1 > "$new_log_file" 2>&1
  
  # 從 log 檔案中提取包含 "eval time" 的整行
  eval_time_line=$(grep "llama_print_timings:        eval time" "$new_log_file")
  
  # 如果找不到數據，設定一個提示
  if [[ -z "$eval_time_line" ]]; then
    eval_time_line="未找到數據"
  fi

  # 將此次執行的結果寫入 csv 檔案，記錄 context size 和整行 eval time
  echo "$context_size,\"$eval_time_line\"" >> "$csv_file"
  
  echo "Log file saved as: $new_log_file"
  
  # 計數器遞增
  ((counter++))
done

echo "執行完成，共執行 $run_count 次。"

# CSV 檔案提示
echo "數據已儲存至 $csv_file，請檢查結果。"
