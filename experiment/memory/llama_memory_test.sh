#!/bin/bash

# 定義模型與參數
MODELS=(
    "../../../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "../../../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-F16.gguf"
)
C_VALUES=(64 1024 8192)

# 測試指令
COMMAND="./../../llama-cli"

# 結果輸出檔案
OUTPUT_FILE="memory_test_results.txt"

# 初始化結果文件
echo "Model, C, Memory (KB), Time (s)" > $OUTPUT_FILE

# 執行測試
for MODEL in "${MODELS[@]}"; do
    for C in "${C_VALUES[@]}"; do
        /usr/bin/time -v $COMMAND -m $MODEL -p "I believe the meaning of life is" -n -2 -t 12 -c $C --keep -1 > time_output.log 2>&1
        MEMORY=$(awk '/Maximum resident set size/ {print $6}' time_output.log)
        TIME=$(awk '/Elapsed (wall clock) time/ {
            if ($8 ~ /:/) {
                split($8,time,":")
                print (time[1]*60) + time[2]
            } else {
                print $8
            }
        }' time_output.log)
        echo "$MODEL, $C, $MEMORY, $TIME" >> $OUTPUT_FILE
    done
done

echo "測試完成，結果已保存至 $OUTPUT_FILE"
