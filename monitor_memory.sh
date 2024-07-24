#!/bin/bash

# 要监控的进程的 PID
TARGET_PID=38983  # 替换为你想监控的进程的 PID
LOG_FILE="memory_usage.log"
INTERVAL=5  # 监控间隔时间（秒）

# 初始化最大内存值
max_rss=0
max_uss=0
max_pss=0

# 打印表头
echo "Time, PID, Command, Swap, USS, PSS, RSS" > $LOG_FILE

while true; do
    # 获取当前时间
    current_time=$(date +'%Y-%m-%d %H:%M:%S')

    # 运行 smem 并筛选出目标进程的信息
    smem_output=$(smem -r -p -k -t | grep -w $TARGET_PID)

    # 提取当前内存使用值
    current_rss=$(echo $smem_output | awk '{print $8}')
    current_uss=$(echo $smem_output | awk '{print $6}')
    current_pss=$(echo $smem_output | awk '{print $7}')

    # 记录到日志文件
    echo "$current_time, $smem_output" >> $LOG_FILE

    # 更新最大内存使用值
    if (( $(echo "$current_rss > $max_rss" | bc -l) )); then
        max_rss=$current_rss
    fi

    if (( $(echo "$current_uss > $max_uss" | bc -l) )); then
        max_uss=$current_uss
    fi

    if (( $(echo "$current_pss > $max_pss" | bc -l) )); then
        max_pss=$current_pss
    fi

    # 打印当前最大内存使用情况
    echo "Max RSS: $max_rss KB"
    echo "Max USS: $max_uss KB"
    echo "Max PSS: $max_pss KB"

    # 等待下一个监控间隔
    sleep $INTERVAL
done
