#!/bin/bash

echo "+memory" | sudo tee /sys/fs/cgroup/cgroup.subtree_control

sudo mkdir /sys/fs/cgroup/my_cgroup

echo $((4 * 1024 * 1024 * 1024)) | sudo tee /sys/fs/cgroup/my_cgroup/memory.max

cat /sys/fs/cgroup/my_cgroup/memory.max

echo $$ | sudo tee /sys/fs/cgroup/my_cgroup/cgroup.procs

cat /sys/fs/cgroup/my_cgroup/cgroup.procs

./build/bin/main -m my_models/llama3-8b-64k-Q8_0.gguf --color -f my_prompts/tmp.txt -n 512
