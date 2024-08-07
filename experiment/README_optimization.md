# llama.cpp inference Optimization

## To-Do
- [ ] How is Llama-server working ?
- [ ] How to improve inference speed

## Recent changes

- [2024 Jun 26] [PR #8006](https://github.com/ggerganov/llama.cpp/pull/8006) (Change to my PR)

## Models
- [LLaMA 3.1 🦙🦙🦙](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

## Usage

- Use `script_llama_cli.sh`
```
./../llama-cli -m ../../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -p "I believe the meaning of life is" -n -2 -t 12 -c 4096 --keep -1

-n: Generate infinite context
    -1: until EOS / timing or memory limit
    -2: until context memory limit

-t: Threads to use

-c: Context saved for KV cache memory

-keep: Save prompt cache
```

- Use `script_llama_server.sh`
```
./../llama-server -m ../../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.ggu --host 0.0.0.0 --port 8080  -t 12 -c 4096 --keep -1

# Basic web UI can be accessed via browser: http://localhost:8080
# Chat completion endpoint: http://localhost:8080/v1/chat/completions
```

## Optimizations
### Before Optimziation
./../llama-bench -m ../../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -r 10 -pg 8,8 -pg 8,16 -pg 8,32 -pg 8,64 -pg 16,8 -pg 16,16 -pg 16,32 -pg 16,64 -n 0 -embd 1 -t 12


```
./../llama-bench -m ../../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -r 10 -p 8,32,128,512 -n 0 -embd 1 -t 12

| model                          |       size |     params | backend    | threads |       embd |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------: | ------------: | ---------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |       6 |          1 |           pp8 |     25.46 ± 0.69 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |       6 |          1 |          pp32 |     28.65 ± 0.48 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |       6 |          1 |         pp128 |     28.60 ± 0.79 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |       6 |          1 |         pp512 |     28.25 ± 0.64 |
```

./../llama-bench -m ../../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -r 10 -n 8,32,128,512 -n 0 -embd 1 -t 12
| model                          |       size |     params | backend    | threads |       embd |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------: | ------------: | ---------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |       6 |          1 |         pp512 |     28.50 ± 0.37 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |       6 |          1 |           tg8 |      9.75 ± 0.25 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |       6 |          1 |          tg32 |     10.11 ± 0.10 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |       6 |          1 |         tg128 |      9.81 ± 0.22 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |       6 |          1 |         tg512 |      9.62 ± 0.31 |

root@DESKTOP-1P405LN:~/Cache# ./llama-bench -m ../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -r 10 -n 8,32,128,512 -n 0 -embd 1 -t 12
| model                          |       size |     params | backend    | threads |       embd |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------: | ------------: | ---------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |         pp512 |     35.51 ± 0.87 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |           tg8 |      8.43 ± 0.81 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |          tg32 |      8.44 ± 0.69 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |         tg128 |      7.20 ± 0.41 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |         tg512 |      7.27 ± 0.59 |

root@DESKTOP-1P405LN:~/Cache# ./llama-bench -m ../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -r 10 -pg 8,8 -pg 8,16 -pg 8,32 -pg 8,64 -pg 16,8 -pg 16,16 -pg 16,32 -pg 16,64 -n 0 -embd 1 -t 12
| model                          |       size |     params | backend    | threads |       embd |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------: | ------------: | ---------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |         pp512 |     35.47 ± 0.41 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |       pp8+tg8 |     12.01 ± 0.73 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |      pp8+tg16 |     10.04 ± 0.70 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |      pp8+tg32 |      8.97 ± 0.44 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |      pp8+tg64 |      8.30 ± 0.46 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |      pp16+tg8 |     16.05 ± 0.47 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |     pp16+tg16 |     12.41 ± 0.88 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |     pp16+tg32 |     10.44 ± 0.52 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |     pp16+tg64 |      8.98 ± 0.50 |
### KV Cache Discard 

### Sliding Window
* Referenece: https://github.com/mit-han-lab/streaming-llm

