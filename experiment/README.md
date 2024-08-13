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
./../llama-server -m ../../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --host 0.0.0.0 --port 8080  -t 12 -c 4096 --keep -1

# Basic web UI can be accessed via browser: http://localhost:8080
# Chat completion endpoint: http://localhost:8080/v1/chat/completions
```

## Optimizations
### Before Optimziation
./../llama-bench -m ../../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -r 10 -pg 8,8 -pg 8,16 -pg 8,32 -pg 8,64 -pg 16,8 -pg 16,16 -pg 16,32 -pg 16,64 -n 0 -embd 1 -t 12

#####  改過 bench.cpp

root@DESKTOP-1P405LN:~/Cache# ./llama-bench -m ../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -r 10 -p 4,8,16,32 -n 0 -embd 1 -t 12
| model                          |       size |     params | backend    | threads |       embd |          test |       prompt t/s |          gen t/s |        total t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------: | ------------: | ---------------: | ---------------: | ---------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |           pp4 |             27.18 |              0.00 |     27.25 ± 1.41 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |           pp8 |             32.01 |              0.00 |     32.06 ± 1.35 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |          pp16 |             35.13 |              0.00 |     35.14 ± 0.76 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |          pp32 |             34.57 |              0.00 |     34.67 ± 1.92 |

root@DESKTOP-1P405LN:~/Cache# ./llama-bench -m ../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -r 10 -n 8,32,128,512 -n 0 -embd 1 -t 12
| model                          |       size |     params | backend    | threads |       embd |          test |       prompt t/s |          gen t/s |        total t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------: | ------------: | ---------------: | ---------------: | ---------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |         pp512 |             36.49 |              0.00 |     36.50 ± 0.58 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |           tg8 |              0.00 |              9.09 |      9.09 ± 0.21 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |          tg32 |              0.00 |              8.72 |      8.78 ± 0.75 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |         tg128 |              0.00 |              8.52 |      8.55 ± 0.49 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |         tg512 |              0.00 |              8.39 |      8.40 ± 0.15 |

root@DESKTOP-1P405LN:~/Cache# ./llama-bench -m ../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -r 10 -pg 8,8 -pg 8,16 -pg 8,32 -pg 8,64 -pg 16,8 -pg 16,16 -pg 16,32 -pg 16,64 -n 0 -embd 1 -t 12
| model                          |       size |     params | backend    | threads |       embd |          test |       prompt t/s |          gen t/s |        total t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------: | ------------: | ---------------: | ---------------: | ---------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |         pp512 |             36.62 |              0.00 |     36.62 ± 0.43 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |       pp8+tg8 |             32.08 |              8.53 |     13.56 ± 1.02 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |      pp8+tg16 |             30.56 |              8.63 |     11.45 ± 1.11 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |      pp8+tg32 |             31.50 |              8.52 |     10.05 ± 0.85 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |      pp8+tg64 |             29.63 |              7.14 |      7.87 ± 0.86 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |      pp16+tg8 |             32.85 |              7.31 |     15.34 ± 1.59 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |     pp16+tg16 |             33.04 |              7.64 |     12.50 ± 1.03 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |     pp16+tg32 |             33.60 |              7.91 |     10.68 ± 0.89 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |     pp16+tg64 |             33.74 |              7.92 |      9.37 ± 0.40 |

### KV Cache Discard 

### Sliding Window
* Referenece: https://github.com/mit-han-lab/streaming-llm

