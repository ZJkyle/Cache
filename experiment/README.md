# llama.cpp inference Optimization

## To-Do
- [x] Memory Usage test

## Recent changes

## Models
- [LLaMA 3.1 🦙🦙🦙](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

## How to use?

- llama_cli
```
./../llama-cli -m ../../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -p "I believe the meaning of life is" -n -2 -t 12 -c 4096 --keep -1

-n: Generate infinite context
    -1: until EOS / timing or memory limit
    -2: until context memory limit

-t: Threads to use

-c: Context saved for KV cache memory

-keep: Save prompt cache
```

- llama_server
```
./../llama-server -m ../../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --host 0.0.0.0 --port 8080  -t 12 -c 4096 --keep -1

# Basic web UI can be accessed via browser: http://localhost:8080
# Chat completion endpoint: http://localhost:8080/v1/chat/completions
```

## Latency

`./llama-bench -m ../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -r 10 -p 4,8,16,32 -n 0 -embd 1 -t 12`

| model                          |       size |     params | backend    | threads |       embd |          test |       prompt t/s |          gen t/s |        total t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------: | ------------: | ---------------: | ---------------: | ---------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |           pp4 |             27.18 |              0.00 |     27.25 ± 1.41 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |           pp8 |             32.01 |              0.00 |     32.06 ± 1.35 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |          pp16 |             35.13 |              0.00 |     35.14 ± 0.76 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |          pp32 |             34.57 |              0.00 |     34.67 ± 1.92 |

`./llama-bench -m ../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -r 10 -n 8,32,128,512 -n 0 -embd 1 -t 12`


| model                          |       size |     params | backend    | threads |       embd |          test |       prompt t/s |          gen t/s |        total t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------: | ------------: | ---------------: | ---------------: | ---------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |         pp512 |             36.49 |              0.00 |     36.50 ± 0.58 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |           tg8 |              0.00 |              9.09 |      9.09 ± 0.21 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |          tg32 |              0.00 |              8.72 |      8.78 ± 0.75 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |         tg128 |              0.00 |              8.52 |      8.55 ± 0.49 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      12 |          1 |         tg512 |              0.00 |              8.39 |      8.40 ± 0.15 |

`./llama-bench -m ../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -r 10 -pg 8,8 -pg 8,16 -pg 8,32 -pg 8,64 -pg 16,8 -pg 16,16 -pg 16,32 -pg 16,64 -n 0 -embd 1 -t 12`

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

## Memory 
* Use `./scripts/llama_memory_test.sh`o
| 模型 | 生成 token 數量 | 記憶體使用 (KB) | 記憶體使用 (GB) |
|------|----------------|----------------|----------------|
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 64   | 4,890,596 KB  | 4.66 GB |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 1024 | 5,008,788 KB  | 4.78 GB |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 8192 | 5,934,244 KB  | 5.66 GB |
| Meta-Llama-3.1-8B-Instruct-F16    | 64   | 15,353,536 KB | 14.64 GB |
| Meta-Llama-3.1-8B-Instruct-F16    | 1024 | 15,474,996 KB | 14.75 GB |
| Meta-Llama-3.1-8B-Instruct-F16    | 8192 | 15,604,076 KB | 14.88 GB |

## Accuracy(Perplexity)

* Wikitest Result
`./build/bin/llama-perplexity -f ../datasets/pg19/test -m ../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -t 12`

| Quantization | Model size [GiB] | PPL                   |
| ------------ | ---------------- | --------------------- |
| f16          | 14.97            | 7.3170 +/- 0.04676    |
| q8_0         | 4.58             | 6.234284 ±   0.037878 |

* Long context perplexity (pg19)
`./build/bin/llama-long-context-perplexity -f wikitext-2-raw/wiki.test.raw -m ../models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-F16.gguf `

| 檔案名稱           | words數量 | perplexity | 測試時間   |
|------------------|----------|------------|-----------|
| 10146.txt        | 253872   | 7.61423    | 1713.7    |
| 10321.txt        | 324979   | 20.1684    | 2814.04   |
| 10356.txt        | 314879   | 12.0528    | 2573.3    |
| 10762.txt        | 348606   | 18.3553    | 2564.46   |
| 12204.txt        | 24408    | 0          | 0.00363162|
| 15562.txt        | 222300   | 10.4606    | 1709.75   |
| 22424.txt        | 444287   | 9.12016    | 3699.8    |
| 24553.txt        | 130627   | 12.6582    | 845.894   |
| 2544.txt         | 352384   | 11.7049    | 2825.49   |
| 25646.txt        | 103323   | 5.42114    | 851.971   |
| 25773.txt        | 488162   | 12.1995    | 4282.44   |
| 25830.txt        | 123962   | 10.854     | 866.44    |
| 26183.txt        | 475456   | 10.148     | 4000.73   |
| 26239.txt        | 354117   | 11.0234    | 2850.87   |
| 26493.txt        | 562729   | 8.81121    | 4517      |
| 26618.txt        | 58957    | 0          | 0.00811233|
| 27454.txt        | 404307   | 10.4352    | 3439.24   |
| 28444.txt        | 335975   | 11.042     | 2628.87   |
| 28988.txt        | 737008   | 9.70537    | 6077.3    |
| 29594.txt        | 86075    | 11.2515    | 584.695   |
| 40579.txt        | 355979   | 9.53644    | 1496.93   |
| 40700.txt        | 148045   | 8.27309    | 746.571   |
| 4128.txt         | 95877    | 9.47184    | 448.154   |
| 41603.txt        | 274624   | 8.24184    | 1198.75   |
| 41607.txt        | 337917   | 8.21867    | 1348.74   |
| 42655.txt        | 882218   | 8.72303    | 3581.12   |
| 43536.txt        | 55719    | 7.30343    | 302.539   |
| 43845.txt        | 57163    | 0          | 0.0113388 |
| 44099.txt        | 147224   | 13.1219    | 594.382   |
| 44557.txt        | 244883   | 12.9964    | 1034.22   |
| 45313.txt        | 4574066  | 6.79685    | 20171.9   |
| 45881.txt        | 217753   | 8.54858    | 898.55    |
| 45888.txt        | 282211   | 13.8841    | 1191.43   |
| 46915.txt        | 429060   | 7.40359    | 1788.77   |
| 47068.txt        | 351283   | 9.3431     | 1485.99   |
| 47558.txt        | 147889   | 11.4966    | 603.889   |
| 47581.txt        | 74970    | 11.3384    | 298.893   |
| 47676.txt        | 311749   | 11.9874    | 1489.09   |
| 48693.txt        | 410440   | 8.76471    | 1652.69   |
| 49078.txt        | 147677   | 9.57285    | 594.57    |
| 49529.txt        | 412383   | 7.41648    | 1789.42   |
| 49596.txt        | 97986    | 7.4804     | 451.383   |
| 50287.txt        | 302777   | 8.33211    | 1199.06   |
| 53345.txt        | 347642   | 6.68285    | 1494.42   |
| 5396.txt         | 956818   | 12.2433    | 4157.72   |
| 54537.txt        | 200630   | 7.98388    | 747.285   |
| 54624.txt        | 70363    | 0          | 0.011491  |
| 55339.txt        | 663065   | 11.113     | 2672.61   |
| 55871.txt        | 1448175  | 8.16085    | 5977.93   |
| 56410.txt        | 111994   | 9.03875    | 442.601   |
| 5734.txt         | 581320   | 8.36425    | 2377.3    |
| 5770.txt         | 246697   | 8.35533    | 1042.48   |
| 57791.txt        | 101957   | 8.81795    | 303.035   |
| 58473.txt        | 150101   | 12.1331    | 745.365   |
| 58553.txt        | 211173   | 11.4778    | 740.817   |
| 58598.txt        | 115036   | 11.5354    | 447.117   |
| 5956.txt         | 260972   | 9.59069    | 1046      |
| 5962.txt         | 459976   | 8.37846    | 2082.36   |
| 6412.txt         | 871566   | 12.1126    | 3736.8    |
| 6941.txt         | 1064483  | 11.9483    | 4643.21   |
| 7412.txt         | 955823   | 11.9497    | 3853.25   |
| 7987.txt         | 1048869  | 13.626     | 4455.51   |
| 8197.txt         | 137711   | 17.3796    | 452.986   |
| 8559.txt         | 236569   | 7.3171     | 895.118   |
| 860.txt          | 280293   | 8.4875     | 1189.45   |
| 8788.txt         | 19415    | 0          | 0.00427489|
| 9315.txt         | 22463    | 0          | 0.00504691|
| 9931.txt         | 521935   | 11.6204    | 2245.07   |

### Sliding Window
* Referenece: https://github.com/mit-han-lab/streaming-llm

