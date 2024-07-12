# Experiment

<img src="https://i.imgur.com/r8dBZDR.png" width=70% />

## Description

Add KV Cache optimizing module to llama.cpp project.
Part of llama.cpp (e.g. regard to KV Cache management function) would be injected with new module.

## Quick start

```bash
./build/bin/main -m my_models/q4/llama3-8B-16k-Q4-1.gguf --repeat-penalty 1.15 --repeat-last-n 128 --temp 0 -c 4096 -f my_prompts/original_context/story_prompt.txt --color -n 2048 --ignore-eos
```

## Measure Perplexity

```bash
./build/bin/perplexity -m my_models/llama3-8b-64k-Q8_0.gguf -f wikitext-2-raw/wiki.test.raw -ctk q4_roy
```

## Build

- Rebuild:

  ```bash
  # for debug
  rm -rf build
  cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so -DLLAMA_BLAS=ON
  cmake --build build
  # normal
  rm -rf build
  cmake -B build -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so -DLLAMA_BLAS=ON
  cmake --build build --config Release
  ```

### Metal Build

On MacOS, Metal is enabled by default. Using Metal makes the computation run on the GPU.
To disable the Metal build at compile time use the `LLAMA_NO_METAL=1` flag or the `LLAMA_METAL=OFF` cmake option.

When built with Metal support, you can explicitly disable GPU inference with the `--n-gpu-layers|-ngl 0` command-line
argument.

## Models

- Llama3-8b-16K
  - [GGUF](https://huggingface.co/PrunaAI/Llama-3-8B-16K-GGUF-smashed)

## Paremeters Of Llama.cpp

- `-m`: directory of models
- `--color`: colorise output to distinguish prompt and user input from generations
- `-t`: number of threads to use during generation (default: 16)
- `-p`: prompt to start generation with (default: empty)
- `--random-prompt`: start with a randomized prompt
- `-f FNAME`: prompt file to start generation
- `-n`: number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
  - Set the number of tokens to predict when generating text. Adjusting this value can influence the length of the generated text
- `-c`: size of the prompt context (default: 512, 0 = loaded from model)
  - Set the size of the prompt context. The default is 512, but LLaMA models were built with a context of 2048, which will provide better results for longer input/inference
- `--ignore-eos`: ignore end of stream token and continue generating (implies --logit-bias 2-inf)
- `--mlock`: Lock the model in memory, preventing it from being swapped out when memory-mapped. This can improve performance but trades away some of the advantages of memory-mapping by requiring more RAM to run and potentially slowing down load times as the model loads into RAM.
- `--no-mmap`: Do not memory-map the model. By default, models are mapped into memory, which allows the system to load only the necessary parts of the model as needed. However, if the model is larger than your total amount of RAM or if your system is low on available memory, using mmap might increase the risk of pageouts, negatively impacting performance. Disabling mmap results in slower load times but may reduce pageouts if you're not using --mlock. Note that if the model is larger than the total amount of RAM, turning off mmap would prevent the model from loading at all.
- `-fa`: enable flash attention
- `--pre-rope-cache`: make key cached before rope
- `--cache-mmap`: use self-defined memory map for kv cache

### Extended Context Size

Some fine-tuned models have extended the context length by scaling RoPE. For example, if the original pre-trained model has a context length (max sequence length) of 4096 (4k) and the fine-tuned model has 32k. That is a scaling factor of 8, and should work by setting the above --ctx-size to 32768 (32k) and --rope-scale to 8.

- `--rope-scale N`: Where N is the linear scaling factor used by the fine-tuned model

## Memory/Disk Requirements

As the models are currently fully loaded into memory, you will need adequate disk space to save them and sufficient RAM to load them. At the moment, memory and disk requirements are the same.

| Model | Original size | Quantized size (Q4_0) |
| ----: | ------------: | --------------------: |
|    7B |         13 GB |                3.9 GB |
|   13B |         24 GB |                7.8 GB |
|   30B |         60 GB |               19.5 GB |
|   65B |        120 GB |               38.5 GB |

### Quantization

Several quantization methods are supported. They differ in the resulting model disk size and inference speed.

_(outdated)_

| Model | Measure      |    F16 |   Q4_0 |   Q4_1 |   Q5_0 |   Q5_1 |   Q8_0 |
| ----: | ------------ | -----: | -----: | -----: | -----: | -----: | -----: |
|    7B | perplexity   | 5.9066 | 6.1565 | 6.0912 | 5.9862 | 5.9481 | 5.9070 |
|    7B | file size    |  13.0G |   3.5G |   3.9G |   4.3G |   4.7G |   6.7G |
|    7B | ms/tok @ 4th |    127 |     55 |     54 |     76 |     83 |     72 |
|    7B | ms/tok @ 8th |    122 |     43 |     45 |     52 |     56 |     67 |
|    7B | bits/weight  |   16.0 |    4.5 |    5.0 |    5.5 |    6.0 |    8.5 |
|   13B | perplexity   | 5.2543 | 5.3860 | 5.3608 | 5.2856 | 5.2706 | 5.2548 |
|   13B | file size    |  25.0G |   6.8G |   7.6G |   8.3G |   9.1G |    13G |
|   13B | ms/tok @ 4th |      - |    103 |    105 |    148 |    160 |    131 |
|   13B | ms/tok @ 8th |      - |     73 |     82 |     98 |    105 |    128 |
|   13B | bits/weight  |   16.0 |    4.5 |    5.0 |    5.5 |    6.0 |    8.5 |

## Perplexity (measuring model quality)

You can use the `perplexity` example to measure perplexity over a given prompt (lower perplexity is better).
For more information, see [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity).

The perplexity measurements in table above are done against the `wikitext2` test dataset (<https://paperswithcode.com/dataset/wikitext-2>), with context length of 512.
The time per token is measured on a MacBook M1 Pro 32GB RAM using 4 and 8 threads.

### How to run

1. Download/extract: <https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip>
2. Run `./perplexity -m models/7B/ggml-model-q4_0.gguf -f wiki.test.raw`
3. Output:

```
perplexity : calculating perplexity over 655 chunks
24.43 seconds per pass - ETA 4.45 hours
[1]4.5970,[2]5.1807,[3]6.0382,...
```

And after 4.45 hours, you will have the final perplexity.
