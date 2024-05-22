# Experiment Environment

![Kimdodo](https://i.imgur.com/r8dBZDR.png "Kimdodo")

## Description

Add KV Cache optimizing module to llama.cpp project.
Part of llama.cpp (e.g. regard to KV Cache management function) would be injected with new module.

## Build

In order to build llama.cpp you have four different options.

-   Using `make`:

    -   On Linux or MacOS:

        ```bash
        make
        ```

        **Note**: for `Debug` builds, run `make LLAMA_DEBUG=1`

-   Using `CMake`:

    ```bash
    cmake -B build
    cmake --build build --config Release
    ```

### Metal Build

On MacOS, Metal is enabled by default. Using Metal makes the computation run on the GPU.
To disable the Metal build at compile time use the `LLAMA_NO_METAL=1` flag or the `LLAMA_METAL=OFF` cmake option.

When built with Metal support, you can explicitly disable GPU inference with the `--n-gpu-layers|-ngl 0` command-line
argument.

### BLAS Build

Building the program with BLAS support may lead to some performance improvements in prompt processing using batch sizes higher than 32 (the default is 512). Support with CPU-only BLAS implementations doesn't affect the normal generation performance. We may see generation performance improvements with GPU-involved BLAS implementations, e.g. cuBLAS, hipBLAS and CLBlast. There are currently several different BLAS implementations available for build and use:

-   #### Accelerate Framework:

    This is only available on Mac PCs and it's enabled by default. You can just build using the normal instructions.

-   #### OpenBLAS:

    This provides BLAS acceleration using only the CPU. Make sure to have OpenBLAS installed on your machine.

    -   Using `make`:

        -   On Linux:

            ```bash
            make LLAMA_OPENBLAS=1
            ```

    -   Using `CMake` on Linux:

        ```bash
        cmake -B build -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS
        cmake --build build --config Release
        ```

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

The perplexity measurements in table above are done against the `wikitext2` test dataset (https://paperswithcode.com/dataset/wikitext-2), with context length of 512.
The time per token is measured on a MacBook M1 Pro 32GB RAM using 4 and 8 threads.

### How to run

1. Download/extract: https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
2. Run `./perplexity -m models/7B/ggml-model-q4_0.gguf -f wiki.test.raw`
3. Output:

```
perplexity : calculating perplexity over 655 chunks
24.43 seconds per pass - ETA 4.45 hours
[1]4.5970,[2]5.1807,[3]6.0382,...
```

And after 4.45 hours, you will have the final perplexity.
