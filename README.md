# llama.cpp

![Manatee](https://github.com/ZJkyle/Cache/blob/main/pexels-koji-kamei-4766707-scaled-e1687878979926-1280x720.jpg)

## To-Do
- [ ] Add Simple-math Evals
- [ ] Try [Prompt Cache](#persistent-interaction)
- [ ] Eval [Perplexity](#perplexity-measuring-model-quality)
- [ ] [Constraint output](#constrained-output-with-grammars)

## Recent changes

- [2024 Jun 26] [PR #8006](https://github.com/ggerganov/llama.cpp/pull/8006) (Change to my PR)

## Models

- [LLaMA 2 🦙🦙](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [LLaMA 3 🦙🦙🦙](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [LLaMA 3.1 🦙🦙🦙](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

## Usage

### Basic usage

- Method 1: Clone this repository and build locally, see [how to build](./docs/build.md)

Run a basic completion using this command:

    ./llama-cli -m ../models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -p "I believe the meaning of life is" -n -2 -t 12 -c 256 --keep -1

    -n: generate infinite context
        -1: until EOS / timing or memory limit
        -2: until context memory limit

    --mlock                  force system to keep model in RAM rather than swapping or compressing
    --no-mmap                do not memory-map model (slower load but may reduce pageouts if not using mlock)  
    --keep                   keep prompt token (default: 0, -1 = all)

    read memory: smem -r -p -k -t

See [this page](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md) for a full list of parameters.

### Web server

Example usage:

    ./llama-server -m ../models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.ggu --host 0.0.0.0 --port 8080  -t 12 -c 256 --keep ?

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions

### Persistent Interaction

The prompt, user inputs, and model generations can be saved and resumed across calls to `./llama-cli` by leveraging `--prompt-cache` and `--prompt-cache-all`. The `./examples/chat-persistent.sh` script demonstrates this with support for long-running, resumable chat sessions. To use this example, you must provide a file to cache the initial chat prompt and a directory to save the chat session, and may optionally provide the same variables as `chat-13B.sh`. The same prompt cache can be reused for new chat sessions. Note that both prompt cache and chat directory are tied to the initial prompt (`PROMPT_TEMPLATE`) and the model file.

    # Start a new chat
    PROMPT_CACHE_FILE=chat.prompt.bin CHAT_SAVE_DIR=./chat/default ./examples/chat-persistent.sh

    # Resume that chat
    PROMPT_CACHE_FILE=chat.prompt.bin CHAT_SAVE_DIR=./chat/default ./examples/chat-persistent.sh

    # Start a different chat with the same prompt/model
    PROMPT_CACHE_FILE=chat.prompt.bin CHAT_SAVE_DIR=./chat/another ./examples/chat-persistent.sh

    # Different prompt cache for different prompt/model
    PROMPT_TEMPLATE=./prompts/chat-with-bob.txt PROMPT_CACHE_FILE=bob.prompt.bin \
        CHAT_SAVE_DIR=./chat/bob ./examples/chat-persistent.sh

### Constrained output with grammars

`llama.cpp` supports grammars to constrain model output. For example, you can force the model to output JSON only:

    ./llama-cli -m ./models/13B/ggml-model-q4_0.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

The `grammars/` folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](./grammars/README.md).

For authoring more complex JSON grammars, you can also check out [Grammar Intrinsic Labs](https://grammar.intrinsiclabs.ai/), a browser app that lets you write TypeScript interfaces which it compiles to GBNF grammars that you can save for local use. Note that the app is built and maintained by members of the community, please file any issues or FRs on [its repo](http://github.com/intrinsiclabsai/gbnfgen) and not this one.

## Tools

### Prepare and Quantize
[Models](#models)

    # install Python dependencies
    python3 -m pip install -r requirements.txt

    # convert the model to ggml FP16 format
    python3 convert_hf_to_gguf.py models/mymodel/

    # quantize the model to 4-bits (using Q4_K_M method)
    ./llama-quantize ./models/mymodel/ggml-model-f16.gguf ./models/mymodel/ggml-model-Q4_K_M.gguf Q4_K_M

    # update the gguf filetype to current version if older version is now unsupported
    ./llama-quantize ./models/mymodel/ggml-model-Q4_K_M.gguf ./models/mymodel/ggml-model-Q4_K_M-v2.gguf COPY

### Perplexity (measuring model quality)

You can use the `perplexity` example to measure perplexity over a given prompt (lower perplexity is better).
For more information, see [Hugging Face Perplexity](https://huggingface.co/docs/transformers/perplexity).

To learn more about how to measure perplexity using llama.cpp, [read this documentation](./examples/perplexity/README.md).

### Evaluation

See [Eval Llama model on Llama.cpp](https://blog.gopenai.com/how-to-evaluate-local-llms-llama-2-on-a-laptop-with-openai-evals-b1921e104edd)

* After installing evals, you may encounter some errors like: "AttributeError: module 'openai' has no attribute 'error'"
* To solve this, simply downgrade openai to version 0.28

* Test result of Llama3.1-8b-instruct

## Other documentations

- [main (cli)](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md)
- [server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)

**Source**
- [Llama.cpp](https://github.com/ggerganov/llama.cpp)
