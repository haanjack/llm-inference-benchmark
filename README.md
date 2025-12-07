# LLM Inference Benchmark Suite

This is benchmark script for extensive inference tests for various setups. This benchmark script help users to test with the following variants.
 - various environment variable sets
 - input/output sequence length
 - num concurrency
 - iteration based num prompts control
 - request rate control
 - Execution support for `docker`, `podman`, and direct in-container runs
 - multiple inference server: vllm, sglang, and remote endpoint
 - multiple benchmark clients: vllm, sglang, and genai-perf

In addition, this benchmark test tries to obey [AMD's vLLM V1 performance optimization](https://rocm.docs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html) guide and validation.

This benchmark script generate vLLM server command, execute, and perform benchmark.

## Execution Modes

The script supports two main execution modes:

1.  **Host Mode (Default):** The script runs on your host machine and launches a `docker` or `podman` container to run the vLLM server and benchmark client. This is the standard way to use the script.

2.  **In-Container Mode (`--in-container`):** When you are already inside a container that has the vLLM environment and GPU access, you can use the `--in-container` flag. This tells the script to bypass `docker`/`podman` and execute the vLLM server and benchmark client as direct subprocesses.

    ```bash
    # Example of running inside a pre-configured container
    python main.py --in-container --model-config ...
    ```

## Basic usage of test

The following command shows how to use this.

```bash
python main.py \
    --model-config configs/models/default-vllm.yaml \
    --model-path-or-id Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --image docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103 \
    --backend vllm \
    --benchmark-client vllm \
    --test-plan test \
    --gpu-devices 0
```

There are several interesting argument: `model-config` and `test-plan`. These are the purpose of this benchmark script. "What is best setting?" and "I just want to roll-out the tests". Check each section for more details.

### Environment File
Each model has optimal configuration and settings. To support this divergency, this benchmark script provide individual model config yaml file.

Following snippet shows the basic format of the model config file.

```yaml
# Default model configuration

env:
  # Example environment variables
  # HUGGING_FACE_HUB_TOKEN: "your_token_here"
  VLLM_ROCM_USE_AITER: 1

  SGLANG_USE_AITER: 1

server_args:
  ## Arguments for the vLLM server
  quantization: fp8
  kv-cache-dtype: auto

  # Add other vLLM server arguments here
  # example:
  # max_model_len: 1024
  # gpu-memory-utilization: 0.95
  # max-num-batched-token: 8192
  # swap-space: 16
  # block-size: 64
  # no-enable-prefix-caching: true
  # async-scheduling: true

  ## Arguments for the SGLang server
  log-level: "info"
  # mem-fraction-static: 0.8
  # disable-raix-cache: true
  chunked-prefill-size: 196608
  max-prefill-tokens: 196608
  num-continuous-decode-steps: 4
  cuda-graph-max-bs: 128

compilation_config:
  cudagraph_mode: "FULL_AND_PIECEWISE"
```

For more available environment variables, please checke the following documents.
- [vLLM environment variables](https://docs.vllm.ai/en/stable/configuration/env_vars.html)
- [AMD vLLM V1 performance optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)
- [AITER switches](https://rocm.docs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html#aiter-ai-tensor-engine-for-rocm-switches)

vLLM server arguments can be changed following the purpose of test. If you want optimize for TTFT, you need to test smaller `max-num-batched-token < 8192`. Or it is recommended to test 32-64k for low ITL or 64k+ for max throughput. But this can be vary depending on the model size and context length. Fill free to change and test your own.

I tries to put most of vLLM controls with this file while keeping the script argument is simple. Because this file can work as arguments set. The benchmark result will record this file and you can seperate the result among the all the mixed test results accordingly.

### Test Plan File
This benchmark script follow test plan in `configs/benchmark_plans/`. This plan file is custom format file that user can put comments with `#` prefix.

The benchmark script loads this file via `--test-plan` by identifying file name. Then, it parses along this order: request_rate, client_count, num_iteration, input_length, and output_length.

By specifying the benchmark plan, you can obtain multiple test results easily.

Following snippet show an example of the plan file.
```yaml
test_scenarios:
  - name: "1k input-1k output"
    description: "Balanced test with 1k input and 1k output tokens"
    request_rate: 0  # inf
    concurrency: [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 192, 256]
    input_length: 1024
    output_length: 1024
    num_iteration: 8
```

There are several test cases in `configs/benchmark_plans`.
 - [test](./configs/benchmark_plans/test.yaml)
 - [sample](./configs/benchmark_plans/sample.yaml)
 - [decode_heavy](./configs/benchmark_plans/decode_heavy.yaml)
 - [prefill_heavy](./configs/benchmark_plans/prefill_heavy.yaml)
 - [hybrid](./configs/benchmark_plans/hybrid.yaml)
 - [throughput_control](./configs/benchmark_plans/throughput_control.yaml)

### Model load or download
There are multiple methods to specify benchmark model path for `--model-path`.

1. Absolute path

  - Model loads from specified model path. If the model does not exist in that path, this script tries to download model from huggingface hub.

2. HuggingFace Model ID

  - Users can specify huggingface hub's model id, then this benchmark script will find model based on `model_root_dir`. `model_root_dir` directs `${HOME}/models` by default, but you can change anywhere you want with `--model-root-dir` argument. If this benchmark could not find the model, then this script will try to download model from the huggingface hub. You might want to update `HF_TOKEN` environment variable in `configs/envs/common` file to get access to the huggingface hub.


## Benchmark Result

For all the benchmark results are parsed from each benchmark log and consolidated in `logs/<model name>/<docker-tag>/results_<server>_<client>.csv` file.

This is an exmaple of the `results_<server>_<client>.csv` file.
```csv
env,TP Size,Request Rate,Num. Iter,Client Count,MaxNumSeqs,Input Length,Output Length,Test Time,Mean TTFT (ms),Median TTFT (ms),P99 TTFT (ms),Mean TPOT (ms),Median TPOT (ms),P99 TPOT (ms),Mean ITL (ms),Median ITL (ms),P99 ITL (ms),Mean E2EL (ms),Median E2EL (ms),P99 E2EL (ms),Request Throughput (req/s),Output token throughput (tok/s),Total Token throughput (tok/s)
default,1,0,4,256,4,512,128,2.32,31.75,35.30,36.81,4.30,4.28,4.39,4.30,4.27,4.75,577.79,578.79,580.43,6.91,884.27,4414.46
default,1,0,4,256,8,512,128,2.48,35.05,33.35,51.13,4.59,4.57,4.74,4.59,4.50,5.70,617.41,619.06,631.81,12.91,1652.74,8249.56
```

### Result Logs
All the benchmark logs are stored in `logs/<model name>/<docker-tag>` directory following the model name and docker images tags. And single benchmark logs are stored in `<model-config>-t<tp size>/<run-configs>`.

## Example
Following command is an example of benchmarks. (server: `vllm` and client: `vllm`)

### LLaMA3.3 70B

```bash
python3 main.py \
  --model-config configs/models/llama-vllm.yaml \
  --model-path ~/models/amd/Llama-3.3-70B-Instruct-FP8-KV \
  --image docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103 \
  --backend vllm \
  --test-plan test \
  --benchmark-client vllm \
  --gpu-devices=0
```

### GPT-OSS-120B

```bash
python3 main.py \
  --model-config configs/models/gpt-oss-vllm.yaml \
  --model-path ~/models/openai/gpt-oss-120b \
  --image docker.io/rocm/vllm:rocm7.0.0_vllm_0.10.2_20251006 \
  --backend vllm \
  --test-plan test \
  --benchmark-client vllm \
  --gpu-devices=0
```

### DeepSeek R1

```bash
python3 main.py \
  --model-config configs/models/deepseek-vllm.yaml \
  --model-path ~/models/deepseek-ai/DeepSeek-R1-0528 \
  --image docker.io/rocm/vllm:rocm7.0.0_vllm_0.10.2_20251006 \
  --backend vllm \
  --test-plan test \
  --benchmark-client vllm \
  --gpu-devices=0,1,2,3,4,5,6,7
```

### Qwen3 30B A3B FP8

```bash
python main.py \
  --model-config configs/models/default-vllm.yaml \
  --model-path ~/models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --image docker.io/rocm/vllm:rocm7.0.0_vllm_0.10.2_20251006 \
  --backend vllm \
  --test-plan test \
  --benchmark-client vllm \
  --gpu-devices 0
```


## Other Features

### Generate Standalone Scripts

You can generate self-contained benchmark scripts that can be shared with others:

```bash
# Generate a script for your benchmark configuration
python main.py \
    --model-config configs/models/default-vllm.yaml \
    --model-path-or-id Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --image docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103 \
    --backend vllm \
    --benchmark-client vllm \
    --test-plan test \
    --gpu-devices 0 \
    --generate-script

# Generate specific scenarios only
python main.py ... --test-plan sample --sub-tasks 1k1k 8k1k --generate-script

# Make the script more shareable by prettifying it
python scripts/generated_script_prettifier.py scripts/generated/run-default-vllm-test-<sub-task>.sh

```

The prettified script:
- Replaces hardcoded paths with `$HOME` variable for portability
- Extracts test parameters into clear variables at the top
- Removes internal cache mounts
- Adds helpful documentation comments
- Makes it easy to customize and share

### Manual test
To ease various testing, this script provides `--dry-run` mode. With this option, benchmark script prints out the command which will be used in the benchmark. You can copy the output and start own test.

Please mind that having full support vLLM command is not this script's objective.


### Profile (TBU)
For ease of analysis performance, this project provides `--profile` argument. Then it exports vllm server profile traces into `./profile` directory.

This repo also has a copy of vllm profile tool - `layerwise profiling`.

```
python examples/offline_inference/profiling.py \\
    --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 --batch-size 4 \\
    --prompt-len 512 --max-num-batched-tokens 8196 --json Llama31-8b-FP8 \\
    --enforce-eager run_num_steps -n 2
```

then you can use various tools to analyze the json output terminal ascii tables:
```
python tools/profiler/print_layerwise_table.py \\
    --json-trace Llama31-8b-FP8.json --phase prefill --table summary
```
or create matplotlib stacked bar charts:
```
python tools/profiler/visualize_layerwise_profile.py \\
    --json-trace Llama31-8b-FP8.json \\
    --output-directory profile_breakdown --plot-metric pct_cuda_time
```

# TODO
1. Having test inferenceMax options:
    - https://github.com/InferenceMAX/InferenceMAX/blob/main/benchmarks/70b_fp4_mi355x_docker.sh
    - https://github.com/InferenceMAX/InferenceMAX/blob/main/benchmarks/70b_fp8_mi355x_docker.sh
1. Benchmark with other parallelism
1. Writing graph drawing code
1. Benchmark with PD disaggregation


# TODO:2
1. supprot sglang / gen-ai perf
1. warmup by this suite
1. total results with jsonl