# Benchmark Test Runner

A tool to orchestrate and automate multiple benchmark tests across distributed hosts using a run_list configuration file.

## Purpose

This tool enables you to plan and execute multiple benchmark configurations in a batch process. It reads a run_list file containing test specifications and automatically runs benchmarks sequentially or in parallel, with optional distributed execution across multiple hosts. Perfect for running extensive benchmark suites overnight or across multiple machines.

## Usage

### Basic Usage

Run all tests from a run_list file:

```bash
bash tools/runs/run.sh benchmark tests/run_list/run_list_example.sh
```

### Run Modes

```bash
# Normal benchmark execution
bash tools/runs/run.sh benchmark tests/run_list/run_list_example.sh

# Dry run - validate configuration without executing tests
bash tools/runs/run.sh dry_run tests/run_list/run_list_example.sh

# Generate individual test scripts for manual execution
bash tools/runs/run.sh generate_script tests/run_list/run_list_example.sh

# Empty mode (same as benchmark)
bash tools/runs/run.sh "" tests/run_list/run_list_example.sh
```

### Override Test Plan

Override the test_plan specified in the run_list:

```bash
bash tools/runs/run.sh benchmark tests/run_list/run_list_example.sh sample
```

This will run all entries using the `sample` test plan instead of the one specified in each line.

## How It Works

1. **Parses run_list file** - Extracts backend, model, docker image, test plan, GPU configuration, etc. from each line
2. **Distributes workload** - If multiple hosts are configured, distributes models across hosts to avoid conflicts
3. **Executes tests** - Runs each test by calling `scripts/run_test.sh` with appropriate parameters
4. **Manages checkpoints** - Optionally removes model checkpoints after use to save disk space
5. **Handles async execution** - Supports parallel execution for tests marked as async

## Run List File Format

```
backend docker_image model_path_or_id model_config test_plan benchmark_client gpu_devices [sub_task] [async]
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `backend` | Yes | Server backend: `vllm` or `sglang` |
| `docker_image` | Yes | Full Docker image path with tag |
| `model_path_or_id` | Yes | HuggingFace model path (e.g., `amd/Llama-3.1-70B-Instruct-FP8-KV`) |
| `model_config` | Yes | Path to model configuration YAML file |
| `test_plan` | Yes | Test plan name (e.g., `sample`, `test`) |
| `benchmark_client` | Yes | Benchmark client: `vllm`, `sglang`, or `genai-perf` |
| `gpu_devices` | Yes | Comma-separated GPU device IDs (e.g., `0` or `0,1,2,3`) |
| `sub_task` | No | Sub-task name or `all` (default: `all`) |
| `async` | No | Set to `async` to run in background (default: synchronous) |

### Example Run List

```bash
# run_list format:
# backend docker_image model_path_or_id model_config test_plan benchmark_client gpu_devices sub_task(optional) async(optional)

# Llama 70B benchmarks with different TP sizes
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0,1
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0,1,2,3

# Async execution example (runs in background)
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-8B-Instruct-FP8-KV configs/models/llama-vllm.yaml test vllm 4 all async

# SGLang backend example
sglang docker.io/rocm/sgl-dev:v0.5.6.post1-rocm700-mi35x-20251211 amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-sglang.yaml sample sglang 0,1,2,3
```

## Configuration Options

Edit the script header to configure behavior:

### Distributed Execution

Configure multiple hosts to distribute workload:

```bash
# in tools/runs/run.sh
host_list=("host1" "host2" "host3")
```

When configured, the script automatically distributes models across hosts based on hostname. Each host will only run benchmarks for its assigned models, preventing multiple hosts from downloading the same model simultaneously.

### Checkpoint Management

Enable automatic checkpoint removal after use:

```bash
# in tools/runs/run.sh
remove_checkpoint=true
```

When enabled, model checkpoints are removed after all tests using that model are complete, saving disk space during long benchmark runs.

## Execution Modes

### Synchronous Execution (Default)

Tests run one after another. If a test fails, the script continues to the next test.

```bash
vllm image_url model_path config.yaml test vllm 0
vllm image_url model_path config.yaml test vllm 1
```

### Asynchronous Execution

Tests marked with `async` run in the background, allowing parallel execution:

```bash
vllm image_url model_path config.yaml test vllm 0 all async
vllm image_url model_path config.yaml test vllm 1 all async
```

**Important:** The script waits for all async jobs before starting any synchronous job.

## Run Modes

### `benchmark` (Default)

Executes all tests and generates results:
- Launches Docker containers
- Runs inference servers
- Executes benchmark clients
- Saves results to `logs/` directory

### `dry_run`

Validates configuration without executing:
- Parses run_list file
- Validates all parameters
- Shows what would be executed
- Does not launch servers or run benchmarks
..xx
### `generate_script`

Generates individual test scripts:
- Creates one `.sh` file per test in `scripts/generated/`
- Useful for manual execution or debugging
- Scripts can be run independently

## Directory Structure

```
llm-inference-benchmark-dev/
├── tools/runs/
│   └── run.sh                    # Main orchestration script
├── scripts/
│   ├── run_test.sh              # Individual test executor
│   └── generated/               # Generated test scripts (generate_script mode)
├── tests/run_list/
│   ├── run_list_example.sh      # Example test suite
│   └── code_llama.sh            # Model-specific tests
├── configs/
│   ├── models/                  # Model configuration files
│   └── benchmark_plans/         # Test plan definitions
└── logs/                        # Benchmark results
```

## Examples

### Run Complete Test Suite

```bash
bash tools/runs/run.sh benchmark tests/run_list/run_list_example.sh
```

### Validate Configuration

```bash
bash tools/runs/run.sh dry_run tests/run_list/run_list_example.sh
```

### Override Test Plan

Run all tests with a quick sample plan instead of full tests:

```bash
bash tools/runs/run.sh benchmark tests/run_list/run_list_example.sh sample
```

### Generate Individual Scripts

```bash
bash tools/runs/run.sh generate_script tests/run_list/run_list_example.sh
# Check scripts/generated/ for individual test scripts
```

### Distributed Execution

On each host, run the same command:

```bash
# Edit tools/runs/run.sh first:
# host_list=("host1" "host2" "host3")

# Run on all hosts (each will run different models)
bash tools/runs/run.sh benchmark tests/run_list/run_list_example.sh
```

## Troubleshooting

### "Model config file not found"

The model_config path is invalid. Check that the file exists:

```bash
ls configs/models/llama-vllm.yaml
```

### "Invalid run_mode"

Run mode must be one of: `""`, `benchmark`, `dry_run`, or `generate_script`

### Tests Not Running on Distributed Hosts

Ensure:
1. `host_list` is configured in the script
2. Hostname matches one of the entries in `host_list`
3. Run the script on all configured hosts

Check which models are assigned to current host:

```bash
bash tools/runs/run.sh benchmark tests/run_list/run_list_example.sh
# Output shows: "Selected keys for host <hostname>: ..."
```

### Disk Space Issues

Enable checkpoint removal to automatically clean up after tests:

```bash
# Edit tools/runs/run.sh
remove_checkpoint=true
```

### Async Jobs Not Completing

The script waits for all async jobs at the end. Check logs for individual job failures:

```
Async job PID 12345 for model 'amd/Llama-3.1-70B-Instruct-FP8-KV' failed.
```

## Integration with Other Tools

### Generate Progress Report

After running benchmarks, generate a progress report:

```bash
python tools/reports/generate_progress_report.py \
  --run-list tests/run_list/run_list_example.sh \
  --output logs/test_results_generated.tsv
```

See [README_REPORTER.md](README_REPORTER.md) for details.

### Plot Results

Visualize benchmark results:

```bash
python tools/plotter/plot_results.py logs/test_results_generated.tsv
```

See [README_PLOTTER.md](README_PLOTTER.md) for details.

## Best Practices

1. **Start with dry_run** - Always validate your run_list before executing
2. **Use sample test plans** - Test with quick sample plans before full benchmarks
3. **Enable checkpoint cleanup** - Set `remove_checkpoint=true` for long runs
4. **Use async strategically** - Mark independent tests as async to parallelize
5. **Monitor disk space** - Model checkpoints can consume hundreds of GBs
6. **Distribute workload** - Use multiple hosts for large benchmark suites
7. **Check logs regularly** - Monitor `logs/` directory for failures

## Requirements

- Bash 4.0+
- Docker with GPU support
- Sufficient disk space for model checkpoints
- GPU devices matching run_list specifications

## Related Documentation

- [README_REPOTER.md](README_REPOTER.md) - Generate progress reports
- [README_PLOTTER.md](README_PLOTTER.md) - Visualize results
- Main README.md - Project overview and setup
