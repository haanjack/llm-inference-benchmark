# Test Results Report Generator

A tool to generate `test_results_generated.tsv` from a run_list file by checking existing benchmark logs.

## Purpose

This tool helps you track benchmark progress when tests are run separately or distributed across time. It scans a run_list file and verifies which tests have been completed by checking if the corresponding CSV result files exist and contain all expected results.

## Usage

### Basic Usage

Generate a report from your run_list file:

```bash
python tools/reports/generate_progress_report.py --run-list tests/run_list/run_list.sh
```

This creates `logs/test_results_generated.tsv` with the status of all tests.

### Custom Output Path

```bash
python tools/reports/generate_progress_report.py \
  --run-list tests/run_list/run_list.sh \
  --output logs/my_custom_results.tsv
```

### Verbose Output

```bash
python tools/reports/generate_progress_report.py \
  --run-list tests/run_list/run_list.sh \
  --verbose
```

### Generate Run Lists for Incomplete/Not Tested

Generate `incomplete.sh` and `not_tested.sh` files for rerunning failed or pending tests:

```bash
python tools/reports/generate_progress_report.py \
  --run-list tests/run_list/run_list.sh \
  --generate-run-list
```

This creates:
- `tests/run_list/incomplete.sh` - Tests with partial results
- `tests/run_list/not_tested.sh` - Tests that haven't been run yet

These files can be used directly with the multi-run tool:

```bash
# Run only incomplete tests
bash tools/runs/run.sh benchmark tests/run_list/incomplete.sh

# Run only not tested items
bash tools/runs/run.sh benchmark tests/run_list/not_tested.sh
```

## How It Works

1. **Parses run_list file** - Extracts backend, model, image tag, test plan, etc. from each line
2. **Loads test plan YAML** - Reads `configs/benchmark_plans/{test_plan}.yaml` to count expected test cases
3. **Checks for results** - Looks for `logs/{model}/{image_tag}/total_results_{backend}_{client}.csv`
4. **Compares counts** - Verifies CSV has all expected benchmark results
5. **Generates report** - Outputs TSV file with status for each test
6. **Creates run lists** (optional) - Generates `incomplete.sh` and `not_tested.sh` for rerunning tests

## Run List File Format

```
backend docker_image model_path model_config test_plan benchmark_client gpu_devices [sub_task]
```

Example:
```bash
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-70B-Instruct-FP8-KV llama-vllm sample vllm 0
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-70B-Instruct-FP8-KV llama-vllm sample vllm 1,2
```

## Output Format

Generated TSV file contains:

| Column | Description |
|--------|-------------|
| `timestamp` | CSV file modification time (ISO format) |
| `model` | Model path (e.g., amd/Llama-3.1-70B-Instruct-FP8-KV) |
| `image_tag` | Docker image tag extracted from docker_image |
| `model_config` | Model configuration name (from test setup) |
| `tp_size` | Tensor parallelism size (derived from GPU count) |
| `test_plan` | Test plan name (e.g., sample) |
| `sub_task` | Sub-task name if specified |
| `result` | Status: `success`, `incomplete`, or `not_tested` |
| `log_path` | Path to the CSV result file or `N/A` |

## Status Values

- **`success`** - CSV exists and has all expected benchmark results
- **`incomplete`** - CSV exists but is missing some results
- **`failure`** - Test run encountered errors (detected in server logs)
- **`not_tested`** - No CSV file exists for this test setup

## Generated Run Lists

When using `--generate-run-list`, the tool creates executable shell scripts for incomplete and not tested benchmarks:

### incomplete.sh

Contains tests that:
- Have CSV results files
- Are missing some expected test cases
- May have encountered errors during execution

Use this to retry tests that didn't complete fully.

### not_tested.sh

Contains tests that:
- Have no CSV results at all
- Haven't been executed yet

Use this to run remaining tests from your original run_list.

### File Location

Generated files are placed in the same directory as your input run_list:
```
tests/run_list/
├── run_list.sh           # Your original run_list
├── incomplete.sh         # Auto-generated incomplete tests
└── not_tested.sh         # Auto-generated not tested tests
```

### Using Generated Run Lists

Run incomplete tests:
```bash
bash tools/runs/run.sh benchmark tests/run_list/incomplete.sh
```

Run not tested items:
```bash
bash tools/runs/run.sh benchmark tests/run_list/not_tested.sh
```

See [README_RUN.md](README_RUN.md) for more details on the multi-run tool.

## Requirements

- Python 3.8+
- PyYAML

Install dependencies:
```bash
pip install pyyaml
```

## Examples

### Check which tests from run_list.sh are done

```bash
python tools/reports/generate_progress_report.py --run-list tests/run_list/run_list.sh
```

Output:
```
2025-12-30 12:34:56 - INFO - Generated report: logs/test_results_generated.tsv (64 entries)

Summary:
  Success:     48
  Incomplete:  8
  Not tested:  8
  Total:       64
```

### Track progress of subset of tests

Create a custom run_list with just the tests you're running:

```bash
python tools/reports/generate_progress_report.py \
  --run-list tests/run_list/subset.sh \
  --output logs/subset_results.tsv
```

### Generate run lists and rerun incomplete tests

```bash
# Generate report and create incomplete.sh/not_tested.sh
python tools/reports/generate_progress_report.py \
  --run-list tests/run_list/run_list.sh \
  --generate-run-list

# Run only the incomplete tests
bash tools/runs/run.sh benchmark tests/run_list/incomplete.sh

# Check progress again
python tools/reports/generate_progress_report.py \
  --run-list tests/run_list/run_list.sh
```

## Troubleshooting

### "Test plan file not found"

The tool couldn't find `configs/benchmark_plans/{test_plan}.yaml`. Verify the test_plan name in your run_list matches the yaml filename.

### "No test entries processed"

The run_list file may be empty or have no valid lines. Check that:
- File is not empty
- Each line has at least 7 space-separated columns
- Comments start with `#`

### Timestamps are empty

The CSV files don't exist yet (status is `not_tested`), so no timestamp is available.

### Generated run_list files not created

The `--generate-run-list` flag must be specified for the tool to create `incomplete.sh` and `not_tested.sh`. If all tests show `success`, no files will be generated since there's nothing to rerun.
