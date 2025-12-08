# Evaluation Module Setup Guide

## Installation

### Automatic Installation (Recommended)

The evaluator will **automatically install** `lm-eval[api]` if it's not found. This allows evaluation to work with any Docker image without rebuilding containers.

When you run evaluation:
- If `lm_eval` is not in PATH, it will be installed automatically
- Installation takes ~1-2 minutes on first run
- Subsequent runs use the installed package

### Manual Installation (Optional)

If you prefer to pre-install:

```bash
# Option 1: Install from requirements file
pip install -r requirements-evaluation.txt

# Option 2: Install directly
pip install lm-eval[api]>=0.4.0

# Verify installation
lm_eval --help
which lm_eval
```

## Key Changes Applied

### 1. **Automatic Installation** (`lm_eval.py`)
- Added `_check_lm_eval_installed()` method with auto-install capability
- Automatically installs `lm-eval[api]>=0.4.0` if not found in PATH
- 5-minute timeout for installation process
- Works with any Docker image without rebuilding containers
- Logs installation progress and verifies successful installation

### 2. **Improved Error Handling** (`lm_eval.py`)
- Enhanced `_classify_error()` for better error categorization
- Added stdout capture in error results
- Changed from `logger.error()` to `logger.exception()` for full tracebacks

### 3. **Better Result Parsing** (`lm_eval.py`)
- Check multiple possible result file locations
- Handle both task-specific and combined result formats
- Include results_file path in output
- Better error logging with checked paths

### 4. **Command Line Fixes** (`lm_eval.py`)
- Fixed: `--model-args` → `--model_args` (underscore)
- Fixed: `--num-fewshot` → `--num_fewshot` (underscore)
- Fixed: `--batch-size` → `--batch_size` (underscore)
- Fixed: `--output-path` → `--output_path` (underscore)
- Fixed: `--log-samples` → `--log_samples` (underscore)
- Changed: `--cache-dir` → `--cache_requests true`

### 5. **Enhanced Error Messages** (`evaluation_client.py`)
- Added separate `RuntimeError` catch for installation errors
- Better error logging with separators
- Include error_type in results

## Usage

### Run with evaluation
```bash
python main.py \
  --model-config llama2_7b \
  --model-path-or-id meta-llama/Llama-2-7b \
  --backend vllm \
  --image vllm/vllm:latest \
  --run-evaluation \
  --evaluation-plan default
```

### With custom cache directory
```bash
python main.py \
  --model-config llama2_7b \
  --model-path-or-id meta-llama/Llama-2-7b \
  --backend vllm \
  --image vllm/vllm:latest \
  --run-evaluation \
  --evaluation-cache-dir /data/hf_cache
```

## Behavior

### First Run
On first evaluation run, you'll see:
```
WARNING: lm-eval-harness not found in PATH, installing...
INFO: lm-eval-harness installed successfully
INFO: lm-eval-harness found: /usr/local/bin/lm_eval
```

### Subsequent Runs
```
INFO: lm-eval-harness found: /usr/local/bin/lm_eval
```

### Installation Failure
If automatic installation fails:
```
ERROR: Failed to install lm-eval-harness:
stdout: ...
stderr: ...
```

## Files Modified

1. `llm_benchmark/evaluators/lm_eval.py` - Core evaluator with installation check
2. `llm_benchmark/clients/evaluation_client.py` - Client with better error handling
3. `requirements-evaluation.txt` - New file with evaluation dependencies

## Testing

```bash
# Test imports
python3 -c "from llm_benchmark.evaluators import LMEvalEvaluator; print('OK')"

# Test with dry-run
python main.py --model-config test --model-path-or-id test --image test --dry-run --run-evaluation
```
