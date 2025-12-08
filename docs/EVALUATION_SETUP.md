# Evaluation Module Setup Guide

## Installation

### Install lm-eval-harness

```bash
# Option 1: Install from requirements file
pip install -r requirements-evaluation.txt

# Option 2: Install directly
pip install lm-eval>=0.4.0

# Verify installation
lm_eval --help
which lm_eval
```

## Key Changes Applied

### 1. **Installation Check** (`lm_eval.py`)
- Added `_check_lm_eval_installed()` method
- Raises `RuntimeError` with installation instructions if not found
- Logs lm-eval path when found

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

## Error Handling

If lm-eval is not installed, you'll see:
```
ERROR: lm-eval-harness is not installed or not in PATH.
Please install it with:
  pip install lm-eval>=0.4.0
Or add it to requirements.txt and run:
  pip install -r requirements.txt
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
