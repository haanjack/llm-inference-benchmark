# LLM Benchmark Plotter

Generate performance comparison charts for LLM inference benchmarks.

## Quick Start

### Streamlit GUI (Recommended)
```bash
# Install dependencies
pip install -r tools/plotter/requirements-plotter.txt

# Launch GUI
streamlit run tools/plotter/plotter_streamlit_app.py
```

The GUI provides:
- Model/config/tag selection
- TP size or config/tag comparison modes
- Throughput metric selection (per-GPU vs total, output vs total)
- Latency subset selection (E2E, ITL, TTFT, Interactivity)
- Strict ISL/OSL separation (auto-generates multiple figures)
- PNG export with custom naming

### Command Line

#### Compare TP sizes (same config):
```bash
python tools/plotter/plot_benchmark_results.py \
  --compare-tp \
  --model "amd/Llama-3.1-70B-Instruct-FP8-KV" \
  --image-tag "rocm7.0.0_vllm_0.11.2_20251210" \
  --model-config "llama-vllm" \
  --throughput tokens_per_sec_per_gpu \
  --latencies e2e,itl,ttft,interactivity \
  --input-length 1024 \
  --output-length 1024 \
  --output-dir plots \
  --output llama_tp_compare.png
```

#### Compare configs/tags (same TP):
```bash
python tools/plotter/plot_benchmark_results.py \
  --compare-configs \
  --model "amd/Llama-3.1-70B-Instruct-FP8-KV" \
  --tp-size 4 \
  --throughput tokens_per_sec_per_gpu \
  --latencies e2e,itl,interactivity \
  --input-length 1024 \
  --output-length 1024 \
  --output-dir plots \
  --output llama_config_compare.png
```

### CLI Options

**Comparison Modes:**
- `--compare-tp`: Compare TP sizes for single model/config/tag
- `--compare-configs`: Compare configs/tags for single model/TP size

**Filters:**
- `--model`: Model name (required)
- `--image-tag`: Docker image tag (required for --compare-tp)
- `--model-config`: Model config (required for --compare-tp)
- `--tp-size`: TP size (required for --compare-configs)
- `--input-length`, `--output-length`: ISL/OSL filters (optional; if omitted, generates one figure per combo)

**Metrics:**
- `--throughput`: Y-axis metric
  - `tokens_per_sec_per_gpu` (default): Total tokens/sec/GPU
  - `tokens_per_sec`: Total tokens/sec
  - `output_tokens_per_sec_per_gpu`: Output tokens/sec/GPU
  - `output_tokens_per_sec`: Output tokens/sec
- `--latencies`: Comma-separated X-axes (default: `e2e,itl,ttft,interactivity`)
  - `e2e`: End-to-End latency
  - `itl`: Inter-Token latency
  - `ttft`: Time to First Token
  - `interactivity`: E2E per user (E2E latency / concurrency)

**Output:**
- `--output-dir`: Directory for saved charts (default: `plots`)
- `--output`: Filename (auto-suffixed with `_islX_oslY.png` when ISL/OSL not fixed)

## Chart Features

- **Dynamic Layout**: 1×N subplots based on selected latencies
- **Series Legend**: Per-subplot legends for TP sizes or config/tag combinations
- **Concurrency Annotations**: Each point labeled with concurrency value
- **Adjacency-Only Lines**: Connect only adjacent doubling steps (1→2→4→8→16)
- **ISL/OSL Separation**: Never mix different sequence length combinations
- **Per-GPU Throughput**: Normalized by TP size for fair comparison

## Data Requirements

Expects TSV file at `logs/test_results.tsv` with columns:
- `model`, `image_tag`, `model_config`, `tp_size`, `concurrency`
- `input_length`, `output_length`, `request_rate`
- `e2el_mean_ms`, `itl_mean_ms`, `ttft_mean_ms`
- `total_token_throughput_tps`, `output_token_throughput_tps`

Missing columns are computed automatically when possible.
