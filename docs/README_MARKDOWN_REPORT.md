# Markdown Report Generator

A tool to generate comprehensive markdown benchmark reports with embedded visualizations from benchmark results.

## Purpose

This tool reads benchmark data from CSV files, generates comparison plots (TP comparison and configuration comparison), and produces individual markdown reports for each model with progress tables and embedded visualizations. It creates a master index report summarizing all benchmarks.

## Usage

### Basic Usage

Generate markdown reports from your run_list file:

```bash
python tools/reports/generate_markdown_report.py --run-list tests/run_list/total.sh
```

This generates:
- Individual model reports in `reports/{model}_report.md`
- Comparison plots in `reports/plots/{model}/`
- Master index report at `reports/README.md`

### With Custom Latencies

Specify which latency metrics to display (default: e2e,itl,ttft,interactivity):

```bash
python tools/reports/generate_markdown_report.py \
  --run-list tests/run_list/total.sh \
  --latencies e2e,itl,interactivity
```

Available metrics: `e2e`, `itl`, `ttft`, `interactivity`

### With Custom Throughput Metric

Specify the throughput metric to plot (default: tokens_per_sec_per_gpu):

```bash
python tools/reports/generate_markdown_report.py \
  --run-list tests/run_list/total.sh \
  --throughput tokens_per_sec
```

Available metrics:
- `tokens_per_sec_per_gpu` (default) - Tokens per second per GPU
- `tokens_per_sec` - Total tokens per second
- `output_tokens_per_sec_per_gpu` - Output tokens per second per GPU
- `output_tokens_per_sec` - Total output tokens per second

### With Custom Output Directory

```bash
python tools/reports/generate_markdown_report.py \
  --run-list tests/run_list/total.sh \
  --reports-dir custom_reports
```

### Custom Logs Directory

```bash
python tools/reports/generate_markdown_report.py \
  --run-list tests/run_list/total.sh \
  --logs-dir custom_logs
```

### Verbose Output

```bash
python tools/reports/generate_markdown_report.py \
  --run-list tests/run_list/total.sh \
  --verbose
```

## How It Works

1. **Updates test_results.tsv** - Runs progress report to get current test status
   - `success`: All expected test cases completed
   - `failure`: Test encountered errors (with specific failure reason)
   - `incomplete`: Test was interrupted/stopped (partial results exist)
   - `not_tested`: No results exist yet
2. **Loads benchmark data** - Reads all CSV files referenced in test_results.tsv (only successful and incomplete tests with data)
3. **Sorts and groups data** - Organizes data by:
   - model_config (alphabetical)
   - tp_size (increasing)
   - output_length (increasing)
   - input_length (increasing)
   - concurrency (increasing)
4. **Generates plots** - Creates two types of comparison plots:
   - **TP Comparison**: Compares tensor parallel sizes for same config/ISL/OSL
   - **Config Comparison**: Compares different configurations for same TP/ISL/OSL
5. **Creates reports** - Generates markdown files with:
   - Progress table showing test status
   - Embedded visualization plots
   - Source file references
6. **Generates index** - Creates master README linking to all model reports

## Plot Features

### Differentiation by Sequence Length

Plots are separated by input sequence length (ISL) and output sequence length (OSL) combinations:
- ISL: 1024 / OSL: 1024
- ISL: 1024 / OSL: 8192
- ISL: 8192 / OSL: 1024

Each combination gets its own plot for better clarity.

### Legend Positioning

- Legends are positioned at the bottom of each plot
- Displayed vertically (ncol=1) for better readability
- Uses `bbox_to_anchor` for precise positioning

### Clean Labels

- Config model names have `configs/models/` prefix removed
- Labels show only the model config name (e.g., `llama-vllm` instead of `configs/models/llama-vllm.yaml`)

### Data Sorting

All data points in plots are properly sorted by:
1. Tensor parallel size (increasing)
2. Output sequence length (increasing)
3. Input sequence length (increasing)
4. Concurrency (increasing)

This ensures smooth, connected lines in plots without disconnected points.

## Output Structure

```
reports/
├── README.md                              # Master index report
├── test_results.tsv                       # Current test status
├── {model}_report.md                      # Individual model reports
├── model1_report.md
├── model2_report.md
└── plots/
    ├── {model}/
    │   ├── {model}_tp_compare_*.png       # TP comparison plots
    │   └── {model}_config_compare_*.png   # Config comparison plots
    ├── model1/
    └── model2/
```

## Report Sections

### Progress Table

Shows test status for each configuration:
| Column | Description |
|--------|-------------|
| model_config | Config filename |
| image_tag | Docker image tag |
| tp_size | Tensor parallelism size |
| test_plan | Test plan name |
| result | Status: `success`, `failure`, `incomplete`, or `not_tested` |

**Status Meanings**:
- **`success`**: Test completed fully with all expected results
- **`failure`**: Test encountered an error/exception and failed to complete (has specific error reason)
- **`incomplete`**: Test was stopped or interrupted during execution (has partial results but not all expected data points)
- **`not_tested`**: Test has not been run yet

### TP Comparison Plots

Plots showing how throughput and latency vary with tensor parallel size:
- X-axis: Tensor parallel size (1, 2, 4, 8)
- Y-axis: Throughput or latency metric
- 3 subplots: One for each latency metric (e2e, itl, interactivity)
- Multiple lines: One for each configuration variant

### Config Comparison Plots

Plots comparing different model configurations:
- X-axis: Configuration name
- Y-axis: Throughput or latency metric
- 3 subplots: One for each latency metric
- Single plot: For maximum TP size

## Data Cleaning

The tool automatically:
- Removes entries with concurrency 96 and 112 (typically duplicate/edge cases)
- Deduplicates identical data points based on key columns
- Sorts by all relevant dimensions for consistent visualization

## Requirements

- Python 3.8+
- pandas
- matplotlib
- PyYAML

Install dependencies:
```bash
pip install pandas matplotlib pyyaml
```

## Examples

### Generate reports with default settings

```bash
python tools/reports/generate_markdown_report.py \
  --run-list tests/run_list/total.sh
```

### Generate reports showing only key latency metrics

```bash
python tools/reports/generate_markdown_report.py \
  --run-list tests/run_list/total.sh \
  --latencies e2e,itl,interactivity
```

### Generate reports with output throughput metric

```bash
python tools/reports/generate_markdown_report.py \
  --run-list tests/run_list/total.sh \
  --throughput output_tokens_per_sec \
  --latencies e2e,itl,interactivity
```

### Generate reports with custom paths

```bash
python tools/reports/generate_markdown_report.py \
  --run-list tests/run_list/subset.sh \
  --reports-dir ./custom_reports \
  --logs-dir ./custom_logs
```

## Troubleshooting

### "Could not import plotter modules"

The script requires `plot_benchmark_results.py` in the `tools/plotter/` directory. Ensure this file exists and the path is correct.

### "No benchmark data points loaded"

The CSV files referenced in test_results.tsv may not exist or are in a different location. Verify:
- Test results are generated (run progress report first)
- CSV files exist at the paths specified in test_results.tsv
- Logs directory path is correct

### Plots are empty or missing data

Common causes:
- CSV files don't have the expected columns
- Data filtering removed all points
- ISL/OSL combinations have no matching data

Check the verbose output for more details.

### Legend overlaps with plot data

The legend position may need adjustment in `plot_benchmark_results.py`. The `bbox_to_anchor` parameter controls vertical spacing.

## Customization

### Changing Plot Dimensions

Edit `tools/plotter/plot_benchmark_results.py`:
- `fig_width = 16` - Width of subplots (pixels)
- `fig_height = 7` - Height of subplots (pixels)

### Changing Legend Position

Edit in `plot_benchmark_results.py`:
- `bbox_to_anchor=(0.5, -0.35)` - Y-position of legend
- Increase negative value to move legend further down

### Changing Color Scheme

Modify color assignments in plotting functions to customize line colors for different configurations.

## Workflow

### Complete Analysis Workflow

1. **Run benchmarks** (distributed or sequential)
   ```bash
   bash tools/runs/run.sh benchmark tests/run_list/total.sh
   ```

2. **Check progress**
   ```bash
   python tools/reports/generate_progress_report.py \
     --run-list tests/run_list/total.sh \
     --generate-run-list
   ```

3. **Run incomplete tests** (if needed)
   ```bash
   bash tools/runs/run.sh benchmark tests/run_list/incomplete.sh
   ```

4. **Generate markdown reports** (final step)
   ```bash
   python tools/reports/generate_markdown_report.py \
     --run-list tests/run_list/total.sh \
     --latencies e2e,itl,interactivity
   ```

5. **View reports**
   - Master index: `reports/README.md`
   - Individual models: `reports/{model}_report.md`

## See Also

- [README_REPORTER.md](README_REPOTER.md) - Progress report generator documentation
- [README_RUN.md](README_RUN.md) - Multi-run tool documentation
- [README_PLOTTER.md](README_PLOTTER.md) - Plotter utilities documentation
