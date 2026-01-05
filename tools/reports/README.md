This folder contains benchmark reporting tools for tracking progress and generating visualizations.

## Report generation tools

### 1. Progress Report Generator
Tracks benchmark test status and generates run lists for incomplete/not tested items.

- **Script**: `python tools/reports/generate_progress_report.py`
- **Documentation**: [docs/README_REPOTER.md](../../docs/README_REPOTER.md)
- **Purpose**: Check which tests are complete, incomplete, or not tested
- **Output**: TSV file with test status and run lists for reruns

**Quick start:**
```bash
python tools/reports/generate_progress_report.py --run-list tests/run_list/run_list.sh --generate-run-list
```

### 2. Markdown Report Generator
Generates comprehensive benchmark reports with embedded plots and analysis.

- **Script**: `python tools/reports/generate_markdown_report.py`
- **Documentation**: [docs/README_MARKDOWN_REPORT.md](../../docs/README_MARKDOWN_REPORT.md)
- **Purpose**: Create markdown reports with TP/config comparison plots
- **Output**: Markdown reports and PNG visualizations

**Quick start:**
```bash
python tools/reports/generate_markdown_report.py --run-list tests/run_list/total.sh --latencies e2e,itl,interactivity
```

## Typical Workflow

1. **Run benchmarks** (distributed or sequential)
2. **Check progress** with progress report generator
3. **Rerun incomplete tests** if needed
4. **Generate markdown reports** for final analysis

See individual tool documentation for detailed usage and examples.