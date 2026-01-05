#!/usr/bin/env python3
"""
Benchmark Results Plotter

This tool reads test_results.tsv and the referenced log files to visualize
benchmark performance metrics. It generates comparison graphs showing:
  - Total token throughput vs End-to-End latency
  - Total token throughput vs Inter-Token Latency (ITL)
  - Total token throughput vs Time-To-First-Token (TTFT)
  - Total token throughput vs ITL per user (ITL/concurrency)

Each data point is labeled with the concurrency value for detailed analysis.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import matplotlib.lines as mlines


class BenchmarkDataLoader:
    """Load and manage benchmark results from test_results.tsv and CSV files."""

    def __init__(self, logs_dir: str = "logs"):
        """
        Initialize data loader.

        Args:
            logs_dir: Root directory containing benchmark logs
        """
        self.logs_dir = Path(logs_dir)
        self.test_results_file = self.logs_dir / "test_results.tsv"
        self.data = None
        self.available_models = set()
        self.available_image_tags = set()
        self.available_model_configs = set()
        self.available_tp_sizes = set()
        self.available_request_rates = set()

    def load_data(self) -> pd.DataFrame:
        """
        Load test results from test_results.tsv and merge with CSV data.

        Returns:
            DataFrame with all benchmark results
        """
        if not self.test_results_file.exists():
            raise FileNotFoundError(f"test_results.tsv not found at {self.test_results_file}")

        # Load test_results.tsv
        test_results = pd.read_csv(self.test_results_file, sep='\t')

        # Handle legacy corrupted header combining 'result' and 'log_path'
        if 'result' not in test_results.columns and 'log_path' not in test_results.columns:
            combined_cols = [c for c in test_results.columns if 'result' in c and 'log_path' in c]
            if combined_cols:
                col = combined_cols[0]
                split_vals = test_results[col].astype(str).str.split(r"\s+", n=1, expand=True)
                test_results['result'] = split_vals[0]
                test_results['log_path'] = split_vals[1] if split_vals.shape[1] > 1 else ''

        # Backward + forward compatible filtering
        # New format: columns include 'result' (success/failure) and 'log_path'
        # Old format: 'result' may contain the CSV path or 'failure'
        has_log_path = 'log_path' in test_results.columns
        if has_log_path and 'result' in test_results.columns:
            mask = (test_results['result'].str.lower() == 'success') & (test_results['log_path'].notna())
            filtered_rows = test_results[mask].copy()
        elif 'result' in test_results.columns:
            mask = (test_results['result'] != 'failure')
            filtered_rows = test_results[mask].copy()
        else:
            raise ValueError(f"test_results.tsv missing expected columns. Found: {list(test_results.columns)}")

        # Load individual CSV files and concatenate
        dataframes = []
        for idx, row in filtered_rows.iterrows():
            csv_path = row['log_path'] if has_log_path else row['result']

            # The path in test_results.tsv is relative to workspace root
            # Need to check if it's absolute or relative to logs_dir
            full_csv_path = Path(csv_path) if Path(csv_path).is_absolute() else self.logs_dir.parent / str(csv_path)

            if not full_csv_path.exists():
                # Try without the 'logs/' prefix if it exists (in case it's duplicated)
                csv_path_str = str(csv_path)
                csv_path_fixed = csv_path_str.replace('logs/', '', 1) if 'logs/' in csv_path_str else csv_path_str
                full_csv_path = self.logs_dir.parent / csv_path_fixed

                if not full_csv_path.exists():
                    print(f"Warning: {full_csv_path} not found, skipping", file=sys.stderr)
                    continue

            try:
                df = pd.read_csv(full_csv_path)
                # Add metadata columns from test_results.tsv
                df['model'] = row['model']
                df['image_tag'] = row['image_tag']
                df['model_config'] = row['model_config']
                df['timestamp'] = row['timestamp']
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {full_csv_path}: {e}", file=sys.stderr)
                continue

        if not dataframes:
            raise ValueError("No valid benchmark data found")

        self.data = pd.concat(dataframes, ignore_index=True)

        # Derived metrics
        if not self.data.empty:
            # Per-GPU throughput: total tokens per second divided by tp_size
            if 'total_token_throughput_tps' in self.data.columns:
                def _per_gpu(row):
                    try:
                        tp = int(row.get('tp_size', 0))
                        return row['total_token_throughput_tps'] / tp if tp else row['total_token_throughput_tps']
                    except Exception:
                        return row['total_token_throughput_tps']
                self.data['total_token_throughput_tps_per_gpu'] = self.data.apply(_per_gpu, axis=1)

            # Output tokens per GPU throughput
            if 'output_token_throughput_tps' in self.data.columns:
                def _out_per_gpu(row):
                    try:
                        tp = int(row.get('tp_size', 0))
                        return row['output_token_throughput_tps'] / tp if tp else row['output_token_throughput_tps']
                    except Exception:
                        return row['output_token_throughput_tps']
                self.data['output_token_throughput_tps_per_gpu'] = self.data.apply(_out_per_gpu, axis=1)

            # Interactivity: End-to-End latency per user (ms/user)
            if 'e2el_mean_ms' in self.data.columns and 'concurrency' in self.data.columns:
                def _e2e_per_user(row):
                    try:
                        cc = float(row.get('concurrency', 0))
                        return row['e2el_mean_ms'] / cc if cc else row['e2el_mean_ms']
                    except Exception:
                        return row['e2el_mean_ms']
                self.data['e2e_per_user_ms'] = self.data.apply(_e2e_per_user, axis=1)

        # Update available options
        self.available_models = set(self.data['model'].unique())
        self.available_image_tags = set(self.data['image_tag'].unique())
        self.available_model_configs = set(self.data['model_config'].unique())
        self.available_tp_sizes = set(self.data['tp_size'].unique())
        self.available_request_rates = set(self.data['request_rate'].unique())

        return self.data

    def filter_data(self,
                   model: str,
                   image_tag: str,
                   model_config: str,
                   tp_size: Optional[int] = None,
                   request_rate: Optional[float] = None,
                   input_length: Optional[int] = None,
                   output_length: Optional[int] = None) -> pd.DataFrame:
        """
        Filter data by specified criteria.

        Args:
            model: Model name (e.g., 'amd/Llama-3.1-70B-Instruct-FP8-KV')
            image_tag: Docker image tag
            model_config: Model configuration name
            tp_size: Optional tensor parallel size
            request_rate: Optional request rate
            input_length: Optional input sequence length
            output_length: Optional output sequence length

        Returns:
            Filtered DataFrame
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first")

        filtered = self.data[
            (self.data['model'] == model) &
            (self.data['image_tag'] == image_tag) &
            (self.data['model_config'] == model_config)
        ].copy()

        if tp_size is not None:
            filtered = filtered[filtered['tp_size'] == tp_size]

        if request_rate is not None:
            filtered = filtered[filtered['request_rate'] == request_rate]

        if input_length is not None:
            filtered = filtered[filtered['input_length'] == input_length]

        if output_length is not None:
            filtered = filtered[filtered['output_length'] == output_length]

        return filtered

    def show_available_options(self,
                              model: str,
                              image_tag: str,
                              model_config: str) -> Dict[str, set]:
        """
        Show available options for optional filters after specifying required ones.

        Args:
            model: Model name
            image_tag: Docker image tag
            model_config: Model configuration name

        Returns:
            Dictionary with available options
        """
        filtered = self.data[
            (self.data['model'] == model) &
            (self.data['image_tag'] == image_tag) &
            (self.data['model_config'] == model_config)
        ]

        return {
            'tp_sizes': sorted(set(filtered['tp_size'].unique())),
            'request_rates': sorted(set(filtered['request_rate'].unique())),
            'input_lengths': sorted(set(filtered['input_length'].unique())),
            'output_lengths': sorted(set(filtered['output_length'].unique()))
        }


class BenchmarkPlotter:
    """Create visualization plots for benchmark results."""

    # Map of latency metrics
    LATENCY_METRICS = {
        'e2e': ('e2el_mean_ms', 'End-to-End Latency (ms)'),
        'itl': ('itl_mean_ms', 'Inter-Token Latency (ms)'),
        'ttft': ('ttft_mean_ms', 'Time-to-First-Token (ms)'),
        'e2e_per_user': ('e2e_per_user_ms', 'Interactivity (E2E/User, ms)')
    }

    def __init__(self, figsize: Tuple[int, int] = (16, 12)):
        """
        Initialize plotter.

        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize

    def plot_throughput_vs_latency(self,
                                   df: pd.DataFrame,
                                   title_prefix: str = "",
                                   output_file: Optional[str] = None) -> plt.Figure:
        """
        Create 4-subplot figure showing total token throughput vs different latency metrics.

        Args:
            df: Filtered DataFrame with benchmark results
            title_prefix: Prefix for plot title
            output_file: Optional file path to save the figure

        Returns:
            Matplotlib figure object
        """
        if df.empty:
            raise ValueError("DataFrame is empty - no data to plot")

        # Sort by input and output length for consistent grouping
        df = df.sort_values(['input_length', 'output_length', 'concurrency'])

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(
            f"{title_prefix}\nTotal Token Throughput vs Latency Metrics",
            fontsize=16, fontweight='bold'
        )

        # Get unique ISL/OSL combinations for coloring
        isl_osl_pairs = df[['input_length', 'output_length']].drop_duplicates()
        colors = plt.cm.tab20(np.linspace(0, 1, len(isl_osl_pairs)))
        color_map = {
            (row['input_length'], row['output_length']): colors[i]
            for i, (_, row) in enumerate(isl_osl_pairs.iterrows())
        }

        metrics = ['e2e', 'itl', 'ttft', 'e2e_per_user']
        axes_flat = axes.flatten()

        for ax, metric in zip(axes_flat, metrics):
            self._plot_single_metric(
                ax, df, metric, color_map, title_prefix
            )

        plt.tight_layout()

        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {output_file}")

        return fig

    def _plot_single_metric(self, ax, df: pd.DataFrame, metric: str,
                           color_map: Dict, title_prefix: str):
        """
        Plot throughput vs a single latency metric.

        Args:
            ax: Matplotlib axis
            df: DataFrame with data
            metric: Metric key ('e2e', 'itl', 'ttft', 'itl_per_user')
            color_map: Mapping of (input_length, output_length) to colors
            title_prefix: For reference in labels
        """
        if metric not in self.LATENCY_METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        col_name, y_label = self.LATENCY_METRICS[metric]

        # Determine series mode: tp_size when single config, else config label when single tp_size
        series_mode = None
        if 'model_config' in df.columns and 'tp_size' in df.columns:
            unique_configs = sorted(df['model_config'].unique())
            unique_tp = sorted(df['tp_size'].unique())
            if len(unique_configs) == 1 and len(unique_tp) > 1:
                series_mode = 'tp_size'
            elif len(unique_tp) == 1 and len(unique_configs) > 1:
                series_mode = 'config'

        # Build series color map
        if series_mode == 'tp_size':
            series_keys = sorted(set(df['tp_size']))
            cmap = plt.get_cmap('tab10')
            series_color = {tp: cmap(i % 10) for i, tp in enumerate(series_keys)}
        elif series_mode == 'config' and 'image_tag' in df.columns:
            series_keys = sorted(set((df['model_config'] + ' | ' + df['image_tag']).unique()))
            cmap = plt.get_cmap('tab10')
            # Build mapping based on per-row values
            series_color = {}
        elif series_mode == 'config':
            series_keys = sorted(set(df['model_config']))
            cmap = plt.get_cmap('tab10')
            series_color = {mc: cmap(i % 10) for i, mc in enumerate(series_keys)}
        else:
            # Fall back to ISL/OSL coloring
            isl_osl_pairs = df[['input_length', 'output_length']].drop_duplicates()
            colors = plt.cm.tab20(np.linspace(0, 1, len(isl_osl_pairs)))
            series_color = {
                (row['input_length'], row['output_length']): colors[i]
                for i, (_, row) in enumerate(isl_osl_pairs.iterrows())
            }

        # Concurrency marker styles
        unique_conc = sorted(set(df['concurrency'])) if 'concurrency' in df.columns else []
        marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '*', '>']
        marker_map = {c: marker_cycle[i % len(marker_cycle)] for i, c in enumerate(unique_conc)}

        # Group by ISL/OSL to restrict line connections within same ISL/OSL
        isl_osl_pairs = df[['input_length', 'output_length']].drop_duplicates()
        # Track which series labels have been added to legend
        series_label_added = set()

        for _, row in isl_osl_pairs.iterrows():
            isl = row['input_length']
            osl = row['output_length']

            subset = df[
                (df['input_length'] == isl) &
                (df['output_length'] == osl)
            ].copy()

            if subset.empty:
                continue

            # Prepare points grouped by series key
            points_by_group = {}
            for _, r in subset.iterrows():
                # X and Y values
                if metric == 'e2e_per_user':
                    x_val = r['e2e_per_user_ms'] if 'e2e_per_user_ms' in subset.columns else (r['e2el_mean_ms'] / r['concurrency'])
                else:
                    x_val = r[col_name]
                y_val = r['total_token_throughput_tps_per_gpu'] if 'total_token_throughput_tps_per_gpu' in subset.columns else r['total_token_throughput_tps']

                # Determine series key and color
                if series_mode == 'tp_size':
                    key = r['tp_size']
                    color = series_color.get(key)
                    label = f"tp{int(key)}" if key not in series_label_added else None
                elif series_mode == 'config' and 'image_tag' in subset.columns:
                    key = f"{r['model_config']} | {r['image_tag']}"
                    if key not in series_color:
                        idx = len(series_color)
                        series_color[key] = plt.get_cmap('tab10')(idx % 10)
                    color = series_color[key]
                    label = key if key not in series_label_added else None
                elif series_mode == 'config':
                    key = r['model_config']
                    color = series_color.get(key)
                    label = key if key not in series_label_added else None
                else:
                    key = (isl, osl)
                    color = series_color.get(key)
                    label = f"ISL={isl}, OSL={osl}" if key not in series_label_added else None

                conc = r['concurrency'] if 'concurrency' in subset.columns else None
                # Scatter point
                ax.scatter([x_val], [y_val], s=80, alpha=0.75, color=color,
                           label=label, edgecolors='black', linewidth=0.5,
                           marker=marker_map.get(conc, 'o'))
                if label:
                    series_label_added.add(label)

                # Annotate with concurrency
                if conc is not None:
                    ax.annotate(str(int(conc)), (x_val, y_val), fontsize=8,
                                ha='center', va='center')

                # Group points for connection lines by series key
                points_by_group.setdefault(key, []).append((conc, x_val, y_val))

            # Draw connection lines only for adjacent doubling steps within series
            for key, points in points_by_group.items():
                pts = sorted([p for p in points if p[0] is not None], key=lambda p: p[0])
                for i in range(len(pts) - 1):
                    c1, x1, y1 = pts[i]
                    c2, x2, y2 = pts[i + 1]
                    if c2 == 2 * c1:
                        # Use series color
                        color = series_color.get(key, 'C0')
                        ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=1.2, alpha=0.6, color=color)

        ax.set_xlabel(y_label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Total Token Throughput per GPU (tok/s/gpu)', fontsize=11, fontweight='bold')
        ax.set_title(f'Throughput vs {y_label}', fontsize=12, fontweight='bold')
        # Single legend: series only (tp_size or config; else ISL/OSL fallback)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=9, ncol=4)
        ax.grid(True, alpha=0.3)

    def plot_comparison(self,
                       df_dict: Dict[str, pd.DataFrame],
                       metric: str = 'e2e',
                       output_file: Optional[str] = None) -> plt.Figure:
        """
        Create comparison plot for multiple data series.

        Args:
            df_dict: Dictionary mapping labels to DataFrames
            metric: Latency metric to plot ('e2e', 'itl', 'ttft', 'itl_per_user')
            output_file: Optional file path to save the figure

        Returns:
            Matplotlib figure object
        """
        if metric not in self.LATENCY_METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        col_name, y_label = self.LATENCY_METRICS[metric]

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(df_dict)))

        # Concurrency marker styles
        all_df = [d for d in df_dict.values() if not d.empty]
        conc_set = set()
        for d in all_df:
            if 'concurrency' in d.columns:
                conc_set.update(set(d['concurrency']))
        unique_conc = sorted(conc_set)
        marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '*', '>']
        marker_map = {c: marker_cycle[i % len(marker_cycle)] for i, c in enumerate(unique_conc)}

        for (label, df), color in zip(df_dict.items(), colors):
            if df.empty:
                continue

            if metric == 'e2e_per_user':
                x_values = df['e2e_per_user_ms'] if 'e2e_per_user_ms' in df.columns else (df['e2el_mean_ms'] / df['concurrency'])
            else:
                x_values = df[col_name]

            # per-GPU throughput on Y
            y_values = df['total_token_throughput_tps_per_gpu'] if 'total_token_throughput_tps_per_gpu' in df.columns else df['total_token_throughput_tps']

            # Plot each point with concurrency-specific marker
            first_label_done = False
            points_by_group = {}
            for _, r in df.iterrows():
                xv = r['e2e_per_user_ms'] if (metric == 'e2e_per_user' and 'e2e_per_user_ms' in df.columns) else (r[col_name] if metric != 'e2e_per_user' else (r['e2el_mean_ms'] / r['concurrency']))
                yv = r['total_token_throughput_tps_per_gpu'] if 'total_token_throughput_tps_per_gpu' in df.columns else r['total_token_throughput_tps']
                conc = r.get('concurrency', None)
                ax.scatter([xv], [yv], s=90, alpha=0.8, color=color,
                           label=(label if not first_label_done else None), edgecolors='black', linewidth=0.5,
                           marker=marker_map.get(conc, 'o'))
                first_label_done = True

                # annotate
                if conc is not None:
                    ax.annotate(str(int(conc)), (xv, yv), fontsize=8, ha='center', va='center')

                # lines per model_config + tp_size
                key = (r.get('model_config', ''), r.get('tp_size', None))
                points_by_group.setdefault(key, []).append((conc, xv, yv))

            # draw only adjacent doubling
            for key, points in points_by_group.items():
                pts = sorted([p for p in points if p[0] is not None], key=lambda p: p[0])
                for i in range(len(pts) - 1):
                    c1, x1, y1 = pts[i]
                    c2, x2, y2 = pts[i + 1]
                    if c2 == 2 * c1:
                        ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=1.2, alpha=0.6, color=color)

            ax.scatter(x_values, y_values, s=100, alpha=0.6, color=color,
                      label=label, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(y_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Token Throughput per GPU (tok/s/gpu)', fontsize=12, fontweight='bold')
        ax.set_title(f'Throughput vs {y_label} (Comparison)', fontsize=14, fontweight='bold')
        # Single legend: series only
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=10, ncol=4)
        ax.grid(True, alpha=0.3)

        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {output_file}")

        return fig

    def plot_config_tp_comparison(self,
                                  df: pd.DataFrame,
                                  title_prefix: str = "",
                                  output_file: Optional[str] = None,
                                  throughput: str = 'tokens_per_sec_per_gpu',
                                  latencies: Optional[List[str]] = None) -> plt.Figure:
        """Plot 4 subplots for one model_config comparing different TP sizes.

        - Y-axis: total token throughput per GPU
        - X-axes: E2E, ITL, TTFT, Interactivity (E2E/User)
        - Series: TP sizes (tp1, tp2, ...)
        - Points: per-concurrency aggregated within each ISL/OSL (mean)
        - Lines: connect only adjacent doubling concurrencies (1→2→4→8→…)
        """
        if df is None or df.empty:
            raise ValueError("Empty dataframe provided")

        required_cols = {'tp_size', 'concurrency', 'total_token_throughput_tps'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure per-GPU throughput and interactivity
        dfx = df.copy()
        if 'total_token_throughput_tps_per_gpu' not in dfx.columns:
            dfx['total_token_throughput_tps_per_gpu'] = dfx['total_token_throughput_tps'] / dfx['tp_size'].replace(0, 1)
        if 'e2e_per_user_ms' not in dfx.columns and 'e2el_mean_ms' in dfx.columns:
            dfx['e2e_per_user_ms'] = dfx['e2el_mean_ms'] / dfx['concurrency'].replace(0, 1)

        # Select metrics to draw first, then create the dynamic layout
        latency_map = {
            'e2e': ('e2el_mean_ms', 'End-to-End Latency (ms)'),
            'itl': ('itl_mean_ms', 'Inter-Token Latency (ms)'),
            'ttft': ('ttft_mean_ms', 'Time to First Token (ms)'),
            'interactivity': ('e2e_per_user_ms', 'Interactivity (E2E/User, ms)')
        }
        latencies = latencies or ['e2e', 'itl', 'ttft', 'interactivity']
        metric_specs = [latency_map[l] for l in latencies if l in latency_map]
        if not metric_specs:
            raise ValueError("No valid latency metrics selected. Choose from {e2e,itl,ttft,interactivity}.")

        # Ensure output per-GPU throughput exists if selected
        if throughput == 'output_tokens_per_sec_per_gpu' and 'output_token_throughput_tps_per_gpu' not in dfx.columns and 'output_token_throughput_tps' in dfx.columns:
            dfx['output_token_throughput_tps_per_gpu'] = dfx['output_token_throughput_tps'] / dfx['tp_size'].replace(0, 1)

        # Axes and color mapping per TP size
        # Dynamic 1xN layout based on selected latencies
        n_plots = len(metric_specs)
        fig_width = max(6 * n_plots, 8)
        fig_height = 5
        fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, fig_height))
        if n_plots == 1:
            axes = [axes]
        fig.suptitle(f"{title_prefix}\nTP-size Comparison (same model-config)", fontsize=14, fontweight='bold')

        tp_sizes = sorted(dfx['tp_size'].unique())
        cmap = plt.get_cmap('tab10')
        tp_color = {tp: cmap(i % 10) for i, tp in enumerate(tp_sizes)}

        # Group by ISL/OSL to limit line connections within the same seq lengths
        if 'input_length' in dfx.columns and 'output_length' in dfx.columns:
            isl_osl_groups = dfx.groupby(['input_length', 'output_length'])
        else:
            # Single group if lengths not provided
            dfx['_isl'] = -1
            dfx['_osl'] = -1
            isl_osl_groups = dfx.groupby(['_isl', '_osl'])

        for ax, (x_col, x_label) in zip(axes, metric_specs):
            for tp in tp_sizes:
                color = tp_color[tp]
                series_label_added = False
                for (_, _), g in isl_osl_groups:
                    gtp = g[g['tp_size'] == tp]
                    if gtp.empty or x_col not in gtp.columns:
                        continue

                    # Aggregate per concurrency for stable lines
                    y_col = {
                        'tokens_per_sec_per_gpu': 'total_token_throughput_tps_per_gpu',
                        'tokens_per_sec': 'total_token_throughput_tps',
                        'output_tokens_per_sec_per_gpu': 'output_token_throughput_tps_per_gpu',
                        'output_tokens_per_sec': 'output_token_throughput_tps',
                    }[throughput]
                    agg = gtp.groupby('concurrency').agg({x_col: 'mean', y_col: 'mean'}).reset_index()

                    # Scatter points and annotations
                    ax.scatter(agg[x_col], agg[y_col],
                               color=color, s=60, alpha=0.85,
                               edgecolors='black', linewidth=0.5,
                               label=(f"tp{int(tp)}" if not series_label_added else None))
                    series_label_added = True
                    for _, r in agg.iterrows():
                        ax.annotate(str(int(r['concurrency'])),
                                    (r[x_col], r[y_col]),
                                    fontsize=8, ha='center', va='bottom')

                    # Connect only adjacent doubling steps
                    concs = sorted(agg['concurrency'].unique())
                    conc_idx = {c: i for i, c in enumerate(concs)}
                    for c in concs:
                        nxt = c * 2
                        if nxt in conc_idx:
                            i = conc_idx[c]
                            j = conc_idx[nxt]
                            ax.plot([agg.loc[i, x_col], agg.loc[j, x_col]],
                                    [agg.loc[i, y_col], agg.loc[j, y_col]],
                                    color=color, linestyle='-', linewidth=1.5, alpha=0.7)

            ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
            y_label_map = {
                'tokens_per_sec_per_gpu': 'Throughput (tokens/sec/gpu)',
                'tokens_per_sec': 'Throughput (tokens/sec)',
                'output_tokens_per_sec_per_gpu': 'Output Throughput (tokens/sec/gpu)',
                'output_tokens_per_sec': 'Output Throughput (tokens/sec)'
            }
            ax.set_ylabel(y_label_map.get(throughput, 'Throughput'), fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)

        # Legend on each subplot
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(title='TP Size', fontsize=9, loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=4)
        plt.tight_layout()
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {output_file}")
        return fig

    def plot_tp_config_comparison(self,
                                  df: pd.DataFrame,
                                  title_prefix: str = "",
                                  output_file: Optional[str] = None,
                                  throughput: str = 'tokens_per_sec_per_gpu',
                                  latencies: Optional[List[str]] = None) -> plt.Figure:
        """Plot 4 subplots for one tp_size comparing different configs/tags.

        - Filtered to a single tp_size beforehand
        - Series: model_config | image_tag (or model_config if tag missing)
        - Y-axis: per-GPU throughput; X-axes: E2E, ITL, TTFT, Interactivity
        - Points annotated by concurrency, lines connect adjacent doubling steps
        """
        if df is None or df.empty:
            raise ValueError("Empty dataframe provided")

        dfx = df.copy()
        if 'total_token_throughput_tps_per_gpu' not in dfx.columns:
            dfx['total_token_throughput_tps_per_gpu'] = dfx['total_token_throughput_tps'] / dfx['tp_size'].replace(0, 1)
        if 'e2e_per_user_ms' not in dfx.columns and 'e2el_mean_ms' in dfx.columns:
            dfx['e2e_per_user_ms'] = dfx['e2el_mean_ms'] / dfx['concurrency'].replace(0, 1)

        # Define metrics to plot before creating axes
        latency_map = {
            'e2e': ('e2el_mean_ms', 'End-to-End Latency (ms)'),
            'itl': ('itl_mean_ms', 'Inter-Token Latency (ms)'),
            'ttft': ('ttft_mean_ms', 'Time to First Token (ms)'),
            'interactivity': ('e2e_per_user_ms', 'Interactivity (E2E/User, ms)')
        }
        latencies = latencies or ['e2e', 'itl', 'ttft', 'interactivity']
        metric_specs = [latency_map[l] for l in latencies if l in latency_map]
        if not metric_specs:
            raise ValueError("No valid latency metrics selected. Choose from {e2e,itl,ttft,interactivity}.")

        # Ensure output per-GPU throughput exists if selected
        if throughput == 'output_tokens_per_sec_per_gpu' and 'output_token_throughput_tps_per_gpu' not in dfx.columns and 'output_token_throughput_tps' in dfx.columns:
            dfx['output_token_throughput_tps_per_gpu'] = dfx['output_token_throughput_tps'] / dfx['tp_size'].replace(0, 1)

        # Dynamic 1xN layout based on selected latencies
        n_plots = len(metric_specs)
        fig_width = max(6 * n_plots, 8)
        fig_height = 7
        fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, fig_height))
        if n_plots == 1:
            axes = [axes]
        fig.suptitle(f"{title_prefix}\nConfig/Tag Comparison (same TP)", fontsize=14, fontweight='bold')

        # Series labels
        if 'image_tag' in dfx.columns:
            dfx['series_label'] = dfx['model_config'].astype(str).str.replace('configs/models/', '', regex=False) + ' | ' + dfx['image_tag'].astype(str)
        else:
            dfx['series_label'] = dfx['model_config'].astype(str).str.replace('configs/models/', '', regex=False)

        series_labels = sorted(dfx['series_label'].unique())
        cmap = plt.get_cmap('tab10')
        series_color = {lab: cmap(i % 10) for i, lab in enumerate(series_labels)}

        # metric_specs already defined above

        # Group by ISL/OSL
        if 'input_length' in dfx.columns and 'output_length' in dfx.columns:
            isl_osl_groups = dfx.groupby(['input_length', 'output_length'])
        else:
            dfx['_isl'] = -1
            dfx['_osl'] = -1
            isl_osl_groups = dfx.groupby(['_isl', '_osl'])

        for ax, (x_col, x_label) in zip(axes, metric_specs):
            for lab in series_labels:
                color = series_color[lab]
                label_added = False
                for (_, _), g in isl_osl_groups:
                    gs = g[g['series_label'] == lab]
                    if gs.empty or x_col not in gs.columns:
                        continue

                    y_col = {
                        'tokens_per_sec_per_gpu': 'total_token_throughput_tps_per_gpu',
                        'tokens_per_sec': 'total_token_throughput_tps',
                        'output_tokens_per_sec_per_gpu': 'output_token_throughput_tps_per_gpu',
                        'output_tokens_per_sec': 'output_token_throughput_tps',
                    }[throughput]
                    agg = gs.groupby('concurrency').agg({x_col: 'mean', y_col: 'mean'}).reset_index()

                    ax.scatter(agg[x_col], agg[y_col],
                               color=color, s=60, alpha=0.85,
                               edgecolors='black', linewidth=0.5,
                               label=(lab if not label_added else None))
                    label_added = True
                    for _, r in agg.iterrows():
                        ax.annotate(str(int(r['concurrency'])), (r[x_col], r[y_col]),
                                    fontsize=8, ha='center', va='bottom')

                    concs = sorted(agg['concurrency'].unique())
                    conc_idx = {c: i for i, c in enumerate(concs)}
                    for c in concs:
                        nxt = c * 2
                        if nxt in conc_idx:
                            i = conc_idx[c]
                            j = conc_idx[nxt]
                            ax.plot([agg.loc[i, x_col], agg.loc[j, x_col]],
                                    [agg.loc[i, y_col], agg.loc[j, y_col]],
                                    color=color, linestyle='-', linewidth=1.5, alpha=0.7)

            ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
            y_label_map = {
                'tokens_per_sec_per_gpu': 'Throughput (tokens/sec/gpu)',
                'tokens_per_sec': 'Throughput (tokens/sec)',
                'output_tokens_per_sec_per_gpu': 'Output Throughput (tokens/sec/gpu)',
                'output_tokens_per_sec': 'Output Throughput (tokens/sec)'
            }
            ax.set_ylabel(y_label_map.get(throughput, 'Throughput'), fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)

        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(title='Config | Tag', fontsize=9, loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=1)
        plt.tight_layout()
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {output_file}")
        return fig


def interactive_mode(loader: BenchmarkDataLoader):
    """Run interactive mode for user input."""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS PLOTTER - Interactive Mode")
    print("="*70)

    # Load data
    print("\nLoading benchmark data...")
    loader.load_data()
    print(f"✓ Loaded data from {loader.test_results_file}")
    print(f"  Models: {len(loader.available_models)}")
    print(f"  Image tags: {len(loader.available_image_tags)}")
    print(f"  Model configs: {len(loader.available_model_configs)}")

    # Ask for required parameters
    print("\n" + "-"*70)
    print("REQUIRED PARAMETERS")
    print("-"*70)

    print("\nAvailable models:")
    models = sorted(loader.available_models)
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    model_choice = int(input("\nSelect model number: ")) - 1
    selected_model = models[model_choice]

    print("\nAvailable image tags:")
    image_tags = sorted(loader.available_image_tags)
    for i, tag in enumerate(image_tags, 1):
        print(f"  {i}. {tag}")
    tag_choice = int(input("\nSelect image tag number: ")) - 1
    selected_tag = image_tags[tag_choice]

    print("\nAvailable model configs:")
    model_configs = sorted(loader.available_model_configs)
    for i, config in enumerate(model_configs, 1):
        print(f"  {i}. {config}")
    config_choice = int(input("\nSelect model config number: ")) - 1
    selected_config = model_configs[config_choice]

    # Show available options for optional parameters
    print("\n" + "-"*70)
    print("OPTIONAL PARAMETERS")
    print("-"*70)
    available = loader.show_available_options(
        selected_model, selected_tag, selected_config
    )

    print("\nAvailable TP sizes:", available['tp_sizes'])
    print("Available request rates:", available['request_rates'])
    print("Available input lengths:", available['input_lengths'])
    print("Available output lengths:", available['output_lengths'])

    # Get filters
    tp_input = input("\nTP size (leave blank for all): ").strip()
    tp_size = int(tp_input) if tp_input else None

    rr_input = input("Request rate (leave blank for all): ").strip()
    request_rate = float(rr_input) if rr_input else None

    isl_input = input("Input sequence length (leave blank for all): ").strip()
    input_length = int(isl_input) if isl_input else None

    osl_input = input("Output sequence length (leave blank for all): ").strip()
    output_length = int(osl_input) if osl_input else None

    # Filter and plot
    print("\nFiltering data...")
    filtered_data = loader.filter_data(
        selected_model, selected_tag, selected_config,
        tp_size=tp_size, request_rate=request_rate,
        input_length=input_length, output_length=output_length
    )

    if filtered_data.empty:
        print("ERROR: No data matches the selected criteria!")
        return

    print(f"✓ Found {len(filtered_data)} data points")

    # Create plots
    print("\nGenerating plots...")
    plotter = BenchmarkPlotter()

    title = f"{selected_model} | {selected_config} | {selected_tag}"
    if tp_size:
        title += f" | TP={tp_size}"
    if request_rate is not None:
        title += f" | RR={request_rate}"

    fig = plotter.plot_throughput_vs_latency(
        filtered_data,
        title_prefix=title,
        output_file=None
    )

    print("✓ Plots generated successfully!")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from test_results.tsv"
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Path to logs directory containing test_results.tsv"
    )
    parser.add_argument(
        "--model",
        help="Model name to filter (e.g., 'amd/Llama-3.1-70B-Instruct-FP8-KV')"
    )
    parser.add_argument(
        "--image-tag",
        help="Docker image tag to filter"
    )
    parser.add_argument(
        "--model-config",
        help="Model configuration to filter"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        help="Tensor parallel size (optional filter)"
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        help="Request rate (optional filter)"
    )
    parser.add_argument(
        "--input-length",
        type=int,
        help="Input sequence length (optional filter)"
    )
    parser.add_argument(
        "--output-length",
        type=int,
        help="Output sequence length (optional filter)"
    )
    parser.add_argument(
        "--output",
        help="Output file path for the figure"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--throughput",
        choices=[
            'tokens_per_sec_per_gpu',
            'tokens_per_sec',
            'output_tokens_per_sec_per_gpu',
            'output_tokens_per_sec'
        ],
        default='tokens_per_sec_per_gpu',
        help="Y-axis throughput metric"
    )
    parser.add_argument(
        "--latencies",
        default='e2e,itl,ttft,interactivity',
        help="Comma-separated latency metrics to plot from {e2e,itl,ttft,interactivity}"
    )
    parser.add_argument(
        "--per-model",
        action="store_true",
        help="Generate 4×(TP sizes + configs) charts for the given --model across all tags/configs"
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to save generated charts when using --per-model or providing --output"
    )
    parser.add_argument(
        "--compare-tp",
        action="store_true",
        help="Plot 4 subplots for a single model/image_tag/model_config comparing different TP sizes"
    )
    parser.add_argument(
        "--compare-configs",
        action="store_true",
        help="Plot 4 subplots for a single model and TP size comparing different model_config and image_tag"
    )

    args = parser.parse_args()

    # Initialize loader
    loader = BenchmarkDataLoader(args.logs_dir)

    # Interactive or command-line mode
    if args.interactive or (not args.per_model and not args.compare_tp and not args.compare_configs and not all([args.model, args.image_tag, args.model_config])):
        interactive_mode(loader)
    else:
        # Command-line mode
        print("Loading benchmark data...")
        loader.load_data()

        if args.per_model and args.model:
            print(f"Generating charts for model: {args.model}")
            out_dir = args.output_dir or "plots"
            os.makedirs(out_dir, exist_ok=True)
            generate_all_charts_for_model(loader, args.model, out_dir)
            print(f"✓ Charts saved under {out_dir}")
        elif args.compare_tp:
            # Require model, image_tag, model_config
            if not all([args.model, args.image_tag, args.model_config]):
                print("ERROR: --compare-tp requires --model, --image-tag, and --model-config")
                sys.exit(1)
            print("Filtering data for TP comparison:")
            print(f"  Model: {args.model}")
            print(f"  Image tag: {args.image_tag}")
            print(f"  Model config: {args.model_config}")

            filtered_data = loader.filter_data(
                args.model, args.image_tag, args.model_config,
                tp_size=args.tp_size, request_rate=args.request_rate,
                input_length=args.input_length, output_length=args.output_length
            )
            if filtered_data.empty:
                print("ERROR: No data matches the specified criteria!")
                sys.exit(1)

            plotter = BenchmarkPlotter()
            base_title = f"{args.model} | {args.model_config} | {args.image_tag}"
            # Separate by ISL/OSL; if not provided, generate one figure per combo
            if args.input_length is not None and args.output_length is not None:
                title = f"{base_title} | ISL={args.input_length}, OSL={args.output_length}"
                out_file = args.output
                if out_file and not os.path.isabs(out_file):
                    os.makedirs(args.output_dir, exist_ok=True)
                    out_file = os.path.join(args.output_dir, out_file)
                latencies = [x.strip() for x in args.latencies.split(',') if x.strip()]
                plotter.plot_config_tp_comparison(filtered_data, title_prefix=title, output_file=out_file,
                                                  throughput=args.throughput, latencies=latencies)
                plt.show()
            else:
                # Multi-figure: one per ISL/OSL combination
                combos = filtered_data[['input_length','output_length']].drop_duplicates().sort_values(['input_length','output_length'])
                os.makedirs(args.output_dir, exist_ok=True)
                for _, comb in combos.iterrows():
                    isl = int(comb['input_length'])
                    osl = int(comb['output_length'])
                    dfc = filtered_data[(filtered_data['input_length']==isl)&(filtered_data['output_length']==osl)].copy()
                    if dfc.empty:
                        continue
                    title = f"{base_title} | ISL={isl}, OSL={osl}"
                    # Construct output name
                    if args.output:
                        name, ext = os.path.splitext(args.output)
                        out_file = os.path.join(args.output_dir, f"{name}_isl{isl}_osl{osl}{ext or '.png'}")
                    else:
                        safe_model = Path(str(args.model)).name
                        out_file = os.path.join(args.output_dir, f"{safe_model}_{args.model_config}_isl{isl}_osl{osl}_tp_compare.png")
                    latencies = [x.strip() for x in args.latencies.split(',') if x.strip()]
                    plotter.plot_config_tp_comparison(dfc, title_prefix=title, output_file=out_file,
                                                      throughput=args.throughput, latencies=latencies)
                plt.show()
        elif args.compare_configs:
            # Require model and tp-size; do not require single image_tag/model_config to allow comparison
            if not args.model or args.tp_size is None:
                print("ERROR: --compare-configs requires --model and --tp-size")
                sys.exit(1)
            print("Filtering data for Config/Tag comparison:")
            print(f"  Model: {args.model}")
            print(f"  TP size: {args.tp_size}")

            # Load and filter only by model and tp_size; optional ISL/OSL and request_rate
            df_all = loader.data
            dfm = df_all[(df_all['model'] == args.model) & (df_all['tp_size'] == args.tp_size)].copy()
            if args.input_length is not None:
                dfm = dfm[dfm['input_length'] == args.input_length]
            if args.output_length is not None:
                dfm = dfm[dfm['output_length'] == args.output_length]
            if args.request_rate is not None:
                dfm = dfm[dfm['request_rate'] == args.request_rate]

            if dfm.empty:
                print("ERROR: No data matches the specified criteria for compare-configs!")
                sys.exit(1)

            plotter = BenchmarkPlotter()
            base_title = f"{args.model} | TP={args.tp_size}"
            # If ISL/OSL not fixed, create one figure per combo
            if args.input_length is not None and args.output_length is not None:
                title = f"{base_title} | ISL={args.input_length}, OSL={args.output_length}"
                out_file = args.output
                if out_file and not os.path.isabs(out_file):
                    os.makedirs(args.output_dir, exist_ok=True)
                    out_file = os.path.join(args.output_dir, out_file)
                latencies = [x.strip() for x in args.latencies.split(',') if x.strip()]
                plotter.plot_tp_config_comparison(dfm, title_prefix=title, output_file=out_file,
                                                  throughput=args.throughput, latencies=latencies)
                plt.show()
            else:
                combos = dfm[['input_length','output_length']].drop_duplicates().sort_values(['input_length','output_length'])
                os.makedirs(args.output_dir, exist_ok=True)
                for _, comb in combos.iterrows():
                    isl = int(comb['input_length'])
                    osl = int(comb['output_length'])
                    dfc = dfm[(dfm['input_length']==isl)&(dfm['output_length']==osl)].copy()
                    if dfc.empty:
                        continue
                    title = f"{base_title} | ISL={isl}, OSL={osl}"
                    if args.output:
                        name, ext = os.path.splitext(args.output)
                        out_file = os.path.join(args.output_dir, f"{name}_isl{isl}_osl{osl}{ext or '.png'}")
                    else:
                        safe_model = Path(str(args.model)).name
                        out_file = os.path.join(args.output_dir, f"{safe_model}_tp{args.tp_size}_isl{isl}_osl{osl}_config_compare.png")
                    latencies = [x.strip() for x in args.latencies.split(',') if x.strip()]
                    plotter.plot_tp_config_comparison(dfc, title_prefix=title, output_file=out_file,
                                                      throughput=args.throughput, latencies=latencies)
                plt.show()
        else:
            print(f"Filtering data for:")
            print(f"  Model: {args.model}")
            print(f"  Image tag: {args.image_tag}")
            print(f"  Model config: {args.model_config}")

            filtered_data = loader.filter_data(
                args.model, args.image_tag, args.model_config,
                tp_size=args.tp_size, request_rate=args.request_rate,
                input_length=args.input_length, output_length=args.output_length
            )

            if filtered_data.empty:
                print("ERROR: No data matches the specified criteria!")
                sys.exit(1)

            print(f"Found {len(filtered_data)} data points")

            plotter = BenchmarkPlotter()
            title = f"{args.model} | {args.model_config} | {args.image_tag}"

            out_file = args.output
            if out_file and not os.path.isabs(out_file):
                out_file = os.path.join(args.output_dir, out_file)
                os.makedirs(args.output_dir, exist_ok=True)

            fig = plotter.plot_throughput_vs_latency(
                filtered_data,
                title_prefix=title,
                output_file=out_file
            )

            plt.show()

def generate_all_charts_for_model(loader: BenchmarkDataLoader, model: str, output_dir: str):
    """Generate 4×(TP sizes + configs) charts for a given model.

    - For each tp_size, compare across configs/image tags (4 charts)
    - For each config, compare across tp_sizes (4 charts)
    """
    df_all = loader.data
    if df_all is None or df_all.empty:
        raise ValueError("No data loaded")
    df_model = df_all[df_all['model'] == model] if 'model' in df_all.columns else df_all

    # Prepare sets
    tp_sizes = sorted(set(df_model['tp_size'])) if 'tp_size' in df_model.columns else []
    configs = sorted(set(df_model['model_config'])) if 'model_config' in df_model.columns else []

    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    plotter = BenchmarkPlotter()
    metrics = ['e2e', 'itl', 'ttft', 'e2e_per_user']

    # Per TP size: compare across configs (and tags)
    for tp in tp_sizes:
        df_tp = df_model[df_model['tp_size'] == tp]
        label_groups = {}
        if 'model_config' in df_tp.columns and 'image_tag' in df_tp.columns:
            for (mc, tag), g in df_tp.groupby(['model_config', 'image_tag']):
                label_groups[f"{mc} | {tag}"] = g
        elif 'model_config' in df_tp.columns:
            for mc, g in df_tp.groupby(['model_config']):
                label_groups[str(mc)] = g
        else:
            label_groups[f"TP={tp}"] = df_tp

        for metric in metrics:
            out_file = out_base / f"{Path(str(model)).name}_tp{tp}_{metric}.png"
            plotter.plot_comparison(label_groups, metric=metric, output_file=str(out_file))

    # Per config: compare across TP sizes
    for mc in configs:
        df_mc = df_model[df_model['model_config'] == mc]
        label_groups = {}
        if 'tp_size' in df_mc.columns:
            for tp, g in df_mc.groupby(['tp_size']):
                label_groups[f"TP={tp}"] = g
        else:
            label_groups[mc] = df_mc

        for metric in metrics:
            out_file = out_base / f"{Path(str(model)).name}_{mc}_{metric}.png"
            plotter.plot_comparison(label_groups, metric=metric, output_file=str(out_file))


if __name__ == "__main__":
    main()
