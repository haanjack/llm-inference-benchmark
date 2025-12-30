#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for LLM Benchmark Plotter
- Choose model, config/tag or TP size
- Select throughput metric and latency set
- Strict ISL/OSL separation
- Save charts as PNG files
"""
import sys
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

# Ensure local scripts directory is importable
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from plot_benchmark_results import (
    BenchmarkDataLoader,
    BenchmarkPlotter,
)

THROUGHPUT_CHOICES = [
    'tokens_per_sec_per_gpu',
    'tokens_per_sec',
    'output_tokens_per_sec_per_gpu',
    'output_tokens_per_sec'
]
LATENCY_CHOICES = ['e2e', 'itl', 'ttft', 'interactivity']

st.set_page_config(page_title="LLM Benchmark Plotter", layout="wide")

# Paths
REPO_ROOT = THIS_DIR.parent.parent
LOGS_DIR = REPO_ROOT / 'logs'
PLOTS_DIR = REPO_ROOT / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
loader = BenchmarkDataLoader(str(LOGS_DIR))
try:
    df = loader.load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

st.sidebar.header("Filters")
models = sorted(set(df['model']))
model = st.sidebar.selectbox("Model", models)

df_m = df[df['model'] == model]
tags = sorted(set(df_m['image_tag']))
configs = sorted(set(df_m['model_config']))
tp_sizes = sorted(set(df_m['tp_size']))
isl_vals = sorted(set(df_m.get('input_length', [])))
osl_vals = sorted(set(df_m.get('output_length', [])))

mode = st.sidebar.radio("Comparison Mode", [
    "Same config: compare TP sizes",
    "Same TP: compare configs/tags"
])
throughput = st.sidebar.selectbox("Throughput", THROUGHPUT_CHOICES, index=0)
latencies = st.sidebar.multiselect("Latencies", LATENCY_CHOICES, default=LATENCY_CHOICES)

isl = st.sidebar.selectbox("ISL (optional)", [None] + isl_vals, format_func=lambda x: "All" if x is None else x)
osl = st.sidebar.selectbox("OSL (optional)", [None] + osl_vals, format_func=lambda x: "All" if x is None else x)

st.sidebar.header("Save")
fname = st.sidebar.text_input("Filename", "plot.png")
out_dir_str = st.sidebar.text_input("Output dir", str(PLOTS_DIR))
out_dir = Path(out_dir_str)
out_dir.mkdir(parents=True, exist_ok=True)

run = st.sidebar.button("Generate")

st.title("LLM Benchmark Plotter")

if run:
    if mode.startswith("Same config"):
        tag = st.sidebar.selectbox("Image tag", tags)
        cfg = st.sidebar.selectbox("Model config", configs)
        filtered = loader.filter_data(model, tag, cfg)
        if filtered.empty:
            st.error("No data matches selection.")
        else:
            title = f"{model} | {cfg} | {tag}"
            out_file = out_dir / fname
            plotter = BenchmarkPlotter()
            # Split per ISL/OSL when not specified
            if isl is not None and osl is not None:
                plotter.plot_config_tp_comparison(filtered, title_prefix=f"{title} | ISL={isl}, OSL={osl}",
                                                  output_file=str(out_file), throughput=throughput, latencies=latencies)
            else:
                combos = filtered[['input_length','output_length']].drop_duplicates().sort_values(['input_length','output_length'])
                for _, comb in combos.iterrows():
                    i_isl = int(comb['input_length'])
                    i_osl = int(comb['output_length'])
                    dfc = filtered[(filtered['input_length']==i_isl)&(filtered['output_length']==i_osl)].copy()
                    if dfc.empty:
                        continue
                    name, ext = out_file.stem, out_file.suffix or '.png'
                    out_file_islosl = out_dir / f"{name}_isl{i_isl}_osl{i_osl}{ext}"
                    plotter.plot_config_tp_comparison(dfc, title_prefix=f"{title} | ISL={i_isl}, OSL={i_osl}",
                                                      output_file=str(out_file_islosl), throughput=throughput, latencies=latencies)
            st.success(f"Charts saved to {out_dir}")
    else:
        tp = st.sidebar.selectbox("TP size", tp_sizes)
        filtered = loader.data[(loader.data['model']==model) & (loader.data['tp_size']==tp)].copy()
        if isl is not None:
            filtered = filtered[filtered['input_length']==isl]
        if osl is not None:
            filtered = filtered[filtered['output_length']==osl]
        if filtered.empty:
            st.error("No data matches selection.")
        else:
            title = f"{model} | TP={tp}"
            out_file = out_dir / fname
            plotter = BenchmarkPlotter()
            if isl is not None and osl is not None:
                plotter.plot_tp_config_comparison(filtered, title_prefix=f"{title} | ISL={isl}, OSL={osl}",
                                                  output_file=str(out_file), throughput=throughput, latencies=latencies)
            else:
                combos = filtered[['input_length','output_length']].drop_duplicates().sort_values(['input_length','output_length'])
                for _, comb in combos.iterrows():
                    i_isl = int(comb['input_length'])
                    i_osl = int(comb['output_length'])
                    dfc = filtered[(filtered['input_length']==i_isl)&(filtered['output_length']==i_osl)].copy()
                    if dfc.empty:
                        continue
                    name, ext = out_file.stem, out_file.suffix or '.png'
                    out_file_islosl = out_dir / f"{name}_isl{i_isl}_osl{i_osl}{ext}"
                    plotter.plot_tp_config_comparison(dfc, title_prefix=f"{title} | ISL={i_isl}, OSL={i_osl}",
                                                      output_file=str(out_file_islosl), throughput=throughput, latencies=latencies)
            st.success(f"Charts saved to {out_dir}")

st.subheader("Data preview")
st.dataframe(df.head(500))
