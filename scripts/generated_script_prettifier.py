#!/usr/bin/env python3
"""
Prettify auto-generated benchmark scripts for sharing.

This script:
1. Replaces hardcoded values with variable references
2. Removes internal cache mounts
3. Adds helpful comments
4. Makes the script more user-friendly for customers/colleagues
"""

import argparse
import re
import sys
from pathlib import Path


def prettify_script(input_path: Path, output_path: Path = None):
    """Prettify a benchmark script for sharing."""

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Read the original script
    with open(input_path, 'r') as f:
        content = f.read()

    # Extract variable values from the header
    model_path_match = re.search(r'^MODEL_PATH="([^"]+)"', content, re.MULTILINE)
    image_match = re.search(r'^IMAGE="([^"]+)"', content, re.MULTILINE)
    container_name_match = re.search(r'^CONTAINER_NAME="([^"]+)"', content, re.MULTILINE)
    port_match = re.search(r'^PORT="([^"]+)"', content, re.MULTILINE)
    tp_size_match = re.search(r'^TP_SIZE="([^"]+)"', content, re.MULTILINE)

    # Detect user's home directory from the script
    import os
    user_home = os.path.expanduser('~')

    if not all([model_path_match, image_match, container_name_match, port_match, tp_size_match]):
        print("Warning: Could not extract all variables from script", file=sys.stderr)

    model_path = model_path_match.group(1) if model_path_match else None
    image = image_match.group(1) if image_match else None
    container_name = container_name_match.group(1) if container_name_match else None
    port = port_match.group(1) if port_match else None
    tp_size = tp_size_match.group(1) if tp_size_match else None

    # Extract loop values to convert into variables
    loop_values = {}
    loop_patterns = {
        'REQUEST_RATES': r'for REQUEST_RATE in ([^;]+);',
        'NUM_PROMPTS_LIST': r'for NUM_PROMPTS in ([^;]+);',
        'INPUT_LENGTHS': r'for INPUT_LENGTH in ([^;]+);',
        'OUTPUT_LENGTHS': r'for OUTPUT_LENGTH in ([^;]+);',
        'CONCURRENCY_LEVELS': r'for CONCURRENCY in ([^;]+);',
        'DATASET_NAMES': r'for DATASET_NAME in ([^;]+);',
    }

    for var_name, pattern in loop_patterns.items():
        match = re.search(pattern, content)
        if match:
            loop_values[var_name] = match.group(1).strip()

    # Replace hardcoded home directory with $HOME variable
    if user_home and user_home in content:
        content = content.replace(user_home, '$HOME')

    # Replace hardcoded values with variable references
    if image:
        # Replace image in docker run command
        content = content.replace(f'\n    {image} \\', '\n    $IMAGE \\')

    if container_name:
        # Replace container name in docker run --name
        content = re.sub(
            r'--name ' + re.escape(container_name),
            '--name $CONTAINER_NAME',
            content
        )

    if model_path:
        # Replace model path in docker run command (but keep volume mount with actual path)
        content = re.sub(
            r'serve \\\n    ' + re.escape(model_path),
            'serve \\\n    $MODEL_PATH',
            content
        )

    if port:
        # Replace port in --port argument
        content = re.sub(
            r'--port ' + port + r'(?!\d)',
            '--port $PORT',
            content
        )
        # Replace port in nc -z localhost check
        content = re.sub(
            r'nc -z localhost ' + port,
            'nc -z localhost $PORT',
            content
        )
        # Replace port in echo message (use double quotes for variable expansion)
        content = re.sub(
            r"echo 'Waiting for server on port " + port + r"\.\.\.\'",
            'echo "Waiting for server on port $PORT..."',
            content
        )
        # Replace port in --port for client
        content = re.sub(
            r'--port ' + port,
            '--port $PORT',
            content
        )

    if tp_size:
        # Replace tensor-parallel-size
        content = re.sub(
            r'--tensor-parallel-size ' + tp_size,
            '--tensor-parallel-size $TP_SIZE',
            content
        )

    # Remove cache-related volume mounts (these are internal details)
    cache_patterns = [
        r'\s+-v [^\s]+/vllm_cache[^\s]+:/root/\.cache[^\s]* \\\n',
        r'\s+-v [^\s]+/vllm_cache[^\s]+:/root/\.cache/compile_config[^\s]* \\\n',
        r'\s+-v [^\s]+/vllm_cache[^\s]+:/root/\.aiter[^\s]* \\\n',
    ]

    for pattern in cache_patterns:
        content = re.sub(pattern, '', content)

    # Build header with loop value variables
    loop_vars_section = ""
    if loop_values:
        loop_vars_section = "\n# Test Parameters\n"
        for var_name, values in loop_values.items():
            # Handle quoted strings in values (e.g., "random" becomes 'random')
            clean_values = values.strip('"')
            if ' ' in clean_values:
                loop_vars_section += f'{var_name}="{clean_values}"\n'
            else:
                loop_vars_section += f'{var_name}="{clean_values}"\n'

    # Add helpful header comment
    header_comment = f"""#!/bin/bash
# Auto-generated benchmark script (prettified for sharing)
#
# This script runs vLLM benchmarks with the specified configuration.
# To customize, edit the variables below before running.
#
# Usage: ./run-llama-vllm-test.sh
#

"""

    # Replace the original header
    content = re.sub(r'^#!/bin/bash\n# Auto-generated benchmark script\n\n', header_comment, content)

    # Insert loop variable declarations after benchmark variables section
    if loop_vars_section:
        # Find the end of benchmark variables (after CONTAINER_NAME)
        container_name_line = re.search(r'^CONTAINER_NAME="[^"]+"\n', content, re.MULTILINE)
        if container_name_line:
            insert_pos = container_name_line.end()
            content = content[:insert_pos] + loop_vars_section + "\n" + content[insert_pos:]

    # Replace loop value literals with variable references
    for var_name, values in loop_values.items():
        loop_var_mapping = {
            'REQUEST_RATES': 'REQUEST_RATE',
            'NUM_PROMPTS_LIST': 'NUM_PROMPTS',
            'INPUT_LENGTHS': 'INPUT_LENGTH',
            'OUTPUT_LENGTHS': 'OUTPUT_LENGTH',
            'CONCURRENCY_LEVELS': 'CONCURRENCY',
            'DATASET_NAMES': 'DATASET_NAME',
        }
        loop_var = loop_var_mapping.get(var_name)
        if loop_var:
            # Replace "for VAR in values;" with "for VAR in $VARNAME;"
            content = re.sub(
                rf'for {loop_var} in {re.escape(values)};',
                f'for {loop_var} in ${var_name};',
                content
            )

    # Remove empty line before podman exec in benchmark loop
    content = re.sub(r'(for DATASET_NAME in [^;]+; do)\n\n(\s+podman \\)', r'\1\n\2', content)

    # Output to file or stdout
    if output_path:
        with open(output_path, 'w') as f:
            f.write(content)
        output_path.chmod(0o755)
        print(f"Prettified script saved to: {output_path}")
    else:
        print(content)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Prettify auto-generated benchmark scripts for sharing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Prettify and save with -prettified suffix
        %(prog)s tests/generated/run-llama-vllm-test.sh

        # Specify output file
        %(prog)s tests/generated/run-llama-vllm-test.sh -o share/benchmark.sh

        # Print to stdout
        %(prog)s tests/generated/run-llama-vllm-test.sh --stdout
                """
    )

    parser.add_argument(
        '-i', '--input',
        type=Path,
        help='Input benchmark script to prettify'
    )

    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file path (default: adds -prettified suffix to input)'
    )

    parser.add_argument(
        '--stdout',
        action='store_true',
        help='Print result to stdout instead of saving to file'
    )

    args = parser.parse_args()

    # Determine output path
    if args.stdout:
        output_path = None
    elif args.output:
        output_path = args.output
    else:
        # Add -prettified suffix
        stem = args.input.stem
        output_path = args.input.parent / f"{stem}-prettified{args.input.suffix}"

    return prettify_script(args.input, output_path)


if __name__ == '__main__':
    sys.exit(main())
