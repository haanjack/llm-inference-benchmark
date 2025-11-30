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
import os
import re
import sys
from pathlib import Path


def prettify_script(input_path: Path, output_path: Path = None):
    """Prettify a benchmark script for sharing."""

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Read the original script
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Early fix: ensure critical line breaks before other substitutions alter whitespace
    # 1) Split INT4 and $IMAGE onto separate lines within docker run
        content = re.sub(
            r'(VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4)\s*\\\s+\$IMAGE\s*\\',
            r'\1 \\\n    $IMAGE \\\\',
            content
        )
    # Also handle literal image case (commonly used vLLM image)
        content = re.sub(
            r'(VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4)\s*\\\s+' + re.escape('docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103') + r'\s*\\',
            r'\1 \\\n    docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103 \\\\',
            content
        )
    # 1) Ensure $IMAGE starts on a new line regardless of preceding tokens
    content = re.sub(
        r'(\\)\s+(\$IMAGE)\s*(\\)',
        r'\1\n    \2 \3',
        content
    )

    # 2) Separate any line with trailing whitespace + 'done' into two lines
    # This handles genai-perf (--warmup-request-count), sglang (--num-prompt), vllm (--percentile-metrics), etc.
    # Preserve the proper indentation by detecting it from context
    # Preserve the proper indentation by reducing by one level (4 spaces)
    content = re.sub(
        r'^(\s*)(\S(?:(?!done).)*\S)\s{2,}done$',
        lambda m: m.group(1) + m.group(2) + '\n' + ' ' * max(0, len(m.group(1)) - 4) + 'done',
        content,
        flags=re.MULTILINE
    )

    # Extract variable values from the header
    model_path_match = re.search(r'^MODEL_PATH="([^"]+)"', content, re.MULTILINE)
    image_match = re.search(r'^IMAGE="([^"]+)"', content, re.MULTILINE)
    container_name_match = re.search(r'^CONTAINER_NAME="([^"]+)"', content, re.MULTILINE)
    port_match = re.search(r'^PORT="([^"]+)"', content, re.MULTILINE)
    tp_size_match = re.search(r'^TP_SIZE="([^"]+)"', content, re.MULTILINE)

    # Detect user's home directory from the script
    user_home = os.path.expanduser('~')

    if not all([model_path_match, image_match, container_name_match, port_match, tp_size_match]):
        print("Warning: Could not extract all variables from script", file=sys.stderr)

    model_path = model_path_match.group(1) if model_path_match else None

    # Extract MODEL_DIR and MODEL_NAME from MODEL_PATH
    model_dir = None
    model_name = None
    if model_path:
        # Split into directory and model name (last two components)
        parts = model_path.strip('/').split('/')
        if len(parts) >= 2:
            model_dir = '/' + parts[0]  # e.g., /models
            model_name = '/'.join(parts[1:])  # e.g., amd/Llama-3.1-8B-Instruct-FP8-KV
        else:
            model_dir = '/models'
            model_name = model_path.strip('/')

    image = image_match.group(1) if image_match else None
    container_name = container_name_match.group(1) if container_name_match else None
    port = port_match.group(1) if port_match else None
    tp_size = tp_size_match.group(1) if tp_size_match else None

    # Extract loop values to convert into variables
    loop_values = {}
    loop_patterns = {
        'DATASET_NAMES': r'for DATASET_NAME in ([^;]+);',
        'NUM_PROMPTS_LIST': r'for NUM_PROMPTS in ([^;]+);',
        'REQUEST_RATES': r'for REQUEST_RATE in ([^;]+);',
        'INPUT_LENGTHS': r'for INPUT_LENGTH in ([^;]+);',
        'OUTPUT_LENGTHS': r'for OUTPUT_LENGTH in ([^;]+);',
        'CONCURRENCY_LEVELS': r'for CONCURRENCY in ([^;]+);',
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
        # Ensure the line break before $IMAGE is correct - fix any malformed line breaks
        # Match: INT4 \ (spaces) $IMAGE \
        content = re.sub(
            r'(VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4)\s*\\\s+\$IMAGE\s*\\',
            r'\1 \\\n    $IMAGE \\',
            content
        )
        # Also handle case where IMAGE is literal
        content = re.sub(
            r'(VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4)\s*\\\s+' + re.escape(image) + r'\s*\\',
            r'\1 \\\n    ' + image + r' \\',
            content
        )

    if container_name:
        # Replace container name in docker run --name
        content = re.sub(
            r'--name ' + re.escape(container_name),
            '--name $CONTAINER_NAME',
            content
        )

    if model_path:
        # Replace MODEL_PATH variable with MODEL_DIR and MODEL_NAME
        content = re.sub(
            r'^MODEL_PATH="[^"]+"\n',
            f'MODEL_DIR="{model_dir}"\nMODEL_NAME="{model_name}"\n',
            content,
            flags=re.MULTILINE
        )

        # Replace model path references in commands with $MODEL_DIR/$MODEL_NAME
        # Handle both literal paths and $MODEL_PATH variable references
        content = re.sub(
            r'serve \\\n    ' + re.escape(model_path),
            r'serve \\\n    $MODEL_DIR/$MODEL_NAME',
            content
        )
        content = re.sub(
            r'serve \\\n    \$MODEL_PATH',
            r'serve \\\n    $MODEL_DIR/$MODEL_NAME',
            content
        )

        # Replace volume mounts to use $MODEL_DIR/$MODEL_NAME
        # Handle various volume mount patterns
        content = re.sub(
            r'-v\s+\$HOME/[^\s]+:' + re.escape(model_path) + r':ro',
            r'-v $MODEL_DIR/$MODEL_NAME:$MODEL_DIR/$MODEL_NAME:ro',
            content
        )
        content = re.sub(
            r'-v\s+\$HOME/\$MODEL_PATH:\$MODEL_PATH:ro',
            r'-v $MODEL_DIR/$MODEL_NAME:$MODEL_DIR/$MODEL_NAME:ro',
            content
        )
        content = re.sub(
            r'-v\s+\$HOME/[^\s:]+:\$MODEL_PATH(\s)',
            r'-v $MODEL_DIR/$MODEL_NAME:$MODEL_DIR/$MODEL_NAME\1',
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
            # Skip if the value is already a variable reference (starts with $)
            if values.startswith('$'):
                continue
            # Handle quoted strings in values (e.g., "random" becomes 'random')
            clean_values = values.strip('"')
            loop_vars_section += f'{var_name}="{clean_values}"\n'
        # If no variables were added, don't include the section
        if loop_vars_section == "\n# Test Parameters\n":
            loop_vars_section = ""

    # Add IMAGE_TAG derivation logic
    image_tag_section = "\n# Derive tag (supports tag, digest, or none)\n"
    image_tag_section += 'if [[ "$IMAGE" == *"@"* ]]; then\n'
    image_tag_section += '    IMAGE_TAG="${IMAGE##*@}"\n'
    image_tag_section += 'elif [[ "$IMAGE" == *":"* ]]; then\n'
    image_tag_section += '    IMAGE_TAG="${IMAGE##*:}"\n'
    image_tag_section += 'else\n'
    image_tag_section += '    IMAGE_TAG="latest"\n'
    image_tag_section += 'fi\n'

    # Replace loop value literals with variable references BEFORE inserting the variables section
    for var_name, values in loop_values.items():
        loop_var_mapping = {
            'DATASET_NAMES': 'DATASET_NAME',
            'NUM_PROMPTS_LIST': 'NUM_PROMPTS',
            'REQUEST_RATES': 'REQUEST_RATE',
            'INPUT_LENGTHS': 'INPUT_LENGTH',
            'OUTPUT_LENGTHS': 'OUTPUT_LENGTH',
            'CONCURRENCY_LEVELS': 'CONCURRENCY',
        }
        loop_var = loop_var_mapping.get(var_name)
        if loop_var:
            # Replace "for VAR in values;" with "for VAR in $VARNAME;"
            content = re.sub(
                rf'for {loop_var} in {re.escape(values)};',
                f'for {loop_var} in ${var_name};',
                content
            )

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

    # Remove duplicated MODEL_NAME lines if present
    content = re.sub(r'^(MODEL_NAME="[^"]+"\n)(MODEL_DIR=)', r'\2', content, flags=re.MULTILINE)

    # Insert loop variable declarations after benchmark variables section
    if loop_vars_section or image_tag_section:
        # Find the end of benchmark variables (after CONTAINER_NAME)
        container_name_line = re.search(r'^CONTAINER_NAME="[^"]+"\n', content, re.MULTILINE)
        if container_name_line:
            insert_pos = container_name_line.end()
            insert_content = ""
            if loop_vars_section:
                insert_content += loop_vars_section
            insert_content += image_tag_section
            content = content[:insert_pos] + insert_content + "\n" + content[insert_pos:]

    # Update mkdir command to use MODEL_NAME and IMAGE_TAG in log path
    content = re.sub(
        r'mkdir -p \$\(pwd\)/logs/[^\n]+',
        r'mkdir -p $(pwd)/logs/$MODEL_NAME/$IMAGE_TAG/llama-vllm-tp1',
        content
    )

    # Update genai-perf volume mount for artifacts to use MODEL_NAME and IMAGE_TAG
    # Handle both $(pwd) and absolute $HOME paths
    content = re.sub(
        r'-v \$\(pwd\)/logs/[^:]+:/tmp/artifacts',
        r'-v $(pwd)/logs/$MODEL_NAME/$IMAGE_TAG/llama-vllm-tp1:/tmp/artifacts',
        content
    )
    content = re.sub(
        r'-v \$HOME/[^:]+/logs/[^:]+:/tmp/artifacts',
        r'-v $(pwd)/logs/$MODEL_NAME/$IMAGE_TAG/llama-vllm-tp1:/tmp/artifacts',
        content
    )

    # Update genai-perf model volume mount to use MODEL_DIR/$MODEL_NAME
    content = re.sub(
        r'-v \$HOME/[^:]+:\$MODEL_PATH(\s)',
        r'-v $MODEL_DIR/$MODEL_NAME:$MODEL_DIR/$MODEL_NAME\1',
        content
    )

    # Update genai-perf -m and --tokenizer arguments to use MODEL_DIR/$MODEL_NAME
    content = re.sub(
        r'(-m|\--tokenizer) \$MODEL_PATH',
        r'\1 $MODEL_DIR/$MODEL_NAME',
        content
    )

    # Update --url to use variable
    content = re.sub(
        r'--url 0\.0\.0\.0:\d+',
        r'--url 0.0.0.0:$PORT',
        content
    )

    # Fix excessive indentation in genai-perf docker command (28 spaces -> 24 spaces to match wanted)
    # This happens because of nested loop structure
    content = re.sub(
        r'(\n)                            (run \\|\--[a-z]|-v |nvcr\.io|genai-perf|profile|-m |\--tokenizer|\--endpoint|\--url|\--streaming|\--random|\--concurrency|\--request|\--synthetic|\--output-tokens|\--extra-inputs|\--profile-export|\--artifact-dir|\--warmup)',
        r'\1                        \2',
        content
    )
    # Convert from 20 spaces to wanted 24 spaces for lines within the docker client block
    content = re.sub(
        r'(\n)                        (run \\|\--[a-z]|-v |nvcr\.io|genai-perf|profile|-m |\--tokenizer|\--endpoint|\--url|\--streaming|\--random|\--concurrency|\--request|\--synthetic|\--output-tokens|\--extra-inputs|\--profile-export|\--artifact-dir|\--warmup)',
        r'\1                        \2',
        content
    )

    # Warmup/done fix already applied earlier

    # Remove REQUEST_RATE loop layer (wanted script has no request rate loop)
    content = re.sub(r'\n\s*for REQUEST_RATE in [^;]+; do\n', '\n', content)

    # Replace any literal model path in server 'serve' with variables
    if model_dir and model_name:
        content = re.sub(
            r'(\n\s+serve \\)\n\s+/[a-zA-Z0-9_/\.-]+',
            r"\1\n    $MODEL_DIR/$MODEL_NAME",
            content
        )

    # Remove dummy compile config file reference (if present during dry-run)
    # The actual --compilation-config JSON is already inlined by the server during generation
    content = re.sub(
        r"\s+--config\s+/root/\.cache/compile_config/[^\s]+\s*",
        " ",
        content
    )

    # Ensure artifacts directory exists prior to client run
    if re.search(r'-v \$\(pwd\)/logs/\$MODEL_NAME/\$IMAGE_TAG/llama-vllm-tp1:/tmp/artifacts', content):
        # Insert mkdir after server ready echo if not present
        content = re.sub(
            r'(echo "Server is ready! \(took \$\(\(TOTAL_TIME / 60\)\)m \$\(\(TOTAL_TIME % 60\)\)s\)"\n)',
            r"\1\nmkdir -p $(pwd)/logs/$MODEL_NAME/$IMAGE_TAG/llama-vllm-tp1\n",
            content
        )

    # Remove duplicate 'done' statements at the end of loops
    content = re.sub(
        r'(\s+done\n)(\s+done\n\s+done\n\s+done\n\s+done\n)',
        r'\2',
        content
    )

    # Remove empty line before podman exec in benchmark loop
    content = re.sub(r'(for DATASET_NAME in [^;]+; do)\n\n(\s+podman \\)', r'\1\n\2', content)

    # Final pass: enforce newline before $IMAGE regardless of earlier ordering
    content = re.sub(
        r'(\\)\s+(\$IMAGE)\s*(\\)',
        r'\1\n    \2 \3',
        content
    )
    # Final pass: ensure any line with trailing whitespace + 'done' are split
    content = re.sub(
        r'^(\s*)(\S(?:(?!done).)*\S)\s{2,}done$',
        lambda m: m.group(1) + m.group(2) + '\n' + ' ' * max(0, len(m.group(1)) - 4) + 'done',
        content,
        flags=re.MULTILINE
    )

    # Output to file or stdout
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
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
