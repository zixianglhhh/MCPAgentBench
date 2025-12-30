#!/usr/bin/env python3
"""
Generate LaTeX tables summarizing TFS and TEFS avg@4 scores
across Daily and Professional task categories.

The tables follow the layout requested by the user and can be
embedded directly inside Markdown documents via fenced LaTeX blocks.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluate import load_skip_input_tools  # type: ignore
from src.utilities import flatten  # type: ignore

# Reuse evaluation helpers from plotting scripts ---------------------------


def normalize_dict(d):
    return {k: str(v) for k, v in d.items()}


def is_finished_task(expected_tools, expected_inputs, tools_used, inputs_used, skip_input_tools):
    expected_tools_flat = []
    expected_inputs_flat = []
    for step_tools, step_inputs in zip(expected_tools, expected_inputs):
        for tool, inp in zip(step_tools, step_inputs):
            expected_tools_flat.append(tool)
            expected_inputs_flat.append((tool, inp))

    actual_tools_flat = []
    actual_inputs_flat = []
    for step_tools, step_inputs in zip(tools_used, inputs_used):
        for tool, inp in zip(step_tools, step_inputs):
            actual_tools_flat.append(tool)
            actual_inputs_flat.append((tool, inp))

    if set(expected_tools_flat) != set(actual_tools_flat):
        return False

    filtered_expected_inputs = [
        (tool_name, input_data)
        for tool_name, input_data in expected_inputs_flat
        if tool_name not in skip_input_tools
    ]
    filtered_actual_inputs = [
        (tool_name, input_data)
        for tool_name, input_data in actual_inputs_flat
        if tool_name not in skip_input_tools
    ]

    if filtered_expected_inputs or filtered_actual_inputs:
        expected_inputs_set = {
            (tool_name, frozenset(normalize_dict(input_data).items()))
            for tool_name, input_data in filtered_expected_inputs
        }
        actual_inputs_set = {
            (tool_name, frozenset(normalize_dict(input_data).items()))
            for tool_name, input_data in filtered_actual_inputs
        }
        if expected_inputs_set != actual_inputs_set:
            return False

    return True


def is_efficiently_finished_task(expected_tools, expected_inputs, tools_used, inputs_used, skip_input_tools):
    if not is_finished_task(expected_tools, expected_inputs, tools_used, inputs_used, skip_input_tools):
        return False

    if len(tools_used) != len(expected_tools):
        return False

    for i, (expected_step_tools, actual_step_tools) in enumerate(zip(expected_tools, tools_used)):
        if set(expected_step_tools) != set(actual_step_tools):
            return False

        expected_step_inputs = expected_inputs[i] if i < len(expected_inputs) else []
        actual_step_inputs = inputs_used[i] if i < len(inputs_used) else []

        expected_map = {}
        for j, tool_name in enumerate(expected_step_tools):
            if tool_name in skip_input_tools:
                continue
            if j < len(expected_step_inputs):
                expected_map[tool_name] = expected_step_inputs[j]

        actual_map = {}
        for j, tool_name in enumerate(actual_step_tools):
            if tool_name in skip_input_tools:
                continue
            if j < len(actual_step_inputs):
                actual_map[tool_name] = actual_step_inputs[j]

        for tool_name in expected_step_tools:
            if tool_name in skip_input_tools:
                continue
            expected_input = expected_map.get(tool_name, {})
            actual_input = actual_map.get(tool_name, {})
            if normalize_dict(expected_input) != normalize_dict(actual_input):
                return False

    return True


def calculate_tfs_for_task(expected_tools, expected_inputs, tools_used, inputs_used, skip_input_tools):
    num_tools = len(list(flatten(expected_tools)))
    if num_tools == 0:
        return 0.0
    return float(num_tools) if is_finished_task(
        expected_tools, expected_inputs, tools_used, inputs_used, skip_input_tools
    ) else 0.0


def calculate_tefs_for_task(expected_tools, expected_inputs, tools_used, inputs_used, skip_input_tools):
    num_tools = len(list(flatten(expected_tools)))
    if num_tools == 0:
        return 0.0
    return float(num_tools) if is_efficiently_finished_task(
        expected_tools, expected_inputs, tools_used, inputs_used, skip_input_tools
    ) else 0.0


# Result loading helpers ---------------------------------------------------


def extract_model_name(filename: str) -> str:
    base_name = Path(filename).name
    marker = "_general_test_run"
    marker_idx = base_name.find(marker)

    if marker_idx == -1:
        return base_name.split("_")[0]

    prefix = base_name[:marker_idx]
    prev_underscore_idx = prefix.rfind("_")

    if prev_underscore_idx == -1:
        return prefix
    return prefix[prev_underscore_idx + 1:]


def load_results_for_model(model_name: str, results_dir: str) -> List[Dict]:
    results_path = Path(results_dir)
    pattern = re.compile(r".+_general_test_run([1-4])_results\.json$")

    run_files = []
    for file_path in results_path.glob("*.json"):
        match = pattern.match(file_path.name)
        if match and extract_model_name(file_path.name) == model_name:
            run_num = int(match.group(1))
            run_files.append((run_num, file_path))

    run_files.sort(key=lambda x: x[0])

    results = []
    for _, file_path in run_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
    return results


def find_all_models(results_dir: str) -> List[str]:
    models = set()
    pattern = re.compile(r".+_general_test_run[1-4]_results\.json$")
    for file_path in Path(results_dir).glob("*.json"):
        if pattern.match(file_path.name):
            models.add(extract_model_name(file_path.name))
    return sorted(models)


# Category helpers ---------------------------------------------------------

CATEGORY_ORDER = [
    ("daily", "single"),
    ("daily", "dual_serial"),
    ("daily", "dual_parallel"),
    ("daily", "multi"),
    ("professional", "single"),
    ("professional", "dual_serial"),
    ("professional", "dual_parallel"),
    ("professional", "multi"),
]


def categorize_task(task_id: str) -> Optional[Tuple[str, str]]:
    if not task_id:
        return None
    task_id_lower = task_id.lower()

    if task_id_lower.startswith("daytask"):
        group = "daily"
    elif task_id_lower.startswith("protask"):
        group = "professional"
    else:
        return None

    if "_1_tool_" in task_id_lower:
        sub = "single"
    elif "_2_sequential" in task_id_lower:
        sub = "dual_serial"
    elif "_2_parallel" in task_id_lower:
        sub = "dual_parallel"
    elif "_3_tools" in task_id_lower:
        sub = "multi"
    else:
        return None

    return (group, sub)


# Metric computation -------------------------------------------------------


def compute_category_scores(model_results: List[Dict], metric: str, skip_input_tools) -> Dict[Tuple[str, str], float]:
    if not model_results:
        return {}

    task_results_by_id = defaultdict(list)
    for result_data in model_results:
        detailed_results = result_data.get("detailed_results", [])
        for task_result in detailed_results:
            task_id = task_result.get("task_id")
            if task_id:
                task_results_by_id[task_id].append(task_result)

    category_weight = defaultdict(float)
    category_score = defaultdict(float)

    for task_id, task_runs in task_results_by_id.items():
        category = categorize_task(task_id)
        if category is None:
            continue

        expected_tools = task_runs[0].get("expected_tools", [])
        weight = float(len(list(flatten(expected_tools))))
        if weight == 0:
            continue

        normalized_scores = []
        for task_result in task_runs:
            expected_tools = task_result.get("expected_tools", [])
            expected_inputs = task_result.get("expected_inputs", [])
            tools_used = task_result.get("tools_used", [])
            inputs_used = task_result.get("inputs_used", [])

            if metric == "tfs":
                score = calculate_tfs_for_task(
                    expected_tools, expected_inputs, tools_used, inputs_used, skip_input_tools
                )
            else:
                score = calculate_tefs_for_task(
                    expected_tools, expected_inputs, tools_used, inputs_used, skip_input_tools
                )
            normalized_scores.append(score / weight if weight > 0 else 0.0)

        avg_normalized = sum(normalized_scores) / len(normalized_scores)
        category_weight[category] += weight
        category_score[category] += avg_normalized * weight

    category_percentages = {}
    for category in CATEGORY_ORDER:
        total_weight = category_weight.get(category, 0.0)
        if total_weight > 0:
            category_percentages[category] = (category_score[category] / total_weight) * 100
        else:
            category_percentages[category] = 0.0

    return category_percentages


# LaTeX rendering ----------------------------------------------------------


def format_latex_table(title: str, label: str, model_scores: Dict[str, Dict[Tuple[str, str], float]]) -> str:
    model_names = sorted(model_scores.keys())
    averages = {cat: [] for cat in CATEGORY_ORDER}
    for model in model_names:
        for cat in CATEGORY_ORDER:
            averages[cat].append(model_scores[model].get(cat, 0.0))

    avg_row = {cat: (sum(vals) / len(vals) if vals else 0.0) for cat, vals in averages.items()}

    def fmt(value: float) -> str:
        return f"{value:5.2f}"

    header_daily = " & ".join(["Single", "Dual Serial", "Dual Parallel", "Multi"])
    header_pro = header_daily

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append("")
    lines.append(r"    \centering")
    lines.append(r"    \setlength{\tabcolsep}{6pt}")
    lines.append("")
    lines.append(f"    \\caption{{{title}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append("")
    lines.append(r"    \begin{tabular*}{0.95\textwidth}{@{\extracolsep{\fill}}lcccccccc}")
    lines.append(r"        \toprule")
    lines.append(r"        \multirow{2}{*}{\textbf{Models}} & \multicolumn{4}{c}{\textbf{Daily}} & \multicolumn{4}{c}{\textbf{Professional}} \\")
    lines.append(r"        \cmidrule(lr){2-5}\cmidrule(lr){6-9}")
    lines.append(r"         & " + header_daily + " & " + header_pro + r" \\")
    lines.append(r"        \midrule")

    for model in model_names:
        values = [fmt(model_scores[model].get(cat, 0.0)) for cat in CATEGORY_ORDER]
        row = r"        \textbf{" + model.replace("_", r"\_") + "} & " + " & ".join(values) + r" \\"
        lines.append(row)

    avg_values = [fmt(avg_row[cat]) for cat in CATEGORY_ORDER]
    lines.append(r"        \midrule")
    lines.append(r"        \textbf{Average} & " + " & ".join(avg_values) + r" \\")
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular*}")
    lines.append("")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# Main ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for TFS and TEFS category scores.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing result files.")
    args = parser.parse_args()

    results_dir = args.results_dir
    skip_input_tools = load_skip_input_tools()
    models = find_all_models(results_dir)

    if not models:
        print("No models found with required run files.")
        return

    tfs_scores = {}
    tefs_scores = {}

    for model in models:
        results = load_results_for_model(model, results_dir)
        if len(results) != 4:
            print(f"Warning: {model} has {len(results)} runs (expected 4). Skipping.")
            continue
        tfs_scores[model] = compute_category_scores(results, "tfs", skip_input_tools)
        tefs_scores[model] = compute_category_scores(results, "tefs", skip_input_tools)

    if not tfs_scores or not tefs_scores:
        print("No valid models with complete runs to generate tables.")
        return

    print("% --- TFS Category Table ---")
    print(format_latex_table("TFS avg@4 by Task Category", "tab:tfs_category", tfs_scores))
    print("\n% --- TEFS Category Table ---")
    print(format_latex_table("TEFS avg@4 by Task Category", "tab:tefs_category", tefs_scores))


if __name__ == "__main__":
    main()

