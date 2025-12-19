#!/usr/bin/env python3
"""
Plot Task Efficiency Finish Score (TEFS) comparison chart.
Shows avg@4 for each model with task-category breakdown.

TEFS requires both:
1. Task is finished (tools and inputs match, ignoring order)
2. Tool invocation sequence exactly matches golden solution (including parallel/sequential order)
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any
import matplotlib.pyplot as plt
import numpy as np

# Import evaluation utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluate import load_skip_input_tools
from src.utilities import flatten, compare_inputs


def is_finished_task(expected_tools: List[List[str]], expected_inputs: List[List[Dict[str, Any]]],
                     tools_used: List[List[str]], inputs_used: List[List[Dict[str, Any]]],
                     skip_input_tools: Set[str]) -> bool:
    """
    Check if a task is finished (TFS definition).
    This is a prerequisite for TEFS.
    
    Args:
        expected_tools: Expected tools (nested list, each inner list is a step)
        expected_inputs: Expected inputs (nested list, each inner list is a step)
        tools_used: Actual tools used (nested list)
        inputs_used: Actual inputs used (nested list)
        skip_input_tools: Set of tool names to skip input comparison
        
    Returns:
        bool: True if task is finished, False otherwise
    """
    # Flatten expected tools and inputs (ignore parallel/sequential structure)
    expected_tools_flat = []
    expected_inputs_flat = []
    for step_tools, step_inputs in zip(expected_tools, expected_inputs):
        for tool, inp in zip(step_tools, step_inputs):
            expected_tools_flat.append(tool)
            expected_inputs_flat.append((tool, inp))
    
    # Flatten actual tools and inputs
    actual_tools_flat = []
    actual_inputs_flat = []
    for step_tools, step_inputs in zip(tools_used, inputs_used):
        for tool, inp in zip(step_tools, step_inputs):
            actual_tools_flat.append(tool)
            actual_inputs_flat.append((tool, inp))
    
    # Check if the same tools are used (ignoring order)
    if set(expected_tools_flat) != set(actual_tools_flat):
        return False
    
    # Check if inputs match (only for tools not in skip list)
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
    
    # Compare filtered inputs (only for tools not in skip list)
    if filtered_expected_inputs or filtered_actual_inputs:
        # Normalize inputs for comparison
        def normalize_dict(d):
            return {k: str(v) for k, v in d.items()}
        
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


def is_efficiently_finished_task(expected_tools: List[List[str]], expected_inputs: List[List[Dict[str, Any]]],
                                 tools_used: List[List[str]], inputs_used: List[List[Dict[str, Any]]],
                                 skip_input_tools: Set[str]) -> bool:
    """
    Check if a task is efficiently finished according to TEFS definition.
    
    A task is efficiently finished if:
    1. The task is finished (IsFinished = 1) - tools and inputs match
    2. The tool invocation sequence exactly matches the golden solution in serial and parallel execution order
    
    Args:
        expected_tools: Expected tools (nested list, each inner list is a step)
        expected_inputs: Expected inputs (nested list, each inner list is a step)
        tools_used: Actual tools used (nested list)
        inputs_used: Actual inputs used (nested list)
        skip_input_tools: Set of tool names to skip input comparison
        
    Returns:
        bool: True if task is efficiently finished, False otherwise
    """
    # First check if task is finished (prerequisite)
    if not is_finished_task(expected_tools, expected_inputs, tools_used, inputs_used, skip_input_tools):
        return False
    
    # Check if the execution order matches exactly (including parallel/sequential structure)
    # The nested list structure represents steps, where each inner list represents parallel tools
    
    # Check if number of steps matches
    if len(tools_used) != len(expected_tools):
        return False
    
    # Check each step
    for i, (expected_step_tools, actual_step_tools) in enumerate(zip(expected_tools, tools_used)):
        # Within each step, tools can be in any order (they are parallel)
        # But the set of tools in each step must match exactly
        if set(expected_step_tools) != set(actual_step_tools):
            return False
        
        # Check inputs for this step (only for tools not in skip list)
        expected_step_inputs = expected_inputs[i] if i < len(expected_inputs) else []
        actual_step_inputs = inputs_used[i] if i < len(inputs_used) else []
        
        # Create mapping from tool name to input for this step
        expected_tool_input_map = {}
        for j, tool_name in enumerate(expected_step_tools):
            if tool_name in skip_input_tools:
                continue
            if j < len(expected_step_inputs):
                expected_tool_input_map[tool_name] = expected_step_inputs[j]
        
        actual_tool_input_map = {}
        for j, tool_name in enumerate(actual_step_tools):
            if tool_name in skip_input_tools:
                continue
            if j < len(actual_step_inputs):
                actual_tool_input_map[tool_name] = actual_step_inputs[j]
        
        # Compare inputs for tools not in skip list
        for tool_name in expected_step_tools:
            if tool_name in skip_input_tools:
                continue
            expected_input = expected_tool_input_map.get(tool_name, {})
            actual_input = actual_tool_input_map.get(tool_name, {})
            
            # Normalize for comparison
            def normalize_dict(d):
                return {k: str(v) for k, v in d.items()}
            
            expected_normalized = normalize_dict(expected_input)
            actual_normalized = normalize_dict(actual_input)
            
            if expected_normalized != actual_normalized:
                return False
    
    return True


def calculate_tefs_for_task(expected_tools: List[List[str]], expected_inputs: List[List[Dict[str, Any]]],
                           tools_used: List[List[str]], inputs_used: List[List[Dict[str, Any]]],
                           skip_input_tools: Set[str]) -> float:
    """
    Calculate TEFS score for a single task.
    Returns the weight (number of tools) if efficiently finished, 0 otherwise.
    """
    num_tools = len(list(flatten(expected_tools)))
    if is_efficiently_finished_task(expected_tools, expected_inputs, tools_used, inputs_used, skip_input_tools):
        return float(num_tools)
    return 0.0


def load_results_for_model(model_name: str, results_dir: str = "results") -> List[Dict[str, Any]]:
    """
    Load all run1/2/3/4 results for a given model.
    
    Args:
        model_name: Model name to search for
        results_dir: Directory containing result files
        
    Returns:
        List of result data dictionaries, sorted by run number
    """
    results = []
    results_path = Path(results_dir)
    
    # Pattern: *_general_test_run{1-4}_results.json
    pattern = re.compile(r".+_general_test_run([1-4])_results\.json$")
    
    run_files = []
    for file_path in results_path.glob("*.json"):
        match = pattern.match(file_path.name)
        if match:
            extracted_name = extract_model_name(file_path.name)
            if extracted_name == model_name:
                run_num = int(match.group(1))
                run_files.append((run_num, file_path))
    
    # Sort by run number
    run_files.sort(key=lambda x: x[0])
    
    for run_num, file_path in run_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
    
    return results


def extract_model_name(filename: str) -> str:
    """
    Extract model name based on substring between `_general_test_run`
    and the previous underscore (if any). If there is no previous underscore,
    take from start to `_general_test_run`.
    
    Args:
        filename: Result filename (e.g., "gemini-2.5-flash_general_test_run1_results.json")
        
    Returns:
        Model name segment according to the rule.
    """
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


def get_task_category(task_id: str) -> str:
    """
    Get task category from task_id.
    
    Args:
        task_id: Task ID (e.g., "daytask_1_tool_18", "protask_2_parallel_tools_10")
        
    Returns:
        Category string: "1_tool", "2_parallel", "2_sequential", "3_tools", or "other"
    """
    task_id_lower = task_id.lower()
    
    if "_1_tool_" in task_id_lower:
        return "1_tool"
    elif "_2_parallel_tools_" in task_id_lower or "_2_parallel_" in task_id_lower:
        return "2_parallel"
    elif "_2_sequential_tools_" in task_id_lower or "_2_sequential_" in task_id_lower:
        return "2_sequential"
    elif "_3_tools_" in task_id_lower:
        return "3_tools"
    else:
        return "other"


def calculate_avg_at_4_with_categories(model_results: List[Dict[str, Any]], skip_input_tools: Set[str]) -> Tuple[float, Dict[str, float]]:
    """
    Calculate avg@4 for a model using TEFS, with category breakdown.
    
    Args:
        model_results: List of result dictionaries (one per run)
        skip_input_tools: Set of tool names to skip input comparison
        
    Returns:
        Tuple of (avg@4, category_breakdown)
        category_breakdown: Dict containing category scores contributing to avg@4
    """
    if not model_results:
        return 0.0, {}
    
    # Group tasks by task_id across all runs
    task_results_by_id = defaultdict(list)
    
    for result_data in model_results:
        detailed_results = result_data.get("detailed_results", [])
        for task_result in detailed_results:
            task_id = task_result.get("task_id")
            if task_id:
                task_results_by_id[task_id].append(task_result)
    
    # Calculate TEFS for each task in each run
    task_tefs_scores = defaultdict(list)  # task_id -> list of TEFS scores across runs
    
    for task_id, task_runs in task_results_by_id.items():
        for task_result in task_runs:
            expected_tools = task_result.get("expected_tools", [])
            expected_inputs = task_result.get("expected_inputs", [])
            tools_used = task_result.get("tools_used", [])
            inputs_used = task_result.get("inputs_used", [])
            
            # Calculate TEFS for this run
            tefs_score = calculate_tefs_for_task(
                expected_tools, expected_inputs,
                tools_used, inputs_used,
                skip_input_tools
            )
            
            # Get weight (number of tools)
            num_tools = len(list(flatten(expected_tools)))
            weight = float(num_tools)
            
            # Store normalized score (TEFS / weight) for avg@4 calculation
            normalized_score = tefs_score / weight if weight > 0 else 0.0
            task_tefs_scores[task_id].append((tefs_score, weight, normalized_score))
    
    # Calculate pass@4 and avg@4 with category breakdown
    total_weight = 0.0
    total_avg_score = 0.0
    
    # Category-specific weights
    category_avg_scores = defaultdict(float)  # category -> total avg score
    category_total_weights = defaultdict(float)  # category -> total weight
    
    for task_id, scores_list in task_tefs_scores.items():
        # Get weight from first run (should be same for all runs)
        weight = scores_list[0][1]
        total_weight += weight
        
        # Get task category
        category = get_task_category(task_id)
        category_total_weights[category] += weight
        
        # Check if at least one run passed (pass@4)
        # Calculate average normalized score across runs (avg@4)
        avg_normalized = sum(score[2] for score in scores_list) / len(scores_list)
        total_avg_score += avg_normalized * weight
        category_avg_scores[category] += avg_normalized * weight
    
    # Calculate final scores
    avg_at_4 = (total_avg_score / total_weight * 100) if total_weight > 0 else 0.0
    
    # Calculate category breakdown (absolute contribution to total score)
    category_breakdown = {}
    
    for category in ["1_tool", "2_parallel", "2_sequential", "3_tools"]:
        cat_avg_score = category_avg_scores.get(category, 0.0)
        
        category_breakdown[category] = (cat_avg_score / total_weight * 100) if total_weight > 0 else 0.0
    
    return avg_at_4, category_breakdown


def find_all_models(results_dir: str = "results") -> List[str]:
    """
    Find all unique model names from result files.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        List of unique model names
    """
    results_path = Path(results_dir)
    models = set()
    
    pattern = re.compile(r".+_general_test_run[1-4]_results\.json$")
    
    for file_path in results_path.glob("*.json"):
        if pattern.match(file_path.name):
            model_name = extract_model_name(file_path.name)
            models.add(model_name)
    
    return sorted(list(models))


def plot_tefs(results_dir: str = "results", output_path: str = "tefs_comparison.png"):
    """
    Plot TEFS comparison chart showing avg@4 for each model with category breakdown.
    
    Args:
        results_dir: Directory containing result files
        output_path: Path to save the plot
    """
    # Load skip input tools
    skip_input_tools = load_skip_input_tools()
    
    # Find all models
    models = find_all_models(results_dir)
    
    if not models:
        print("No models found with run1/2/3/4 results")
        return
    
    print(f"Found {len(models)} models: {', '.join(models)}")
    
    # Category labels
    category_labels = {
        "1_tool": "Single Tool",
        "2_parallel": "Dual Tool Parallel",
        "2_sequential": "Dual Tool Serial",
        "3_tools": "Multi Tool"
    }
    
    # Category colors
    category_colors = {
        "1_tool": "#FFbc31",      # Yellow
        "2_parallel": "#a4d690",   # Light Green
        "2_sequential": "#ff7830", # Orange
        "3_tools": "#52a4d9"       # Blue
    }
    
    # Calculate avg@4 for each model
    model_scores = {}
    model_category_breakdown = {}
    
    for model_name in models:
        print(f"Processing {model_name}...")
        results = load_results_for_model(model_name, results_dir)
        
        if len(results) != 4:
            print(f"Warning: {model_name} has {len(results)} runs, expected 4. Skipping.")
            continue
        
        avg_at_4, category_breakdown = calculate_avg_at_4_with_categories(results, skip_input_tools)
        model_scores[model_name] = avg_at_4
        model_category_breakdown[model_name] = category_breakdown
        print(f"  avg@4: {avg_at_4:.2f}%")
    
    if not model_scores:
        print("No valid models found")
        return
    
    # Prepare data for plotting
    model_names = list(model_scores.keys())
    n_models = len(model_names)
    x = np.arange(n_models)
    width = 0.6  # Width of bars
    
    fig, ax = plt.subplots(figsize=(max(12, n_models * 0.8), 6))
    
    # Prepare category data for stacked bars
    categories = ["1_tool", "2_parallel", "2_sequential", "3_tools"]
    
    # Extract category contributions (already calculated as absolute contributions to total score)
    category_data = {cat: [] for cat in categories}
    
    for model_name in model_names:
        breakdown = model_category_breakdown[model_name]
        for cat in categories:
            category_data[cat].append(breakdown.get(cat, 0.0))
    
    # Calculate total scores for labels
    avg_totals = [model_scores[m] for m in model_names]
    
    # Plot avg@4 stacked bars
    bottom_avg = np.zeros(n_models)
    for cat in categories:
        ax.bar(x, category_data[cat], width,
               bottom=bottom_avg,
               color=category_colors[cat], alpha=0.8,
               edgecolor='black', linewidth=0.5)
        bottom_avg += np.array(category_data[cat])
    
    # Add total score labels on top of bars
    for i, avg_total in enumerate(avg_totals):
        if avg_total > 0:
            ax.text(x[i], avg_total + 1,
                   f'{avg_total:.1f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Customize chart
    ax.set_ylabel('TEFS avg@4 (%)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(0, 100)
    
    # Create custom legend with category labels
    handles = []
    labels = []
    for cat in categories:
        handles.append(plt.Rectangle((0,0),1,1, facecolor=category_colors[cat], alpha=0.8))
        labels.append(category_labels[cat])
    
    ax.legend(handles, labels, fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot TEFS comparison chart")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing result files (default: results)")
    parser.add_argument("--output", type=str, default="tefs_comparison.png",
                       help="Output file path (default: tefs_comparison.png)")
    
    args = parser.parse_args()
    
    plot_tefs(args.results_dir, args.output)

