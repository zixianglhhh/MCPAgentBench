#!/usr/bin/env python3
"""
Plot model size vs TEFS score line chart.
Shows TEFS score (average of model_score from run1-4) for different model sizes.
Two lines: qwen3 series (8b, 14b, 32b) and qwen2.5 series (7b, 14b, 32b, 72b).
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


def extract_model_name(filename: str) -> str:
    """
    Extract model name from filename.
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


def extract_model_size_and_series(model_name: str) -> Optional[Tuple[str, str, int]]:
    """
    Extract model series (qwen3 or qwen2.5) and size from model name.
    
    Args:
        model_name: Model name like "qwen3-8b" (after extract_model_name) or "qwen2.5-7b-instruct"
        
    Returns:
        Tuple of (series, size_str, size_int) or None if not matched
        series: "qwen3" or "qwen2.5"
        size_str: "8b", "14b", etc.
        size_int: 8, 14, 32, 72 for sorting
    """
    # Match qwen3 series: qwen3-8b (extracted name from qwen_qwen3-8b)
    qwen3_match = re.match(r"qwen3-(\d+)b", model_name)
    if qwen3_match:
        size_str = qwen3_match.group(1) + "b"
        size_int = int(qwen3_match.group(1))
        return ("qwen3", size_str, size_int)
    
    # Match qwen2.5 series: qwen2.5-7b-instruct
    qwen25_match = re.match(r"qwen2\.5-(\d+)b-instruct", model_name)
    if qwen25_match:
        size_str = qwen25_match.group(1) + "b"
        size_int = int(qwen25_match.group(1))
        return ("qwen2.5", size_str, size_int)
    
    return None


def load_results_for_model(model_name: str, results_dir: str = "results") -> List[Dict]:
    """
    Load all run1/2/3/4 results for a given model.
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


def calculate_tefs_avg_at_4(model_results: List[Dict]) -> float:
    """
    Calculate TEFS avg@4 from model_score in evaluation_summary.
    
    Args:
        model_results: List of result dictionaries (one per run)
        
    Returns:
        Average model_score across 4 runs
    """
    if not model_results:
        return 0.0
    
    scores = []
    for result_data in model_results:
        evaluation_summary = result_data.get("evaluation_summary", {})
        model_score = evaluation_summary.get("model_score", 0.0)
        scores.append(model_score)
    
    # Average across runs
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score


def find_models_by_series(results_dir: str = "results") -> Dict[str, List[Tuple[str, int]]]:
    """
    Find all qwen3 and qwen2.5 models and extract their sizes.
    
    Returns:
        Dict with keys "qwen3" and "qwen2.5", values are lists of (model_name, size_int) tuples
    """
    results_path = Path(results_dir)
    pattern = re.compile(r".+_general_test_run[1-4]_results\.json$")
    
    models_found = set()
    for file_path in results_path.glob("*.json"):
        if pattern.match(file_path.name):
            model_name = extract_model_name(file_path.name)
            models_found.add(model_name)
    
    # Group by series
    qwen3_models = []
    qwen25_models = []
    
    for model_name in models_found:
        info = extract_model_size_and_series(model_name)
        if info:
            series, size_str, size_int = info
            if series == "qwen3":
                # Only include 8b, 14b, 32b
                if size_int in [8, 14, 32]:
                    qwen3_models.append((model_name, size_int))
            elif series == "qwen2.5":
                # Only include 7b, 14b, 32b, 72b
                if size_int in [7, 14, 32, 72]:
                    qwen25_models.append((model_name, size_int))
    
    # Sort by size
    qwen3_models.sort(key=lambda x: x[1])
    qwen25_models.sort(key=lambda x: x[1])
    
    return {
        "qwen3": qwen3_models,
        "qwen2.5": qwen25_models
    }


def plot_model_size_tefs(results_dir: str = "results", output_path: str = "model_size_tefs.png"):
    """
    Plot model size vs TEFS score line chart.
    
    Args:
        results_dir: Directory containing result files
        output_path: Path to save the plot
    """
    # Find models by series
    models_by_series = find_models_by_series(results_dir)
    
    qwen3_models = models_by_series.get("qwen3", [])
    qwen25_models = models_by_series.get("qwen2.5", [])
    
    if not qwen3_models and not qwen25_models:
        print("No qwen3 or qwen2.5 models found")
        return
    
    print(f"Found qwen3 models: {[m[0] for m in qwen3_models]}")
    print(f"Found qwen2.5 models: {[m[0] for m in qwen25_models]}")
    
    # Calculate TEFS scores for each model
    qwen3_data = []  # List of (size_int, size_str, score) tuples
    qwen25_data = []  # List of (size_int, size_str, score) tuples
    
    for model_name, size_int in qwen3_models:
        print(f"Processing {model_name}...")
        results = load_results_for_model(model_name, results_dir)
        
        if len(results) != 4:
            print(f"Warning: {model_name} has {len(results)} runs, expected 4. Skipping.")
            continue
        
        tefs_score = calculate_tefs_avg_at_4(results)
        size_str = f"{size_int}b"
        qwen3_data.append((size_int, size_str, tefs_score))
        print(f"  TEFS avg@4: {tefs_score:.2f}%")
    
    for model_name, size_int in qwen25_models:
        print(f"Processing {model_name}...")
        results = load_results_for_model(model_name, results_dir)
        
        if len(results) != 4:
            print(f"Warning: {model_name} has {len(results)} runs, expected 4. Skipping.")
            continue
        
        tefs_score = calculate_tefs_avg_at_4(results)
        size_str = f"{size_int}b"
        qwen25_data.append((size_int, size_str, tefs_score))
        print(f"  TEFS avg@4: {tefs_score:.2f}%")
    
    if not qwen3_data and not qwen25_data:
        print("No valid data to plot")
        return
    
    # Prepare data for plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine all unique sizes and create mapping to equal-width positions
    all_sizes_set = set([d[0] for d in qwen3_data] + [d[0] for d in qwen25_data])
    has_7 = 7 in all_sizes_set
    has_8 = 8 in all_sizes_set
    merge_7_8 = has_7 and has_8
    
    # Create size to position mapping (4 equal-width positions: 0, 1, 2, 3)
    # Collect all unique size categories
    size_categories = []
    if merge_7_8:
        size_categories.append("7/8")
    else:
        if has_7:
            size_categories.append("7")
        if has_8:
            size_categories.append("8")
    
    for size in sorted(all_sizes_set):
        if size not in [7, 8]:
            size_categories.append(str(size))
    
    # Create mapping: size -> position (0, 1, 2, 3)
    size_to_pos = {}
    pos_to_label = {}
    for pos, size_str in enumerate(size_categories):
        if size_str == "7/8":
            size_to_pos[7] = pos
            size_to_pos[8] = pos
            pos_to_label[pos] = "7/8b"
        else:
            size_int = int(size_str)
            size_to_pos[size_int] = pos
            pos_to_label[pos] = f"{size_int}b"
    
    def map_size_to_position(size):
        """Map actual size to equal-width position"""
        return size_to_pos.get(size, size)
    
    # Plot qwen3 series
    if qwen3_data:
        qwen3_positions = [map_size_to_position(d[0]) for d in qwen3_data]
        qwen3_scores = [d[2] for d in qwen3_data]
        ax.plot(qwen3_positions, qwen3_scores, marker='o', linewidth=5, markersize=14,
                label='Qwen3', color='#52a4d9', alpha=0.8)
    
    # Plot qwen2.5 series
    if qwen25_data:
        qwen25_positions = [map_size_to_position(d[0]) for d in qwen25_data]
        qwen25_scores = [d[2] for d in qwen25_data]
        ax.plot(qwen25_positions, qwen25_scores, marker='s', linewidth=5, markersize=14,
                label='Qwen2.5', color='#FFbc31', alpha=0.8)
    
    # Customize chart
    ax.set_xlabel('Model Size (B)', fontsize=22)
    ax.set_ylabel('TEFS avg@4 (%)', fontsize=22)
    
    # Set x-axis ticks: 4 equal-width positions
    num_ticks = len(size_categories)
    tick_positions = list(range(num_ticks))
    tick_labels = [pos_to_label[pos] for pos in tick_positions]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=20, loc='best')
    
    # Set y-axis limits to 0-100
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot model size vs TEFS score line chart")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing result files (default: results)")
    parser.add_argument("--output", type=str, default="model_size_tefs.png",
                       help="Output file path (default: model_size_tefs.png)")
    
    args = parser.parse_args()
    
    plot_model_size_tefs(args.results_dir, args.output)

