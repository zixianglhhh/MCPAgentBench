#!/usr/bin/env python3
"""
Plot Time Efficiency comparison chart.
Shows avg@4 Time Efficiency for each model.

Time Efficiency = Total Score / (sum of Time in minutes)
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np


def load_results_for_model(model_name: str, results_dir: str = "results") -> List[Dict[str, Any]]:
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


def calculate_time_efficiency_avg_at_4(model_results: List[Dict[str, Any]]) -> float:
    """
    Calculate avg@4 Time Efficiency for a model.
    
    Time Efficiency = Total Score / (sum of Time in minutes)
    
    According to the formula:
    - Time_i represents the execution time in minutes for task i
    - Time Efficiency = Total Score / sum(Time_i)
    
    For each run:
    - Get model_score from evaluation_summary (Total Score, in percentage 0-100)
    - Get total_test_time from evaluation_summary (in seconds)
    - Convert total_test_time to minutes: total_test_time / 60
    - Calculate: time_efficiency = model_score / (total_test_time / 60)
    
    Then average across 4 runs.
    
    Args:
        model_results: List of result dictionaries (one per run)
        
    Returns:
        avg@4 Time Efficiency score
    """
    if not model_results:
        return 0.0
    
    time_efficiencies = []
    
    for result_data in model_results:
        evaluation_summary = result_data.get("evaluation_summary", {})
        
        # Get model_score (this is the Total Score, already in percentage 0-100)
        model_score = evaluation_summary.get("model_score", 0.0)
        
        # Get total_test_time (in seconds)
        total_test_time_seconds = evaluation_summary.get("total_test_time", 0.0)
        
        if total_test_time_seconds > 0:
            # Convert seconds to minutes
            total_test_time_minutes = total_test_time_seconds / 60.0
            
            # Calculate time efficiency
            # Time Efficiency = model_score / total_test_time_minutes
            time_efficiency = model_score / total_test_time_minutes
            time_efficiencies.append(time_efficiency)
        else:
            time_efficiencies.append(0.0)
    
    # Average across 4 runs
    avg_time_efficiency = sum(time_efficiencies) / len(time_efficiencies) if time_efficiencies else 0.0
    
    return avg_time_efficiency


def extract_model_name(filename: str) -> str:
    """
    Extract model name based on substring between `_general_test_run`
    and the previous underscore (if any). If there is no previous underscore,
    take from start to `_general_test_run`.
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


def find_all_models(results_dir: str = "results") -> List[str]:
    """
    Find all unique model names from result files.
    """
    results_path = Path(results_dir)
    models = set()
    
    pattern = re.compile(r"^.+_general_test_run[1-4]_results\.json$")
    
    for file_path in results_path.glob("*.json"):
        if pattern.match(file_path.name):
            model_name = extract_model_name(file_path.name)
            models.add(model_name)
    
    return sorted(list(models))


def plot_time_efficiency(results_dir: str = "results", output_path: str = "time_efficiency.png"):
    """
    Plot Time Efficiency comparison chart showing avg@4 for each model.
    
    Args:
        results_dir: Directory containing result files
        output_path: Path to save the plot
    """
    # Find all models
    models = find_all_models(results_dir)
    
    if not models:
        print("No models found with run1/2/3/4 results")
        return
    
    print(f"Found {len(models)} models: {', '.join(models)}")
    
    # Color
    bar_color = "#52a4d9"  # Blue
    
    # Calculate avg@4 Time Efficiency for each model
    model_scores = {}
    
    for model_name in models:
        print(f"Processing {model_name}...")
        results = load_results_for_model(model_name, results_dir)
        
        if len(results) != 4:
            print(f"Warning: {model_name} has {len(results)} runs, expected 4. Skipping.")
            continue
        
        time_efficiency = calculate_time_efficiency_avg_at_4(results)
        model_scores[model_name] = time_efficiency
        print(f"  Time Efficiency avg@4: {time_efficiency:.3g}")
    
    if not model_scores:
        print("No valid models found")
        return
    
    # Prepare data for plotting
    model_names = list(model_scores.keys())
    n_models = len(model_names)
    x = np.arange(n_models)
    width = 0.6  # Width of bars
    
    fig, ax = plt.subplots(figsize=(max(12, n_models * 0.8), 6))
    
    # Extract scores
    scores = [model_scores[m] for m in model_names]
    
    # Plot bars
    bars = ax.bar(x, scores, width,
                  color=bar_color, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    
    # Add score labels on top of bars
    for i, score in enumerate(scores):
        if score > 0:
            # Format with 3 significant digits
            ax.text(x[i], score + max(scores) * 0.01,
                   f'{score:.3g}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Customize chart
    ax.set_ylabel('Time Efficiency', fontsize=16)
    # ax.set_xlabel('Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=18)
    ax.tick_params(axis='both', labelsize=12)
    max_score = max(scores) if scores else 1
    ax.set_ylim(0, max_score * 1.3)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot Time Efficiency comparison chart")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing result files (default: results)")
    parser.add_argument("--output", type=str, default="time_efficiency.png",
                       help="Output file path (default: time_efficiency.png)")
    
    args = parser.parse_args()
    
    plot_time_efficiency(args.results_dir, args.output)

