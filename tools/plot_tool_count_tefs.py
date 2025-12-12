#!/usr/bin/env python3
"""
Plot TEFS avg@4 vs tool count line chart.
Shows how TEFS score changes with different tool counts (10/20/30/40) for three models:
- DeepSeek-V3.2-Exp
- kimi-k2-thinking
- qwen3-235b-a22b-instruct-2507
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import sys

# Import evaluation utilities
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.evaluate import load_skip_input_tools
from tools import plot_tefs as tefs_module


def load_results_for_model_in_tool_dir(
    model_name: str, tool_dir: Path
) -> List[Dict]:
    """
    Load all run1/2/3/4 results for a given model from a specific tool directory.
    
    Args:
        model_name: Model name to search for
        tool_dir: Path to tool directory (e.g., tool10, tool20, etc.)
        
    Returns:
        List of result data dictionaries, sorted by run number
    """
    results = []
    
    # Pattern: {model_name}_general_test_run{1-4}_results.json
    pattern = re.compile(rf"^{re.escape(model_name)}_general_test_run([1-4])_results\.json$")
    
    run_files = []
    for file_path in tool_dir.glob("*.json"):
        match = pattern.match(file_path.name)
        if match:
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


def calculate_tefs_avg_at_4_for_tool_count(
    model_name: str, tool_count: int, results_dir: str = "results"
) -> float:
    """
    Calculate TEFS avg@4 for a model at a specific tool count.
    
    Args:
        model_name: Model name
        tool_count: Tool count (10, 20, 30, or 40)
        results_dir: Base results directory
        
    Returns:
        TEFS avg@4 score
    """
    tool_dir = Path(results_dir) / f"tool{tool_count}"
    
    if not tool_dir.exists():
        print(f"Warning: Directory {tool_dir} does not exist")
        return 0.0
    
    results = load_results_for_model_in_tool_dir(model_name, tool_dir)
    
    if len(results) != 4:
        print(f"Warning: {model_name} in tool{tool_count} has {len(results)} runs, expected 4")
        return 0.0
    
    skip_input_tools = load_skip_input_tools()
    avg_at_4, _ = tefs_module.calculate_avg_at_4_with_categories(results, skip_input_tools)
    
    return avg_at_4


def plot_tool_count_tefs(
    results_dir: str = "results", output_path: str = "tool_count_tefs.png"
):
    """
    Plot TEFS avg@4 vs tool count line chart.
    
    Args:
        results_dir: Directory containing result files
        output_path: Path to save the plot
    """
    # Define models and tool counts
    models = [
        "deepseek-chat",
        "kimi-k2-thinking",
        "qwen3-235b-a22b-instruct-2507"
    ]
    tool_counts = [10, 20, 30, 40]
    
    # Model display names and colors
    model_display_names = {
        "deepseek-chat": "DeepSeek-V3.2",
        "kimi-k2-thinking": "Kimi-K2-Thinking",
        "qwen3-235b-a22b-instruct-2507": "Qwen3-235B"
    }
    model_colors = {
        "deepseek-chat": "#52a4d9",
        "kimi-k2-thinking": "#FFbc31",
        "qwen3-235b-a22b-instruct-2507": "#a4d690"
    }
    model_markers = {
        "deepseek-chat": "o",
        "kimi-k2-thinking": "s",
        "qwen3-235b-a22b-instruct-2507": "^"
    }
    
    # Calculate TEFS scores for each model at each tool count
    model_data = {}  # model_name -> list of (tool_count, score) tuples
    
    for model_name in models:
        print(f"Processing {model_name}...")
        scores = []
        for tool_count in tool_counts:
            print(f"  Tool count {tool_count}...")
            score = calculate_tefs_avg_at_4_for_tool_count(
                model_name, tool_count, results_dir
            )
            scores.append((tool_count, score))
            print(f"    TEFS avg@4: {score:.2f}%")
        model_data[model_name] = scores
    
    # Prepare data for plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each model
    for model_name in models:
        data = model_data[model_name]
        tool_counts_plot = [d[0] for d in data]
        scores_plot = [d[1] for d in data]
        
        display_name = model_display_names[model_name]
        color = model_colors[model_name]
        marker = model_markers[model_name]
        
        ax.plot(
            tool_counts_plot,
            scores_plot,
            marker=marker,
            linewidth=5,
            markersize=14,
            label=display_name,
            color=color,
            alpha=0.8
        )
    
    # Customize chart
    ax.set_xlabel('Tool Count', fontsize=22)
    ax.set_ylabel('TEFS avg@4 (%)', fontsize=22)
    ax.set_xticks(tool_counts)
    ax.set_xticklabels([f"{tc}" for tc in tool_counts])
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
    
    parser = argparse.ArgumentParser(
        description="Plot TEFS avg@4 vs tool count line chart"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing result files (default: results)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tool_count_tefs.png",
        help="Output file path (default: tool_count_tefs.png)"
    )
    
    args = parser.parse_args()
    
    plot_tool_count_tefs(args.results_dir, args.output)

