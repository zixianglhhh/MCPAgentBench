#!/usr/bin/env python3
"""
Plot TEFS and TFS avg@4 on a single chart with paired bars per model.
"""

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Ensure project modules are importable
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import load_skip_input_tools
from tools import plot_tefs as tefs_module
from tools import plot_tfs as tfs_module


def compute_metric_for_model(
    model_name: str,
    results_dir: str,
    skip_input_tools,
) -> Dict[str, float]:
    """
    Load all runs for the given model and compute TEFS/TFS avg@4 scores.
    """
    results = tefs_module.load_results_for_model(model_name, results_dir)

    if len(results) != 4:
        print(f"Warning: {model_name} has {len(results)} runs, expected 4. Skipping.")
        return {}

    tefs_avg, _ = tefs_module.calculate_avg_at_4_with_categories(results, skip_input_tools)
    tfs_avg, _ = tfs_module.calculate_avg_at_4_with_categories(results, skip_input_tools)

    return {"tefs": tefs_avg, "tfs": tfs_avg}


def plot_tefs_tfs(results_dir: str = "results", output_path: str = "tefs_tfs_comparison.png"):
    """
    Render a combined TFS/TEFS comparison bar chart.
    """
    skip_input_tools = load_skip_input_tools()
    models = tefs_module.find_all_models(results_dir)

    if not models:
        print("No models found with 4-run results.")
        return

    print(f"Found {len(models)} models: {', '.join(models)}")

    tefs_scores: Dict[str, float] = {}
    tfs_scores: Dict[str, float] = {}

    for model in models:
        print(f"Processing {model} ...")
        metrics = compute_metric_for_model(model, results_dir, skip_input_tools)
        if not metrics:
            continue

        tefs_scores[model] = metrics["tefs"]
        tfs_scores[model] = metrics["tfs"]
        print(f"  TEFS avg@4: {metrics['tefs']:.2f}%, TFS avg@4: {metrics['tfs']:.2f}%")

    if not tefs_scores:
        print("No data available for plotting.")
        return

    model_names: List[str] = list(tefs_scores.keys())
    n_models = len(model_names)
    x = np.arange(n_models)
    width = 0.35

    tefs_vals = [tefs_scores[m] for m in model_names]
    tfs_vals = [tfs_scores[m] for m in model_names]

    fig, ax = plt.subplots(figsize=(max(12, n_models * 0.8), 6))

    bar_tfs = ax.bar(
        x - width / 2,
        tfs_vals,
        width,
        label="TFS",
        color="#808080",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    bar_tefs = ax.bar(
        x + width / 2,
        tefs_vals,
        width,
        label="TEFS",
        color="#FFbc31",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    for i, (tfs, tefs) in enumerate(zip(tfs_vals, tefs_vals)):
        if tfs > 0:
            ax.text(
                x[i] - width / 2,
                tfs + 1,
                f"{tfs:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        if tefs > 0:
            ax.text(
                x[i] + width / 2,
                tefs + 1,
                f"{tefs:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_ylabel("avg@4 (%)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=18)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=12, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nChart saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot combined TEFS/TFS comparison chart")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing result files")
    parser.add_argument("--output", type=str, default="tefs_tfs_comparison.png", help="Output image path")

    args = parser.parse_args()
    plot_tefs_tfs(args.results_dir, args.output)

