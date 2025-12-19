from .utilities import *
import json
import os
from pathlib import Path


def load_skip_input_tools(config_path: str = "configs/evaluation_config.json") -> Set[str]:
    """
    Load skip input tools list from configuration file.
    
    Args:
        config_path: Path to evaluation configuration file
        
    Returns:
        Set[str]: Set of tool names to skip when comparing inputs
    """
    config_file = Path(config_path)
    if not config_file.exists():
        # Return default empty set if config file doesn't exist
        print(f"Warning: Evaluation config file not found at {config_path}, using empty skip list")
        return set()
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        skip_tools = config.get("skip_input_tools", [])
        return set(skip_tools)
    except Exception as e:
        print(f"Warning: Failed to load skip_input_tools from {config_path}: {e}, using empty skip list")
        return set()


def evaluate_task_performance(response_data: Dict[str, Any], expected_tools: List[List[str]], expected_inputs: List[List[Dict[str, Any]]], num_expected_tools: int) -> int:
    """
    Evaluate a single task's performance.
    Returns 1 * num_expected_tools (i.e give different weights to tasks with different number of expected tools) if both tools and inputs match exactly, 0 otherwise.
    """

    # Load tools to skip when comparing inputs from config
    SKIP_INPUT_TOOLS = load_skip_input_tools()

    tools_used, inputs_used = extract_tools_and_inputs(response_data)
    
    # Check if tools match exactly (including order in different toolcallrequest steps)
    for i, (actual_tool, expected_tool) in enumerate(zip(tools_used, expected_tools)):
        if set(actual_tool) != set(expected_tool):
            return 0

    # Check if inputs length match exactly
    if len(inputs_used) != len(expected_inputs):
        return 0
    
    for i, (actual_inputs_step, expected_inputs_step) in enumerate(zip(inputs_used, expected_inputs)):
        # Get the tools for this step
        expected_tool_names_in_step = expected_tools[i]
        actual_tool_names_in_step = tools_used[i]
        
        # Create list of (tool_name, input) pairs to handle duplicate tool names
        expected_tool_input_pairs = [
            (tool_name, expected_inputs_step[j])
            for j, tool_name in enumerate(expected_tool_names_in_step)
            if j < len(expected_inputs_step)
        ]
        
        actual_tool_input_pairs = [
            (tool_name, actual_inputs_step[j])
            for j, tool_name in enumerate(actual_tool_names_in_step)
            if j < len(actual_inputs_step)
        ]
        
        # Filter out inputs for tools in the skip list
        filtered_expected_inputs = [
            input_data
            for tool_name, input_data in expected_tool_input_pairs
            if tool_name not in SKIP_INPUT_TOOLS
        ]
        
        filtered_actual_inputs = [
            input_data
            for tool_name, input_data in actual_tool_input_pairs
            if tool_name not in SKIP_INPUT_TOOLS
        ]
        
        # Compare filtered inputs (only for tools not in skip list)
        if filtered_actual_inputs or filtered_expected_inputs:
            if not compare_inputs(filtered_actual_inputs, filtered_expected_inputs):
                return 0

    score = 1 * num_expected_tools

    return score


def evaluate_task_performance_ignore_parallel(response_data: Dict[str, Any], expected_tools: List[List[str]], expected_inputs: List[List[Dict[str, Any]]], num_expected_tools: int) -> int:
    """
    Evaluate a single task's performance, ignoring parallel/sequential differences.
    If tools that should be parallel are called sequentially, it's still considered correct.
    This focuses on whether the task is completed, not the efficiency of execution.
    Returns 1 * num_expected_tools if tools and inputs match (ignoring parallel/sequential), 0 otherwise.
    """

    # Load tools to skip when comparing inputs from config
    SKIP_INPUT_TOOLS = load_skip_input_tools()

    tools_used, inputs_used = extract_tools_and_inputs(response_data)
    
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
        return 0
    
    # Check if inputs match (ignoring order and parallel/sequential structure)
    # Create sets of (tool_name, input_dict) pairs for comparison
    filtered_expected_inputs = [
        (tool_name, input_data)
        for tool_name, input_data in expected_inputs_flat
        if tool_name not in SKIP_INPUT_TOOLS
    ]
    
    filtered_actual_inputs = [
        (tool_name, input_data)
        for tool_name, input_data in actual_inputs_flat
        if tool_name not in SKIP_INPUT_TOOLS
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
            return 0

    score = 1 * num_expected_tools

    return score
