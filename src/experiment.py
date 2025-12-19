from .utilities import *
from .evaluate import *
from .config import *
from .agenttest import *
import os
import json
import datetime
from tqdm import tqdm


def load_experiment_config(config_path: str = "configs/config.json"):
    """
    Load experiment configuration from config.json.
    
    Args:
        config_path: Path to the experiment configuration file
        
    Returns:
        dict: Configuration dictionary with concurrency, num_servers
    """
    if not os.path.exists(config_path):
        # Return default values if config file doesn't exist
        return {
            "concurrency": 10,
            "num_servers": 10
        }
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return {
        "concurrency": config.get("concurrency", 10),
        "num_servers": config.get("num_servers", 10)
    }


def get_experiment_config(model, tasks_type, output_name=None):
    """
    Get the experiment configuration.

    Args:
        model: The model to test.
        tasks_type: The type of tasks to test. Support daily tasks (day), professional tasks (pro), and all combined tasks (general_test).
        output_name: Custom output file name (without extension). If None, will use timestamp-based naming.

    Returns:
        log_path: The path to the log file.
        task_path: The path to the task file.
        output_path: The path to the result file.
    """
    model_safe = model.replace("/", "_")
    
    if tasks_type == "day":
        task_path = "data/daytasks.json"
        if output_name:
            log_path = f"logs/{output_name}_response.json"
            output_path = f"results/{output_name}_results.json"
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"logs/{model_safe}_response_day_{timestamp}.json"
            output_path = f"results/{model_safe}_results_day_{timestamp}.json"

    elif tasks_type == "pro":
        task_path = "data/protasks.json"
        if output_name:
            log_path = f"logs/{output_name}_response.json"
            output_path = f"results/{output_name}_results.json"
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"logs/{model_safe}_response_pro_{timestamp}.json"
            output_path = f"results/{model_safe}_results_pro_{timestamp}.json"
    
    elif tasks_type == "general_test":
        task_path = "data/tasks.json"
        if output_name:
            log_path = f"logs/{output_name}_response.json"
            output_path = f"results/{output_name}_results.json"
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"logs/{model_safe}_response_general_test_{timestamp}.json"
            output_path = f"results/{model_safe}_results_general_test_{timestamp}.json"
    else:
        raise ValueError(f"Unknown tasks_type: {tasks_type}")

    return log_path, task_path, output_path


async def run_experiment(model, tasks_type, concurrency=None, num_servers=None, output_name=None):
    """
    Run the benchmark.

    Args:
        model: The model to test (required).
        tasks_type: The type of tasks to test (required). Options: day, pro, general_test
        concurrency: The number of concurrent requests to send. If None, will be read from config.json
        num_servers: The number of servers for agent construction. If None, will be read from config.json
        output_name: Custom output file name (without extension). If None, will use timestamp-based naming.
    """
    # Load configuration from file
    config = load_experiment_config()
    
    # Use provided parameters or fall back to config file values
    concurrency = concurrency if concurrency is not None else config["concurrency"]
    num_servers = num_servers if num_servers is not None else config["num_servers"]

    scores = []
    ignore_parallel_scores = []
    detailed_results = []
    tools_summed_up = []
    num_ignore_parallel_success_tasks = 0

    log_path, task_path, output_path = get_experiment_config(model, tasks_type, output_name)
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    # Generate responses
    await generate_responses_concurrent(model, task_path, log_path, concurrency, num_servers)

    task_data = load_data(task_path)
    response_data = load_data(log_path)
    num_tasks = len(task_data)
    num_empty_responses = 0
    num_success_tasks = 0

    category_labels = {
        "daytask_1_tool": "Daytask 1 Tool",
        "daytask_2_sequential": "Daytask 2 Sequential",
        "daytask_2_parallel": "Daytask 2 Parallel",
        "daytask_3_tools": "Daytask 3 Tools",
        "protask_1_tool": "Protask 1 Tool",
        "protask_2_sequential": "Protask 2 Sequential",
        "protask_2_parallel": "Protask 2 Parallel",
        "protask_3_tools": "Protask 3 Tools",
    }
    category_stats = {key: {"total": 0, "passed": 0, "ignore_parallel_passed": 0} for key in category_labels}

    for i in tqdm(range(num_tasks), desc="Evaluating tasks", unit="task"):
        
        response = response_data[i]
        task = task_data[i]
        task_id = task.get('id', f'task_{i+1}')

        # Extract expected tools and inputs
        expected_tools = task.get('tools', [])
        expected_inputs = task.get('inputs', [])
        num_expected_tools = len(list(flatten(expected_tools)))
        tools_summed_up.append(num_expected_tools)
        # assert num_expected_tools == int(task.get('id')[8])

        # Determine task category for summary statistics
        category_key = None
        if task_id.startswith("daytask"):
            if "_1_tool_" in task_id:
                category_key = "daytask_1_tool"
            if "_2_sequential_" in task_id:
                category_key = "daytask_2_sequential"
            elif "_2_parallel_" in task_id:
                category_key = "daytask_2_parallel"
            elif "_3_tools_" in task_id:
                category_key = "daytask_3_tools"
        elif task_id.startswith("protask"):
            if "_1_tool_" in task_id:
                category_key = "protask_1_tool"
            if "_2_sequential_" in task_id:
                category_key = "protask_2_sequential"
            elif "_2_parallel_" in task_id:
                category_key = "protask_2_parallel"
            elif "_3_tools_" in task_id:
                category_key = "protask_3_tools"

        if category_key:
            category_stats[category_key]["total"] += 1

        try:
            tools_used, inputs_used = extract_tools_and_inputs(response)
            if not tools_used:
                num_empty_responses += 1

            # Evaluate performance
            score = evaluate_task_performance(response, expected_tools, expected_inputs, num_expected_tools)
            ignore_parallel_score = evaluate_task_performance_ignore_parallel(response, expected_tools, expected_inputs, num_expected_tools)

            scores.append(score)
            ignore_parallel_scores.append(ignore_parallel_score)
            
            if score == num_expected_tools:
                num_success_tasks += 1
                if category_key:
                    category_stats[category_key]["passed"] += 1
            
            if ignore_parallel_score == num_expected_tools:
                num_ignore_parallel_success_tasks += 1
                if category_key:
                    category_stats[category_key]["ignore_parallel_passed"] += 1
        except Exception as e:
            print(f"Error evaluating task {i+1}: {str(e)}")
            scores.append(0)
            ignore_parallel_scores.append(0)
            tools_used, inputs_used = [], []

        # Store detailed results
        result = {
            'task_id': task_id,
            'task_content': task.get('content', '')[:100] + '...' if len(task.get('content', '')) > 100 else task.get('content', ''),
            'expected_tools': expected_tools,
            'expected_inputs': expected_inputs,
            'tools_used': tools_used,
            'inputs_used': inputs_used,
            'score': score,
            'match': score == num_expected_tools,
            'index': i+1
        }
        detailed_results.append(result)

    # Calculate overall performance
    full_marks = sum(tools_summed_up)
    total_score = sum(scores)
    model_score = total_score / full_marks if scores else 0
    model_score = round(model_score * 100, 2)
    
    # Calculate ignore_parallel_score
    total_ignore_parallel_score = sum(ignore_parallel_scores)
    ignore_parallel_model_score = total_ignore_parallel_score / full_marks if scores else 0
    ignore_parallel_model_score = round(ignore_parallel_model_score * 100, 2)

    total_prompt_tokens, total_completion_tokens = calculate_total_tokens(response_data)
    total_tokens = total_prompt_tokens + total_completion_tokens
    average_tokens = total_tokens / (len(response_data) - num_empty_responses) if (len(response_data) - num_empty_responses) > 0 else 0

    total_test_time = calculate_total_time(response_data)
    average_time = total_test_time / (len(response_data) - num_empty_responses) if (len(response_data) - num_empty_responses) > 0 else 0
    
    # Print summary
    print("\n" + "=" * 50)
    print("📈 EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Tasks Evaluated: {len(scores)}")
    print(f"Tasks Passed: {num_success_tasks}")
    print(f"Tasks Failed: {len(scores) - num_success_tasks}")
    print(f"Task Finish Score: {model_score}")
    print(f"Ignore Parallel Score: {ignore_parallel_model_score}")
    print(f"full marks: {full_marks}")
    print(f"Total Completion Tokens: {total_completion_tokens:,}")
    print(f"Total Test Time: {total_test_time:.2f} seconds")

    print("\nCategory Scores (Strict):")
    for key, label in category_labels.items():
        stats = category_stats[key]
        total = stats["total"]
        passed = stats["passed"]
        if total > 0:
            accuracy = passed / total
            print(f"- {label}: {passed}/{total} ({accuracy:.2%})")
        else:
            print(f"- {label}: no tasks")
    
    print("\nCategory Scores (Ignore Parallel):")
    for key, label in category_labels.items():
        stats = category_stats[key]
        total = stats["total"]
        ignore_parallel_passed = stats["ignore_parallel_passed"]
        if total > 0:
            accuracy = ignore_parallel_passed / total
            print(f"- {label}: {ignore_parallel_passed}/{total} ({accuracy:.2%})")
        else:
            print(f"- {label}: no tasks")
    
    # Save detailed results to file
    category_summary = {}
    category_summary_ignore_parallel = {}
    for key, label in category_labels.items():
        stats = category_stats[key]
        total = stats["total"]
        passed = stats["passed"]
        ignore_parallel_passed = stats["ignore_parallel_passed"]
        accuracy = round(passed / total * 100, 2) if total > 0 else 0
        ignore_parallel_accuracy = round(ignore_parallel_passed / total * 100, 2) if total > 0 else 0
        category_summary[key] = {
            "label": label,
            "passed": passed,
            "total": total,
            "accuracy": accuracy,
        }
        category_summary_ignore_parallel[key] = {
            "label": label,
            "passed": ignore_parallel_passed,
            "total": total,
            "accuracy": ignore_parallel_accuracy,
        }

    evaluation_summary = {
        'total_tasks': len(scores),
        'tasks_passed': num_success_tasks,
        'tasks_failed': len(scores) - num_success_tasks,
        'model_score': model_score,
        'ignore_parallel_score': ignore_parallel_model_score,
        'ignore_parallel_tasks_passed': num_ignore_parallel_success_tasks,
        'total_completion_tokens': total_completion_tokens,
        'total_prompt_tokens': total_prompt_tokens,
        'total_tokens': total_tokens,
        'average_tokens': average_tokens,
        'total_test_time': total_test_time,
        'average_time': average_time,
        'category_scores': category_summary,
        'category_scores_ignore_parallel': category_summary_ignore_parallel
    }

    results_summary = {
        'evaluation_summary': evaluation_summary,
        'detailed_results': detailed_results
    }
    
    # Remove old category_scores from evaluation_summary if it exists (now included above)
    
    os.makedirs("results", exist_ok=True)
    save_data(output_path, results_summary)
    
    print(f"💾 Detailed results saved to '{output_path}'")
    
    return model_score
