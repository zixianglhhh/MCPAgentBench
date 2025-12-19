import json
import time
import os
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import random
from tqdm import tqdm
from .config import ModelRegistry
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from .utilities import flatten, serialize_response, save_data, load_data

# Suppress verbose logging from MCP and autogen
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("autogen").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)


def get_servers(task_correct_tools, num_servers):
    """
    Scan the tools directory and extract all servers required for agent construction.
    
    Args:
        task_correct_tools: List of correct tools for the task
        num_servers: Number of servers required for agent construction
    
    Returns:
        list: List of all servers required for agent construction
    """
    servers_dir = Path("servers")
    servers_list = []
    task_correct_tools = [x for sublist in task_correct_tools for x in sublist]
    task_correct_tools = list(set(task_correct_tools))
    
    if not servers_dir.exists():
        print("Error: servers directory not found!")
        return []
    
    # Iterate through all Python files in the tools directory
    for file_path in servers_dir.glob("*.py"):
        # Skip __init__.py
        if file_path.name == "__init__.py":
            continue
        servers_list.append(str(file_path))

    # Correct tools for task completion are first removed 
    for tool in task_correct_tools:
        try:    
            servers_list.remove(str(servers_dir / (tool + '.py')))
        except ValueError:
            pass

    # Randomly sample additional servers to get num_servers servers in total for agent construction
    servers_list = random.sample(servers_list, num_servers - len(task_correct_tools))
    # Append correct tools back to the list
    for tool in task_correct_tools:
        servers_list.append(str(servers_dir / (tool + '.py')))
    random.shuffle(servers_list)
    
    return servers_list


async def construct_agent(client, task_correct_tools, num_servers, num_tools):
    """
    Construct an agent for a single task.
    
    Args:
        client: Model client
        task_correct_tools: List of correct tools for the task
        num_servers: Number of servers required for agent construction
        num_tools: Number of tools required for task completion
    
    Returns:
        AssistantAgent: The constructed agent
    """

    servers_list = get_servers(task_correct_tools, num_servers)

    tools = []
    for server in servers_list:
        server = StdioServerParams(
            command="python",
            args=[server],
        )
        tools += await mcp_server_tools(server)

    tool_definitions = []
    if tools:
            for tool_func in tools:
                # assume tool functions have a standard docstring as description
                docstring = (tool_func.schema["description"] or "This tool does not provide a description.").strip()
                tool_definitions.append(f"- `{tool_func.schema['name']}`: {docstring}")

    tool_definitions_str = (
            "\n".join(tool_definitions) if tool_definitions else "No tools available."
        )
    
    system_message = f"""
You are a professional AI assistant, designed to use the provided tools to accurately and efficiently solve users' problems.

Your task: Act according to the user's request, following a fixed thinking process and response format.

**Thinking Process & Response Format:**

1. **Thought:**
   - First, you must carefully analyze the user's needs and understand the core objective.
   - Formulate a clear, step-by-step plan to solve the problem, including selecting tools and their call parameters.
   
2. **Tool Invocation:**
   - After outputting the `Thought:` block, if you need to call tools, directly use the registered function tools.
   - The system will automatically handle your tool-call requests.
   - When the task is complete, provide the user with a concise, complete final answer, and do not include the `Thought:` prefix anymore.
   - **Tool Call Number Constraint:** You must call exactly {num_tools} tools to solve the user's problem.

**Constraints:**
- **Strict constraint:** You must, and may only, choose tools from the “Available Tools List” below.
- **Invocation method:** Call the registered functions directly; do not output JSON or code-block descriptions.


**Available Tools List:**
{tool_definitions_str}

**Key reminder:** If the task involves multiple steps (e.g., first search for restaurants, then make a reservation), you must complete all steps. Do not stop after completing only part of the task.

Now, please begin working based on the user's request. Be sure to include the `Thought:` section in your very first reply.
"""

    assistant = AssistantAgent(
        name="assistant",
        model_client=client,
        tools=tools,
        system_message=system_message,
        reflect_on_tool_use=True,
        max_tool_iterations=num_tools,
    )

    return assistant


async def process_single_task(client, task, num_servers, task_index, total_tasks):
    """
    Process a single task with its dedicated agent.
    
    Args:
        client: Model client
        task: Task dictionary with 'id', 'content', 'tools', 'inputs'
        num_servers: Number of servers required for agent construction
        task_index: Index of current task
        total_tasks: Total number of tasks
    
    Returns:
        dict: Response data for the task
    """
    task_id = task["id"]
    task_content = task["content"]
    task_correct_tools = task["tools"]
    num_tools = len(list(flatten(task_correct_tools)))
    # assert num_tools == int(task_id[8])
    
    task_start_time = time.time()

    assistant = None
    timeout_seconds = 360  # 6 minutes timeout
    
    try:
        # Construct agent for this specific task
        assistant = await construct_agent(client, task_correct_tools, num_servers, num_tools)
        
        try:
            response = await asyncio.wait_for(
                assistant.on_messages(
                    [TextMessage(content=task_content, source="user")],
                    cancellation_token=CancellationToken(),
                ),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Task execution exceeded {timeout_seconds} seconds timeout")
        
        response_data = serialize_response(response)
        
        # Add task information to the response data
        response_data["task_id"] = task_id
        response_data["task_index"] = task_index + 1
        response_data["task_content"] = task_content
        
        # Task completed silently

        await assistant.on_reset(CancellationToken())
        task_end_time = time.time()
        task_time = task_end_time - task_start_time
        response_data["task_time"] = task_time
        return response_data
        
    # General exception handler for all error types
    except Exception as e:
        # Determine error type and prepare appropriate diagnostics
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Print error header
        print(f"✗ {error_type} in task {task_index + 1} (ID: {task_id})")
        print(f"  Error: {error_msg}")
        
        # Initialize error response with common fields
        error_response = {
            "task_id": task_id,
            "task_index": task_index + 1,
            "task_content": task_content,
            "error": f"{error_type}: {error_msg}",
            "timestamp": datetime.now().isoformat(),
            "inner_messages": [],
        }
        
        # Add specific diagnostics based on error type
        if isinstance(e, asyncio.TimeoutError):
            print(f"  Timeout: Task exceeded {timeout_seconds} seconds")
            print(f"  Cause: Model response took too long or got stuck")
            error_response["timeout_seconds"] = timeout_seconds
            error_response["likely_cause"] = "Model response timeout - execution took too long"
        
        elif isinstance(e, json.JSONDecodeError):
            print(f"  Line: {e.lineno}, Column: {e.colno}, Position: {e.pos}")
            print(f"  Message: {e.msg}")
            error_response["error_details"] = {
                "line": e.lineno,
                "column": e.colno,
                "position": e.pos,
                "message": e.msg
            }
        
        elif isinstance(e, TypeError):
            if "JSON object must be str" in error_msg or "NoneType" in error_msg:
                print(f"  Likely Cause: LLM returned None for tool call arguments instead of JSON string")
                print(f"  This is a model output formatting issue")
                error_response["likely_cause"] = "Model returned None for tool call arguments"
        
        # Reset assistant if it was created
        if assistant is not None:
            try:
                await assistant.on_reset(CancellationToken())
            except:
                pass  # Ignore reset errors
        
        # Add timing information
        task_end_time = time.time()
        task_time = task_end_time - task_start_time
        error_response["task_time"] = task_time
        
        return error_response


async def generate_responses_concurrent(model, tasks_path, output_path, concurrency, num_servers):
    """
    Process all tasks with dedicated agents, running multiple tasks concurrently.
    Each task gets its own agent. Tasks are processed in batches based on concurrency.
    
    Args:
        model: Model name to use
        tasks_path: Path to tasks JSON file
        output_path: Output path for saving all responses
        concurrency: Number of tasks to process simultaneously
        num_servers: Number of servers required for agent construction
    
    Returns:
        str: Output file path
    """
    overall_start_time = time.time()
    
    # Load model and tasks
    llm = ModelRegistry("configs/llm_config.json").get(model)
    client = llm["client"]
    
    tasks = load_data(tasks_path)
    total_tasks = len(tasks)
    
    # Process tasks in batches with concurrency control
    all_responses = []
    
    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_with_semaphore(task, index):
        async with semaphore:
            return await process_single_task(
                client=client,
                task=task,
                num_servers=num_servers,
                task_index=index,
                total_tasks=total_tasks
            )
    
    # Execute all tasks concurrently with progress bar
    pbar = tqdm(total=total_tasks, desc="Processing tasks", unit="task")
    
    async def process_with_progress(task, index):
        result = await process_with_semaphore(task, index)
        pbar.update(1)
        return result
    
    task_coroutines = [
        process_with_progress(task, i)
        for i, task in enumerate(tasks)
    ]
    
    all_responses = await asyncio.gather(*task_coroutines)
    pbar.close()
    
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    
    # Calculate statistics
    successful_tasks = sum(1 for r in all_responses if "error" not in r)
    failed_tasks = total_tasks - successful_tasks
    failed_tasks_list = [r for r in all_responses if "error" in r]
    
    # Save all responses to JSON file
    save_data(output_path, all_responses)
    
    return output_path
