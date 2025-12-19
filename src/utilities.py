import json
from typing import List, Dict, Any, Set
from datetime import datetime


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(file_path: str, data) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        print (f"Data saved to {file_path}!")


def serialize_response(response):
    """Serialize the response object to a JSON-serializable format"""
    def convert_to_serializable(obj, depth=0):
        # Prevent infinite recursion
        if depth > 50:
            return str(obj)
            
        try:
            if hasattr(obj, '__dict__'):
                # Convert objects with __dict__ to dictionaries
                result = {}
                for key, value in obj.__dict__.items():
                    if not key.startswith('_'):  # Skip private attributes
                        try:
                            result[key] = convert_to_serializable(value, depth + 1)
                        except Exception as e:
                            result[key] = f"<Error serializing: {str(e)}>"
                return result
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item, depth + 1) for item in obj]
            elif isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    try:
                        result[key] = convert_to_serializable(value, depth + 1)
                    except Exception as e:
                        result[key] = f"<Error serializing: {str(e)}>"
                return result
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, 'isoformat'):  # Handle other datetime-like objects
                return obj.isoformat()
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif hasattr(obj, '__str__'):
                return str(obj)
            else:
                return str(obj)
        except Exception as e:
            return f"<Serialization error: {str(e)}>"
    
    return convert_to_serializable(response)


def flatten(lst):
    """Flatten a nested list."""
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def extract_tools_and_inputs(response_data: Dict[str, Any]) -> tuple[List[List[str]], List[List[Dict[str, Any]]]]:
    """
    Extract tool calls and their inputs from a single response.
    Returns a tuple of (tools, inputs), where each group contains called tools/inputs.
    """
    tools = []
    inputs = []
    
    # Check inner_messages for ToolCallRequestEvent
    if 'inner_messages' in response_data:
        for message in response_data['inner_messages']:
            if message.get('type') == 'ToolCallRequestEvent':
                if 'content' in message and isinstance(message['content'], list):
                    # Extract all tools and inputs called in this single event (parallel execution)
                    tools_in_one_call = []
                    inputs_in_one_call = []
                    for tool_call in message['content']:
                        if 'name' in tool_call:
                            tools_in_one_call.append(tool_call['name'])
                            # Extract arguments
                            if 'arguments' in tool_call:
                                try:
                                    # Check if arguments is empty or whitespace before parsing
                                    args_str = tool_call['arguments']
                                    if args_str and isinstance(args_str, str) and args_str.strip():
                                        args = json.loads(args_str)
                                        inputs_in_one_call.append(args)
                                    else:
                                        # Empty or whitespace arguments
                                        inputs_in_one_call.append({})
                                except Exception as e:
                                    print(f"Warning: Failed to parse arguments for tool {tool_call.get('name', 'unknown')}: {str(e)}")
                                    print(f"  Arguments value: {repr(tool_call['arguments'])}")
                                    inputs_in_one_call.append({})
                            else:
                                inputs_in_one_call.append({})
                    
                    tools.append(tools_in_one_call)
                    inputs.append(inputs_in_one_call)
    
    return tools, inputs


def compare_inputs(actual_input, expected_input) -> bool:
    """
    Compare actual inputs with expected inputs.
    Returns True if they match, False otherwise.
    Parameter order within each input set does not matter.
    """
    # Convert values to strings for comparison to handle type differences
    def normalize_dict(d):
        return {k: str(v) for k, v in d.items()}

    actual_input = [normalize_dict(input) for input in actual_input]
    expected_input = [normalize_dict(input) for input in expected_input]

    actual_input_set = {frozenset(d.items()) for d in actual_input}
    expected_input_set = {frozenset(d.items()) for d in expected_input}
    
    # Compare the normalized dictionaries
    return actual_input_set == expected_input_set

def calculate_total_tokens(response_data: List[Dict[str, Any]]) -> tuple[int, int]:
    """Calculate total prompt and completion tokens from all response data."""
    total_prompt = 0
    total_completion = 0
    
    def extract_tokens_recursive(obj):
        nonlocal total_prompt, total_completion
        
        if isinstance(obj, dict):
            # Check if this is a models_usage object
            if 'models_usage' in obj:
                usage = obj['models_usage']
                if isinstance(usage, dict):
                    # Extract prompt and completion tokens
                    prompt_tokens = usage.get('prompt_tokens', '0')
                    completion_tokens = usage.get('completion_tokens', '0')
                    
                    # Convert to int, handling string values
                    try:
                        total_prompt += int(prompt_tokens) if prompt_tokens != 'None' else 0
                        total_completion += int(completion_tokens) if completion_tokens != 'None' else 0
                    except (ValueError, TypeError):
                        pass
            
            # Recursively search in all values
            for value in obj.values():
                extract_tokens_recursive(value)
                
        elif isinstance(obj, (list, tuple)):
            # Recursively search in all items
            for item in obj:
                extract_tokens_recursive(item)
    
    # Process all responses
    for response in response_data:
        if  isinstance(response['inner_messages'], list) and response['inner_messages']:
            extract_tokens_recursive(response)
    
    return total_prompt, total_completion


def calculate_total_time(response_data: List[Dict[str, Any]]) -> float:
    """Calculate total time from all response data (sum of all task times)."""
    total_time = 0
    for response in response_data:
        # Sum all task times, including failed tasks
        task_time = response.get('task_time', 0)
        if isinstance(task_time, (int, float)):
            total_time += task_time
    return total_time
