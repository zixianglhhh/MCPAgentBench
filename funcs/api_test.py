#!/usr/bin/env python3
"""
API connectivity test script.
Tests the connection to the LLM API with a simple "hello" message.
Reads model configuration from configs/llm_config.json.
"""

import os
import sys
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_model_config(model_name: str, config_path: str = "configs/llm_config.json"):
    """
    Load model configuration from llm_config.json.
    
    Args:
        model_name: Name of the model to find
        config_path: Path to the llm_config.json file
        
    Returns:
        dict: Model configuration with base_url, api_key_env_name, etc.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_list = json.load(f)
    
    # Find the model in the config
    for cfg in config_list:
        if cfg.get("model") == model_name:
            return cfg
    
    # If not found, raise error
    available_models = [cfg.get("model") for cfg in config_list]
    raise ValueError(
        f"Model '{model_name}' not found in {config_path}\n"
        f"Available models: {', '.join(available_models)}"
    )


def test_api_connectivity(model_name: str, config_path: str = "configs/llm_config.json"):
    """
    Test API connectivity by sending a simple "hello" message.
    Reads model configuration from llm_config.json.
    
    Args:
        model_name: Name of the model to test
        config_path: Path to the llm_config.json file
    """
    # Load model configuration
    try:
        model_config = load_model_config(model_name, config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Extract configuration
    base_url = model_config.get("api_base") or model_config.get("base_url") or "https://jacob.api-store.store/v1"
    api_key_env_name = model_config.get("api_key_env_name", "API_KEY")
    model_info = model_config.get("model_info", {})
    
    # Get API key from environment variable
    api_key = os.getenv(api_key_env_name)
    
    if not api_key:
        print(f"Error: {api_key_env_name} environment variable is not set!")
        print(f"Please set it using: export {api_key_env_name}='your-api-key'")
        sys.exit(1)
    
    print(f"Testing API connectivity...")
    print(f"Model: {model_name}")
    print(f"Base URL: {base_url}")
    print(f"Environment Variable: {api_key_env_name}")
    if model_info:
        print(f"Model Info: {json.dumps(model_info, indent=2)}")
    print(f"API Key: {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '****'}")
    print("-" * 50)
    
    try:
        # Initialize OpenAI client
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Send a simple test message
        print("Sending test message: 'hello'")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "hello"}
            ]
        )
        
        # Extract and print the response
        output_content = response.choices[0].message.content
        print("\n" + "=" * 50)
        print("✅ API Test Successful!")
        print("=" * 50)
        print(f"Response from {model_name}:")
        print(output_content)
        print("=" * 50)
        
        # Print usage information if available
        if hasattr(response, 'usage') and response.usage:
            print(f"\nToken Usage:")
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Completion tokens: {response.usage.completion_tokens}")
            print(f"  Total tokens: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 50)
        print("❌ API Test Failed!")
        print("=" * 50)
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("=" * 50)
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python api_test.py <model_name> [config_path]")
        print("\nArguments:")
        print("  model_name  : Name of the model to test (required)")
        print("                Must match a model in configs/llm_config.json")
        print("  config_path : Path to llm_config.json (optional, default: configs/llm_config.json)")
        print("\nExamples:")
        print("  python api_test.py gpt-4o-mini")
        print("  python api_test.py gemini-2.5-flash")
        print("  python api_test.py openai/gpt-5")
        print("\nNote: Model configuration (base_url, api_key_env_name, etc.)")
        print("      will be read from configs/llm_config.json")
        sys.exit(1)
    
    model_name = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "configs/llm_config.json"
    
    success = test_api_connectivity(model_name, config_path)
    sys.exit(0 if success else 1)

