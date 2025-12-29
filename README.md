# MCPAgentBench

MCPAgentBench is a comprehensive benchmarking framework for evaluating Large Language Models (LLMs) on their ability to use tools through the Model Context Protocol (MCP). The framework tests models across various task categories including single tool usage, parallel tool execution, sequential tool chains, and multi-tool scenarios, providing detailed metrics such as Task Finish Score (TFS) and Task Efficiency Finish Score (TEFS) to assess both task completion accuracy and execution efficiency.

## Project Structure

```
MCPAgentBench/
├── configs/                    # Configuration files
│   ├── config.json            # General configuration (concurrency, num_servers)
│   ├── llm_config.json        # LLM model configurations (API endpoints, keys)
│   └── evaluation_config.json # Evaluation settings (skip_input_tools list)
├── data/                       # Task datasets
│   ├── daytasks.json          # Daily task dataset
│   ├── protasks.json          # Professional task dataset
│   ├── tasks.json             # General task dataset
│   └── *_with_*_tools.json   # Task variants (single, parallel, sequential, multi-tool)
├── src/                        # Core source code
│   ├── config.py              # Model registry and configuration management
│   ├── experiment.py          # Main experiment execution logic
│   ├── evaluate.py            # Evaluation metrics (TFS, TEFS calculation)
│   ├── agenttest.py           # Agent testing utilities
│   └── utilities.py           # Helper functions (tool extraction, input comparison)
├── tools/                      # Analysis and visualization tools
│   ├── plot_tefs.py           # TEFS comparison chart generator
│   ├── plot_tfs.py            # TFS comparison chart generator
│   ├── plot_tefs_tfs.py       # Combined TEFS/TFS comparison chart
│   ├── plot_time_efficiency.py # Time efficiency visualization
│   ├── plot_token_efficiency.py # Token efficiency visualization
│   ├── plot_model_size_tefs.py # Model size vs TEFS line chart
│   ├── plot_tool_count_tefs.py # Tool count vs TEFS line chart
│   ├── plot_comparison.py     # General comparison chart
│   └── generate_score_tables.py # Score table generation
├── servers/                    # MCP server implementations
│   └── *.py                   # Individual tool server implementations
├── results/                    # Experiment results
│   ├── *_general_test_run*_results.json # Individual run results
│   └── tool*/                 # Results organized by tool count (tool10, tool20, etc.)
├── logs/                       # Execution logs
├── runbenchmark.py            # Main entry point for running benchmarks
├── run_one_model_experiment.sh # Script to run 4 runs for a single model
├── run_batch_experiment.sh    # Script to run experiments for multiple models
├── set_api_key.sh             # API key configuration script
└── requirements.txt           # Python dependencies
```

## Usage

### Prerequisites

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys:**
   Edit `set_api_key.sh` with your API keys, then source it:
   ```bash
   source set_api_key.sh
   ```
   
   Or manually set environment variables:
   ```bash
   export API_KEY=your_openai_api_key
   export ROUTER_API_KEY=your_openrouter_api_key
   export KIMI_API_KEY=your_kimi_api_key
   export DEEPSEEK_API_KEY=your_deepseek_api_key
   export QWEN_API_KEY=your_qwen_api_key
   export ZHIPU_API_KEY=your_zhipu_api_key
   ```

### Configuration

#### 1. General Configuration (`configs/config.json`)

Configure general experiment settings:

```json
{
    "concurrency": 2,
    "num_servers": 40
}
```

- `concurrency`: Number of concurrent requests during testing
- `num_servers`: Number of MCP servers to use for agent construction

#### 2. LLM Configuration (`configs/llm_config.json`)

Configure models to test. Each entry should include:

```json
{
    "model": "openai/gpt-5",
    "model_info": {
        "vision": true,
        "function_calling": true,
        "json_output": true,
        "family": "gpt-5",
        "structured_output": true
    },
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_env_name": "ROUTER_API_KEY"
}
```

- `model`: Model identifier (must match the name used in commands)
- `model_info`: Model capabilities and metadata
- `base_url`: API endpoint URL
- `api_key_env_name`: Environment variable name containing the API key

#### 3. Evaluation Configuration (`configs/evaluation_config.json`)

Configure tools to skip during input comparison:

```json
{
    "skip_input_tools": [
        "security_guidance",
        "search_news",
        "vector_search",
        ...
    ]
}
```

Tools in this list will only be checked for tool name matching, not input parameter matching.

### Running Experiments

#### Single Model Experiment (4 Runs)

Run a single model through 4 runs for statistical reliability:

```bash
./run_one_model_experiment.sh <model_name> [tasks_type] [concurrency] [num_servers]
```

**Arguments:**
- `model_name`: Model name to test (required, e.g., `gpt-4o-mini`, `gemini-2.5-flash`)
- `tasks_type`: Type of tasks (optional, default: `general_test`)
  - Options: `day`, `pro`, `general_test`
- `concurrency`: Number of concurrent requests (optional, uses config.json default)
- `num_servers`: Number of servers (optional, uses config.json default)

**Examples:**
```bash
./run_one_model_experiment.sh gpt-4o-mini
./run_one_model_experiment.sh gemini-2.5-flash general_test
./run_one_model_experiment.sh gpt-4o-mini general_test 5 20
```

#### Batch Experiments

Run multiple models sequentially:

```bash
./run_batch_experiment.sh
```

Edit the script to uncomment/add models you want to test.

#### Direct Python Execution

Run a single experiment directly:

```bash
python runbenchmark.py --model <model_name> --tasks_type <tasks_type> [--concurrency N] [--num_servers N] [--output_name <name>]
```

**Examples:**
```bash
python runbenchmark.py --model gpt-4o-mini --tasks_type general_test
python runbenchmark.py --model gemini-2.5-flash --tasks_type day --concurrency 3 --num_servers 30
```

### Generating Visualizations

After running experiments, generate comparison charts:

#### TEFS Comparison Chart
```bash
python tools/plot_tefs.py --results_dir results --output tefs_comparison.png
```

#### TFS Comparison Chart
```bash
python tools/plot_tfs.py --results_dir results --output tfs_comparison.png
```

#### Combined TEFS/TFS Comparison
```bash
python tools/plot_tefs_tfs.py --results_dir results --output tefs_tfs_comparison.png
```

#### Time Efficiency Chart
```bash
python tools/plot_time_efficiency.py --results_dir results --output time_efficiency.png
```

#### Token Efficiency Chart
```bash
python tools/plot_token_efficiency.py --results_dir results --output token_efficiency.png
```

#### Model Size vs TEFS (for Qwen models)
```bash
python tools/plot_model_size_tefs.py --results_dir results --output model_size_tefs.png
```

#### Tool Count vs TEFS
```bash
python tools/plot_tool_count_tefs.py --results_dir results --output tool_count_tefs.png
```

#### Generate Score Tables
```bash
python tools/generate_score_tables.py --results_dir results --output scores.md
```

### Understanding Results

Results are saved in `results/` directory as JSON files with the naming pattern:
- `{model_name}_general_test_run{1-4}_results.json` for individual runs
- Results include:
  - `evaluation_summary`: Overall scores (model_score, ignore_parallel_score, token/time statistics)
  - `detailed_results`: Per-task results with tool usage, inputs, and scores
  - `category_scores`: Breakdown by task category

**Key Metrics:**
- **TFS (Task Finish Score)**: Percentage of tasks completed correctly (tools and inputs match, ignoring execution order)
- **TEFS (Task Efficiency Finish Score)**: Percentage of tasks completed correctly with exact execution order matching
- **avg@4**: Average score across 4 runs for statistical reliability

