import argparse
import asyncio
from src import *


async def main():
    parser = argparse.ArgumentParser(description="Run MCP benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model name to test (e.g., gpt-4o-mini, gemini-2.5-flash)")
    parser.add_argument("--tasks_type", type=str, required=True, choices=["day", "pro", "general_test"], 
                       help="Type of tasks to test: day, pro, or general_test")
    parser.add_argument("--concurrency", type=int, default=None, 
                       help="Number of concurrent requests (default: from config.json)")
    parser.add_argument("--num_servers", type=int, default=None,
                       help="Number of servers for agent construction (default: from config.json)")
    parser.add_argument("--output_name", type=str, default=None,
                       help="Custom output file name (without extension). If not specified, will use timestamp-based naming.")
    
    args = parser.parse_args()
    
    await run_experiment(
        model=args.model,
        tasks_type=args.tasks_type,
        concurrency=args.concurrency,
        num_servers=args.num_servers,
        output_name=args.output_name
    )

if __name__ == "__main__":
    asyncio.run(main())
