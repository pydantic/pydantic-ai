"""CodeMode Examples.

These examples compare traditional tool calling vs code mode tool calling.
Each example shows the reduction in LLM roundtrips and token usage.

Available Examples:
    - pr_discussion.py: Analyze GitHub PRs (requires GITHUB_PERSONAL_ACCESS_TOKEN)
    - batch_operations.py: Create multiple calendar events (no external deps)
    - expense_analysis.py: Analyze team expenses with aggregation (no external deps)
    - file_processing.py: Process files with dynamic loops (no external deps)

Run any example:
    uv run -m pydantic_ai_examples.code_mode.<example_name>

Key Benefits Demonstrated:
    1. Batch Operations: Code mode uses loops to batch multiple tool calls
    2. Data Aggregation: Code mode processes data in code, returning only summaries
    3. Dynamic Iteration: Code mode handles unknown iteration counts naturally
"""
