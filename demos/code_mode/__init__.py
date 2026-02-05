"""CodeMode Demos.

These demos compare traditional tool calling vs code mode tool calling.
Each demo shows the reduction in LLM roundtrips and token usage.

Available Demos:
    - pr_discussion_demo.py: Analyze GitHub PRs (requires GITHUB_PERSONAL_ACCESS_TOKEN)
    - batch_operations_demo.py: Create multiple calendar events (no external deps)
    - expense_analysis_demo.py: Analyze team expenses with aggregation (no external deps)
    - file_processing_demo.py: Process files with dynamic loops (no external deps)
    - sql_analysis_demo.py: Regional sales SQL analysis with N+1 pattern (no external deps)

Run any demo:
    uv run python demos/code_mode/<demo_name>.py

Key Benefits Demonstrated:
    1. Batch Operations: Code mode uses loops to batch multiple tool calls
    2. Data Aggregation: Code mode processes data in code, returning only summaries
    3. Dynamic Iteration: Code mode handles unknown iteration counts naturally
    4. N+1 Query Pattern: Code mode collapses nested queries into single loops
"""
