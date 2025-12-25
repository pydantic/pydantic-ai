from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolUsageLimits:
    """Usage limits for an individual tool.

    This class defines constraints on how many times a specific tool can be called
    during an agent run. Set on individual tools via the `usage_limits` parameter
    when registering tools with `@agent.tool(usage_limits=...)`.

    These limits apply to the specific tool only and take precedence over any
    agent-level [`ToolsUsagePolicy`][pydantic_ai.ToolsUsagePolicy] settings.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai import ToolUsageLimits

        agent = Agent('openai:gpt-4o')

        # Tool can only be called once per run
        @agent.tool(usage_limits=ToolUsageLimits(max_uses=1))
        def get_secret_key() -> str:
            return 'secret-key-123'

        # Tool can be called up to 3 times per step, 10 times total
        @agent.tool(usage_limits=ToolUsageLimits(max_uses=10, max_uses_per_step=3))
        def search_database(query: str) -> str:
            return f'Results for {query}'
        ```

    Note:
        The `tools_use_counts` attribute on `RunContext` tracks successful use counts
        per tool (only tools that have been used appear in the dict).
    """

    max_uses: int | None = None
    """Maximum number of successful calls allowed for this tool across the entire agent run.

    Once reached, the tool will be excluded from the available tools list and any
    further calls will return an error message to the model.

    Set to `None` for unlimited (default).
    """

    max_uses_per_step: int | None = None
    """Maximum number of successful calls allowed for this tool within a single run step.

    A "step" is one iteration of: model request → tool calls → response. This counter
    resets at the start of each new step.

    Set to `None` for unlimited (default).
    """

    min_uses: int | None = None
    """Minimum number of calls required for this tool before the agent run can complete.

    If the agent tries to produce a final output before meeting this minimum, additional
    tool calls will be required.

    Set to `None` for no minimum (default).
    """

    min_uses_per_step: int | None = None
    """Minimum number of calls required for this tool within each run step.

    Set to `None` for no minimum (default).
    """


@dataclass
class ToolsUsagePolicy:
    """Agent-level usage policy for tool calls in a run.

    This class defines two types of constraints:

    1. **Aggregate limits** (`max_uses`, `max_uses_per_step`, `min_uses`, `min_uses_per_step`):
       These apply to the **total** number of tool calls across all tools combined.

    2. **Per-tool overrides** (`tool_usage_limits`): A dict mapping tool names to
       [`ToolUsageLimits`][pydantic_ai.ToolUsageLimits], allowing you to set specific
       limits for individual tools without modifying the tool definitions.

    Set on the [`Agent`][pydantic_ai.Agent] via the `tools_usage_policy` parameter
    or passed to `agent.run()` / `agent.run_sync()` / `agent.run_stream()`.

    The per-tool limits in `tool_usage_limits` are merged with limits defined directly
    on tools via `@agent.tool(usage_limits=...)`. See [`CombinedToolset`][pydantic_ai.toolsets.CombinedToolset]
    for details on how limits are merged.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai import ToolsUsagePolicy, ToolUsageLimits

        # Agent can make at most 5 tool calls per step, 20 total (across ALL tools)
        agent = Agent(
            'openai:gpt-4o',
            tools_usage_policy=ToolsUsagePolicy(max_uses=20, max_uses_per_step=5)
        )

        # Set per-tool limits for specific tools
        policy = ToolsUsagePolicy(
            max_uses=50,  # Aggregate: max 50 tool calls total across all tools
            tool_usage_limits={
                'expensive_api_call': ToolUsageLimits(max_uses=3),  # Per-tool: max 3 calls
                'cheap_lookup': ToolUsageLimits(max_uses=100),  # Per-tool: max 100 calls
            }
        )
        ```

    Note:
        "Per step" refers to one iteration of: model request → tool calls → response.
        A typical agent run may have multiple steps if the model needs to call tools
        and then reason about the results before producing a final output.
    """

    tool_usage_limits: dict[str, ToolUsageLimits] = field(default_factory=dict)
    """Per-tool usage limits, merged with any limits defined on the tools themselves.

    A mapping from tool name to [`ToolUsageLimits`][pydantic_ai.ToolUsageLimits].
    These limits are merged with any `usage_limits` defined directly on tools via
    `@agent.tool(usage_limits=...)`. See [`CombinedToolset`][pydantic_ai.toolsets.CombinedToolset]
    for the merging behavior.

    Note: These are **per-tool** limits (e.g., "tool X can be called at most 3 times"),
    not to be confused with the **aggregate** limits on this class (`max_uses`, etc.)
    which apply to the total number of calls across all tools.
    """

    max_uses: int | None = None
    """Maximum total number of successful tool calls allowed across all tools for the entire run.

    This is an aggregate limit—once the total number of tool calls across all tools
    reaches this limit, no more tools can be called and the model will be prompted
    to produce a final output.

    Set to `None` for unlimited (default).
    """

    max_uses_per_step: int | None = None
    """Maximum total number of successful tool calls allowed across all tools within a single step.

    This counter resets at the start of each new step.

    Set to `None` for unlimited (default).
    """

    min_uses: int | None = None
    """Minimum total number of tool calls required across all tools before the run can complete.

    Set to `None` for no minimum (default).
    """

    min_uses_per_step: int | None = None
    """Minimum total number of tool calls required within each run step.

    Set to `None` for no minimum (default).
    """
