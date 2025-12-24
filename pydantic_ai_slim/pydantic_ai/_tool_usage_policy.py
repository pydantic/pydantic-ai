from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolUsageLimits:
    """Usage limits for an individual tool.

    This class defines constraints on how many times a specific tool can be called
    during an agent run. It is set on the `Tool` itself via the `usage_limits` parameter.

    Attributes:
        max_uses: Maximum number of successful calls allowed for this tool across
            the entire agent run. Once reached, the tool will be excluded from
            the available tools list. Set to `None` for unlimited (default).
        max_uses_per_step: Maximum number of successful calls allowed for this tool
            within a single run step (one model request → tool calls → response cycle).
            Resets at the start of each new step. Set to `None` for unlimited (default).
        min_uses: Minimum number of calls required for this tool before the agent
            run can complete successfully. Set to `None` for no minimum (default).
        min_uses_per_step: Minimum number of calls required for this tool within
            each run step. Set to `None` for no minimum (default).

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai._tool_usage_policy import ToolUsageLimits

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
    max_uses_per_step: int | None = None
    min_uses: int | None = None
    min_uses_per_step: int | None = None


@dataclass
class AgentToolsPolicy:
    """Agent-level usage policy applied to all tools in a run.

    This class defines constraints that apply collectively to all tools during
    an agent run. It is set on the `Agent` and affects every tool unless
    individual tools have their own `ToolUsageLimits` that override these defaults.

    Attributes:
        max_uses: Maximum total number of successful tool calls allowed across
            all tools for the entire agent run. Set to `None` for unlimited (default).
        max_uses_per_step: Maximum total number of successful tool calls allowed
            across all tools within a single run step. Resets at each new step.
            Set to `None` for unlimited (default).
        min_uses: Minimum total number of tool calls required across all tools
            before the agent run can complete. Set to `None` for no minimum (default).
        min_uses_per_step: Minimum total number of tool calls required within
            each run step. Set to `None` for no minimum (default).

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai._tool_usage_policy import AgentToolsPolicy

        # Agent can make at most 5 tool calls per step, 20 total
        agent = Agent(
            'openai:gpt-4o',
            tools_policy=AgentToolsPolicy(max_uses=20, max_uses_per_step=5)
        )
        ```

    Note:
        - Individual tool limits (`ToolUsageLimits`) take precedence over agent-level
          policies for that specific tool.
        - "Per step" refers to one iteration of: model request → tool calls → response.
          A typical agent run may have multiple steps if the model needs to call tools
          and then reason about the results before producing a final output.
    """

    max_uses: int | None = None
    max_uses_per_step: int | None = None
    min_uses: int | None = None
    min_uses_per_step: int | None = None
