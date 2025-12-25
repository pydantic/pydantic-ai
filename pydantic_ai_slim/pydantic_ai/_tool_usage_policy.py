from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolLimits:
    """Usage limits for an individual tool.

    This class defines constraints on how many times a specific tool can be called
    during an agent run. Set on individual tools via the `usage_limits` parameter
    when registering tools with `@agent.tool(usage_limits=...)`.

    These limits apply to the specific tool only and take precedence over any
    agent-level [`AgentToolPolicy`][pydantic_ai.AgentToolPolicy] settings.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai import ToolLimits

        agent = Agent('openai:gpt-4o')

        # Tool can only be called once per run
        @agent.tool(usage_limits=ToolLimits(max_uses=1))
        def get_secret_key() -> str:
            return 'secret-key-123'

        # Tool can be called up to 3 times per step, 10 times total
        @agent.tool(usage_limits=ToolLimits(max_uses=10, max_uses_per_step=3))
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

    partial_acceptance: bool = True
    """Whether this tool allows partial acceptance when its usage limits would be exceeded.

    When a model requests multiple calls to this tool in a single step, and those calls
    would exceed the tool's `max_uses` or `max_uses_per_step` limit:

    - `True` (default): Accept as many calls as allowed, reject the rest individually.
    - `False`: Reject ALL calls to this tool if not all of them can be accepted.
      Use this for tools that have transactional semantics or require all calls to
      succeed together for correct behavior.

    Note:
        This setting only takes effect if [`AgentToolPolicy.partial_acceptance`][pydantic_ai.AgentToolPolicy.partial_acceptance]
        is also `True`. The policy-level setting acts as a master switch—if the policy
        disables partial acceptance, no tool can be partially accepted regardless of
        this setting.

    Example:
        ```python
        # A tool that must process all items together or none at all
        @agent.tool(usage_limits=ToolLimits(max_uses=5, partial_acceptance=False))
        def batch_process(items: list[str]) -> str:
            # If the model tries to call this 7 times but only 5 are allowed,
            # all 7 calls will be rejected (not 5 accepted + 2 rejected)
            return process_batch(items)
        ```
    """


@dataclass
class AgentToolPolicy:
    """Agent-level usage policy for tool calls in a run.

    This class defines two types of constraints:

    1. **Aggregate limits** (`max_uses`, `max_uses_per_step`):
       These apply to the **total** number of tool calls across all tools combined.

    2. **Per-tool overrides** (`tool_usage_limits`): A dict mapping tool names to
       [`ToolLimits`][pydantic_ai.ToolLimits], allowing you to set specific
       limits for individual tools without modifying the tool definitions.

    Set on the [`Agent`][pydantic_ai.Agent] via the `tools_usage_policy` parameter
    or passed to `agent.run()` / `agent.run_sync()` / `agent.run_stream()`.

    The per-tool limits in `tool_usage_limits` are merged with limits defined directly
    on tools via `@agent.tool(usage_limits=...)`. See [`CombinedToolset`][pydantic_ai.toolsets.CombinedToolset]
    for details on how limits are merged.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai import AgentToolPolicy, ToolLimits

        # Agent can make at most 5 tool calls per step, 20 total (across ALL tools)
        agent = Agent(
            'openai:gpt-4o',
            tools_usage_policy=AgentToolPolicy(max_uses=20, max_uses_per_step=5)
        )

        # Set per-tool limits for specific tools
        policy = AgentToolPolicy(
            max_uses=50,  # Aggregate: max 50 tool calls total across all tools
            tool_usage_limits={
                'expensive_api_call': ToolLimits(max_uses=3),  # Per-tool: max 3 calls
                'cheap_lookup': ToolLimits(max_uses=100),  # Per-tool: max 100 calls
            }
        )
        ```

    Note:
        "Per step" refers to one iteration of: model request → tool calls → response.
        A typical agent run may have multiple steps if the model needs to call tools
        and then reason about the results before producing a final output.
    """

    tool_usage_limits: dict[str, ToolLimits] = field(default_factory=dict)
    """Per-tool usage limits, merged with any limits defined on the tools themselves.

    A mapping from tool name to [`ToolLimits`][pydantic_ai.ToolLimits].
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

    partial_acceptance: bool = True
    """Master switch controlling whether partial acceptance of tool calls is allowed.

    When a model requests more tool calls than the policy limits allow:

    - `True` (default): Accept as many calls as allowed by `max_uses` and `max_uses_per_step`,
      reject the rest individually. This is the "accept what we can" behavior.
    - `False`: Reject the ENTIRE batch of tool calls if not all of them can be accepted.
      This is the "all-or-nothing" behavior.

    This setting acts as a global master switch. When set to `False`, no partial acceptance
    occurs anywhere—neither at the aggregate level nor at the per-tool level (regardless of
    individual [`ToolLimits.partial_acceptance`][pydantic_ai.ToolLimits.partial_acceptance] settings).

    Example:
        ```python
        # With partial_acceptance=True (default):
        # If max_uses=4 and model requests 5 calls → 4 accepted, 1 rejected

        # With partial_acceptance=False:
        # If max_uses=4 and model requests 5 calls → all 5 rejected
        policy = AgentToolPolicy(max_uses=4, partial_acceptance=False)
        ```
    """
