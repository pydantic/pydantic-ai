from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolPolicy:
    """Policy for a single tool's usage.

    This class defines constraints on how many times a specific tool can be called
    during an agent run. Set on individual tools via the `usage_policy` parameter
    when registering tools with `@agent.tool(usage_policy=...)`.

    Example:
        ```python
        from pydantic_ai import Agent, ToolPolicy

        agent = Agent('test')

        # Tool can only be called once per run
        @agent.tool_plain(usage_policy=ToolPolicy(max_uses=1))
        def get_secret_key() -> str:
            return 'secret-key-123'

        # Tool can be called up to 3 times per step, 10 times total
        @agent.tool_plain(usage_policy=ToolPolicy(max_uses=10, max_uses_per_step=3))
        def search_database(query: str) -> str:
            return f'Results for {query}'
        ```

    Note:
        The `tools_use_counts` attribute on `RunContext` tracks successful use counts
        per tool (only tools that have been used appear in the dict).
    """

    max_uses: int | None = None
    """Maximum number of successful uses allowed for this tool across the entire agent run.

    Once reached, the tool will be excluded from the available tools list in subsequent
    steps. In the current step, further calls are rejected with an error message to
    the model.

    Set to `None` for unlimited (default).
    """

    max_uses_per_step: int | None = None
    """Maximum number of successful uses allowed for this tool within a single run step.

    A "step" is one iteration of: model request → tool calls → response. This counter
    resets at the start of each new step.

    Set to `None` for unlimited (default).
    """

    partial_acceptance: bool | None = None
    """Whether this tool allows partial acceptance when its usage limits would be exceeded.

    When a model requests multiple calls to this tool in a single step, and those calls
    would exceed the tool's `max_uses` or `max_uses_per_step` limit:

    - `True`: Accept as many calls as allowed, reject the rest individually.
    - `False`: Reject ALL calls to this tool if not all of them can be accepted.
      Use this for tools that have transactional semantics or require all calls to
      succeed together for correct behavior.
    - `None` (default): Inherit the default behavior (equivalent to `True`).

    Inheritance behavior:
        - If a tool has no `usage_policy` set, it inherits the policy-level
          `partial_acceptance` setting from [`ToolsPolicy`][pydantic_ai.ToolsPolicy].
        - If no policy is set either, the default `True` behavior is used.
        - The policy-level [`ToolsPolicy.partial_acceptance`][pydantic_ai.ToolsPolicy.partial_acceptance]
          acts as a master switch—if the policy has `partial_acceptance=False`, this per-tool
          setting has no effect and all calls will be rejected when limits are exceeded.

    Example:
        ```python
        from pydantic_ai import Agent, ToolPolicy

        agent = Agent('test')

        # A tool that must process all items together or none at all
        @agent.tool_plain(usage_policy=ToolPolicy(max_uses=5, partial_acceptance=False))
        def batch_process(items: list[str]) -> str:
            # If the model tries to call this 7 times but only 5 are allowed,
            # all 7 calls will be rejected (not 5 accepted + 2 rejected)
            return f'Processed {len(items)} items'
        ```
    """


@dataclass
class ToolsPolicy(ToolPolicy):
    """Policy for all tools in an agent run.

    This class extends [`ToolPolicy`][pydantic_ai.ToolPolicy] with:

    1. **Aggregate limits** (`max_uses`, `max_uses_per_step`):
       These apply to the **total** number of successful tool uses across all tools combined.

    2. **Per-tool overrides** (`per_tool`): A dict mapping tool names to
       [`ToolPolicy`][pydantic_ai.ToolPolicy], allowing you to set specific
       limits for individual tools without modifying the tool definitions.

    Set on the [`Agent`][pydantic_ai.Agent] via the `tools_policy` parameter
    or passed to `agent.run()` / `agent.run_sync()` / `agent.run_stream()`.

    The per-tool limits in `per_tool` are merged with limits defined directly
    on tools via `@agent.tool(usage_policy=...)`. See [`CombinedToolset`][pydantic_ai.toolsets.CombinedToolset]
    for details on how limits are merged.

    Example:
        ```python
        from pydantic_ai import Agent, ToolsPolicy, ToolPolicy

        # Agent can make at most 5 successful tool uses per step, 20 total (across ALL tools)
        agent = Agent(
            'openai:gpt-4o',
            tools_policy=ToolsPolicy(max_uses=20, max_uses_per_step=5)
        )

        # Set per-tool limits for specific tools
        policy = ToolsPolicy(
            max_uses=50,  # Aggregate: max 50 successful tool uses total across all tools
            per_tool={
                'expensive_api_call': ToolPolicy(max_uses=3),  # Per-tool: max 3 uses
                'cheap_lookup': ToolPolicy(max_uses=100),  # Per-tool: max 100 uses
            }
        )
        ```

    Note:
        "Per step" refers to one iteration of: model request → tool calls → response.
        A typical agent run may have multiple steps if the model needs to call tools
        and then reason about the results before producing a final output.
    """

    per_tool: dict[str, ToolPolicy] = field(default_factory=dict)
    """Per-tool usage policies, merged with any policies defined on the tools themselves.

    A mapping from tool name to [`ToolPolicy`][pydantic_ai.ToolPolicy].
    These policies are merged with any `usage_policy` defined directly on tools via
    `@agent.tool(usage_policy=...)`. See [`CombinedToolset`][pydantic_ai.toolsets.CombinedToolset]
    for the merging behavior.

    Note: These are **per-tool** limits (e.g., "tool X can be used at most 3 times"),
    not to be confused with the **aggregate** limits inherited from `ToolPolicy` (`max_uses`, etc.)
    which apply to the total number of successful uses across all tools.
    """
