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
    """Whether to allow partial acceptance when usage limits would be exceeded.

    When a model requests multiple calls to this tool in a single step, and those calls
    would exceed the `max_uses` or `max_uses_per_step` limit:

    - `True`: Accept as many calls as allowed, reject the rest individually.
    - `False`: Reject ALL calls to this tool if the batch would exceed limits.
      Use this for tools with transactional semantics where all calls must succeed together.
    - `None` (default): Defaults to `True` (partial acceptance allowed).

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

    This class extends [`ToolPolicy`][pydantic_ai.ToolPolicy] to provide agent-wide controls:

    1. **Aggregate limits** (`max_uses`, `max_uses_per_step`):
       Apply to the **total** number of successful tool uses across all tools combined.

    2. **Master switch** (`partial_acceptance`):
       When set to `False`, acts as a master switch that disables partial acceptance
       for ALL tools, regardless of their individual settings.

    3. **Run-time per-tool overrides** (`per_tool`):
       Override tool-level settings at run-time without modifying tool code.
       Values in `per_tool` take precedence over settings defined on tools via
       `@agent.tool(usage_policy=...)`.

    Set on the [`Agent`][pydantic_ai.Agent] via the `tools_policy` parameter,
    or pass to `agent.run()` / `agent.run_sync()` / `agent.run_stream()` to override per-run.

    Example:
        ```python
        from pydantic_ai import Agent, ToolsPolicy, ToolPolicy

        # Agent can make at most 5 successful tool uses per step, 20 total (across ALL tools)
        agent = Agent(
            'openai:gpt-4o',
            tools_policy=ToolsPolicy(max_uses=20, max_uses_per_step=5)
        )

        # Override per-tool limits at run-time
        result = agent.run_sync(
            'Do something',
            tools_policy=ToolsPolicy(
                per_tool={
                    'expensive_api': ToolPolicy(max_uses=3),  # Override: max 3 uses
                }
            )
        )
        ```

    Note:
        "Per step" refers to one iteration of: model request → tool calls → response.
        A typical agent run may have multiple steps if the model needs to call tools
        and then reason about the results before producing a final output.
    """

    per_tool: dict[str, ToolPolicy] = field(default_factory=dict)
    """Run-time per-tool overrides that take precedence over tool-level settings.

    A mapping from tool name to [`ToolPolicy`][pydantic_ai.ToolPolicy].
    Values here override any `usage_policy` defined on tools via `@agent.tool(usage_policy=...)`,
    allowing you to adjust limits at run-time without modifying tool code.

    Note: These are **per-tool** limits (e.g., "tool X can be used at most 3 times"),
    not to be confused with the **aggregate** limits (`max_uses`, `max_uses_per_step`)
    which apply to the total number of successful uses across all tools.
    """
