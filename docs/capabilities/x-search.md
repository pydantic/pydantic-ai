# X Search

The [`XSearch`][pydantic_ai.capabilities.XSearch] [capability](overview.md) gives your agent search over X (Twitter) posts. It's a [provider-adaptive tool](overview.md#provider-adaptive-tools) backed by [`XSearchTool`][pydantic_ai.native_tools.XSearchTool] on the native side — see [X Search Tool](../native-tools.md#x-search-tool) for configuration options.

Unlike [Web Search](web-search.md) and [Web Fetch](web-fetch.md), there is no default non-xAI fallback: X search is only available natively on xAI models. If your agent is not running on an xAI model, set `fallback_model` explicitly to an xAI model that supports [`XSearchTool`][pydantic_ai.native_tools.XSearchTool], and search requests are delegated to that model as a subagent tool:

```python {title="x_search.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[XSearch(fallback_model='xai:grok-4.3')],
)
```

`native=` can be a factory: a callable taking [`RunContext`][pydantic_ai.tools.RunContext] that returns [`XSearchTool`][pydantic_ai.native_tools.XSearchTool] or `None` — [`XSearchNativeTool`][pydantic_ai.common_tools.x_search.XSearchNativeTool] is the type the `fallback_model` subagent accepts. It resolves on each model request, and again when the subagent runs — so keep it free of one-shot side effects. Both resolutions belong to the same run and carry the same `deps`, but they do not share a [`RunContext`][pydantic_ai.tools.RunContext]: the subagent resolves from its own tool call, so `tool_call_id` and `tool_name` name that call instead of being `None`, and `messages` holds the run so far. Read `ctx.deps` for configuration that has to match across both. On the subagent, capability-level fields such as `include_output` override the factory result. See [Dynamic Configuration](../native-tools.md#dynamic-configuration).

!!! note "A factory returning `None`"
    `None` omits the tool for that request only when no `fallback_model` is set. With a `fallback_model`, the subagent tool is offered to the model whenever the factory returns `None` — even on a model that supports [`XSearchTool`][pydantic_ai.native_tools.XSearchTool] natively — and calling it raises [`UserError`][pydantic_ai.exceptions.UserError] rather than searching X with default settings.

!!! note "Durable execution with Temporal"
    The `fallback_model` subagent's tool call runs inside a Temporal activity, so a `native=` factory resolved there receives the limited [`TemporalRunContext`][pydantic_ai.durable_exec.temporal.TemporalRunContext]: reading `ctx.messages`, or any other field Temporal does not carry, raises a [`UserError`][pydantic_ai.exceptions.UserError]. `ctx.deps` does cross the boundary, so keep the factory reading `deps` only. See [Agent Run Context and Dependencies](../durable_execution/temporal.md#agent-run-context-and-dependencies) for the fields that are available. [DBOS](../durable_execution/dbos.md) and [Prefect](../durable_execution/prefect.md) pass the live `RunContext` and are unaffected.
