"""Code mode capability that runs model-written code in a persistent IPython kernel."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field

from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, ToolSearch
from pydantic_ai.tools import AgentDepsT, ToolSelector
from pydantic_ai_harness.ipython_code_mode._kernel import IPythonKernel
from pydantic_ai_harness.ipython_code_mode._toolset import IPythonCodeModeToolset


@dataclass
class IPythonCodeMode(AbstractCapability[AgentDepsT]):
    """Capability that gives the model a `run_code` tool backed by a persistent IPython kernel.

    !!! warning "Not a sandbox"
        Code runs as a full CPython process with your user's permissions: it can read and write
        files, reach the network, and start processes. Do not use this capability in a webserver,
        or anywhere it handles untrusted input, without a sandbox around the kernel. For production
        and web-facing agents, use the Monty-based [`CodeMode`][pydantic_ai_harness.CodeMode].

    Like `CodeMode`, it wraps the agent's eligible tools behind a single `run_code` tool: they
    become synchronous functions in the kernel namespace, and each call is dispatched back through
    Pydantic AI. Unlike `CodeMode`, the code can import any installed package, and variables,
    imports, functions, and loaded data carry over between calls for the whole run.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import IPythonCodeMode

    # Local agent only: model-written code runs with your permissions.
    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[IPythonCodeMode()])
    ```
    """

    tools: ToolSelector[AgentDepsT] = field(default='all')
    """Which tools to define as functions in the kernel.

    - `'all'` (default): every eligible regular tool the agent has.
    - `Sequence[str]`: only tools whose names are listed.
    - Callable `(ctx, tool_def) -> bool | Awaitable[bool]`: tools where the callable returns `True`.

    The rest stay visible to the model as normal tool calls, as do framework control tools, tools
    not yet revealed by tool search, native fallbacks, and other code-execution tools.
    """

    max_retries: int = 3
    """Maximum number of retries for the `run_code` tool; code that raises counts as one."""

    _: KW_ONLY

    max_tool_calls: int = 100
    """Maximum function calls one `run_code` call may make; calls past it raise `ToolCallError` in the kernel."""

    timeout: float | None = 60.0
    """Seconds one `run_code` call may run, not counting time spent in tool calls.

    Code still running then is interrupted with `KeyboardInterrupt`, and the kernel keeps its state.
    Code that ignores the interrupt for 5 more seconds gets its kernel shut down, and the next call
    starts a fresh one. `None` removes the limit.
    """

    kernel: IPythonKernel | None = None
    """A kernel to keep across agent runs, with its state.

    Unset, each run starts its own kernel when the model first runs code and shuts it down when the
    run ends. Pass an `IPythonKernel` to share one kernel across the runs of a conversation; you own
    it and shut it down.
    """

    def get_ordering(self) -> CapabilityOrdering:
        """Wrap around `ToolSearch` so that `search_tools` stays native, as `CodeMode` does."""
        return CapabilityOrdering(position='outermost', wraps=[ToolSearch])

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset, splitting it into native tools and kernel functions."""
        return IPythonCodeModeToolset(
            wrapped=toolset,
            tool_selector=self.tools,
            max_retries=self.max_retries,
            max_tool_calls=self.max_tool_calls,
            timeout=self.timeout,
            kernel=self.kernel,
            capability=self,
        )
