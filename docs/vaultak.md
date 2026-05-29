# Runtime Security with Vaultak

[Vaultak](https://vaultak.com) is a runtime security platform for AI agents. It intercepts every
tool call in real time, scores risk on a 0–10 scale, enforces your policy rules, masks PII in
outputs, and blocks dangerous actions before they reach your production systems.

Vaultak integrates with Pydantic AI through the native
[`Hooks`][pydantic_ai.capabilities.Hooks] system — no subclassing required for quick setup, and
a clean [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] subclass available
for reusable, packaged deployments.

!!! tip "Pydantic AI Hooks"
    Vaultak uses [`before_tool_execute`][pydantic_ai.capabilities.AbstractCapability.before_tool_execute]
    to risk-score and block tool calls,
    [`after_tool_execute`][pydantic_ai.capabilities.AbstractCapability.after_tool_execute]
    to mask PII in outputs, and
    [`on_tool_execute_error`][pydantic_ai.capabilities.AbstractCapability.on_tool_execute_error]
    to alert on failures. See [Hooks](hooks.md) for the full lifecycle.

## Install

```bash
pip install vaultak pydantic-ai
```

Sign up at [vaultak.com](https://vaultak.com) to get your API key (starts with `vtk_`).

## Quick start with `Hooks`

The fastest way to add Vaultak to an existing agent is to register hook functions on a
[`Hooks`][pydantic_ai.capabilities.Hooks] instance and pass it as a capability.
Every tool call the agent makes is intercepted before and after execution.

```python {title="vaultak_hooks.py" test="skip"}
import asyncio
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from vaultak import Vaultak

vt = Vaultak(api_key="vtk_...", agent_name="my-agent")
RISK_THRESHOLD = 7.0

hooks = Hooks()


@hooks.on.before_tool_execute
async def vaultak_score(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Risk-score the tool call; block if score meets or exceeds the threshold."""
    result = await asyncio.to_thread(
        vt.score_action, action=call.tool_name, context=args
    )
    if result.score >= RISK_THRESHOLD:
        # SkipToolExecution returns the message to the LLM as the tool's result
        raise SkipToolExecution(
            f"[Vaultak] Tool '{call.tool_name}' blocked — risk score "
            f"{result.score:.1f}/10 meets or exceeds threshold {RISK_THRESHOLD}. "
            "Review at app.vaultak.com"
        )
    await asyncio.to_thread(vt.check_policy, tool_name=call.tool_name, input_data=str(args))
    return args


@hooks.on.after_tool_execute
async def vaultak_mask_pii(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
    result: Any,
) -> Any:
    """Mask PII in tool output before it reaches the model."""
    if isinstance(result, str):
        return await asyncio.to_thread(vt.mask_pii, result)
    return result


agent = Agent("openai:gpt-4o", capabilities=[hooks])

result = agent.run_sync(
    "Look up customer record #42 and email the summary to alice@example.com"
)
print(result.output)
```

## Reusable `AbstractCapability`

For production deployments or shared libraries, subclass
[`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] to bundle configuration
state with the hook logic. The same class can then be shared across agents, configured per
environment, and tested in isolation.

```python {title="vaultak_capability.py" test="skip"}
import asyncio
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from vaultak import Vaultak


@dataclass
class VaultakSecurity(AbstractCapability[Any]):
    """Runtime security capability — risk-scores tool calls and masks PII in outputs."""

    api_key: str
    agent_name: str = "pydantic-ai-agent"
    risk_threshold: float = 7.0
    verbose: bool = False
    _vt: Vaultak = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._vt = Vaultak(api_key=self.api_key, agent_name=self.agent_name)

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(
            self._vt.score_action, action=call.tool_name, context=args
        )
        if self.verbose:
            print(f"[Vaultak] {call.tool_name}: risk {result.score:.1f}/10")
        if result.score >= self.risk_threshold:
            raise SkipToolExecution(
                f"[Vaultak] Tool '{call.tool_name}' blocked — risk score "
                f"{result.score:.1f}/10 meets or exceeds threshold {self.risk_threshold}. "
                "Review at app.vaultak.com"
            )
        await asyncio.to_thread(
            self._vt.check_policy, tool_name=call.tool_name, input_data=str(args)
        )
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        if isinstance(result, str):
            return await asyncio.to_thread(self._vt.mask_pii, result)
        return result

    async def on_tool_execute_error(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        # Alert Vaultak dashboard, then re-raise so the agent can retry normally
        await asyncio.to_thread(
            self._vt.alert,
            level="error",
            message=f"Tool '{call.tool_name}' raised {type(error).__name__}: {error}",
        )
        raise error


# Attach to any agent — all tool calls are now monitored
agent = Agent(
    "openai:gpt-4o",
    capabilities=[
        VaultakSecurity(
            api_key="vtk_...",
            agent_name="production-agent",
            risk_threshold=7.0,
            verbose=True,
        )
    ],
)
```

!!! note "Stricter threshold for sensitive agents"
    For agents with access to databases, payment systems, or external APIs, lower
    `risk_threshold` to `5.0` to block anything above medium risk:

    ```python
    VaultakSecurity(api_key="vtk_...", agent_name="finance-agent", risk_threshold=5.0)  # noqa: F821
    ```

## Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | — | Your Vaultak API key (starts with `vtk_`) — required |
| `agent_name` | `str` | `"pydantic-ai-agent"` | Label for this agent in the Vaultak dashboard |
| `risk_threshold` | `float` | `7.0` | Score (0–10) at or above which tool calls are blocked |
| `verbose` | `bool` | `False` | Print each risk score to stdout |

## What gets monitored

| Pydantic AI event | Vaultak action |
|---|---|
| Tool call selected by model (`before_tool_execute`) | Risk-scores the call; blocks via `SkipToolExecution` if score ≥ threshold; checks policy rules |
| Tool output returned (`after_tool_execute`) | Scans for PII and masks before result reaches the model |
| Tool raises an exception (`on_tool_execute_error`) | Sends an error alert to the Vaultak dashboard, then re-raises |

## Links

- [Vaultak documentation](https://docs.vaultak.com)
- [PyPI: `vaultak`](https://pypi.org/project/vaultak)
- [GitHub](https://github.com/vaultak/vaultak-python)
- [Dashboard](https://app.vaultak.com)
- [Hooks reference](hooks.md)
- [Capabilities reference](capabilities.md)
