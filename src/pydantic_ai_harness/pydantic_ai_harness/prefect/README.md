# Prefect Durability

`PrefectDurability` makes an agent run durable on [Prefect](https://www.prefect.io). Run the agent
inside a Prefect flow and its model requests, tool calls, and MCP communication run as Prefect tasks,
so a retry of the same flow run loads the results of tasks that already completed instead of running
them again. Outside a flow the capability is transparent.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/prefect/)

`pydantic_ai_harness.prefect` is the import path for Pydantic AI's Prefect integration. The code
lives in [`pydantic_ai.durable_exec.prefect`](https://github.com/pydantic/pydantic-ai/tree/main/pydantic_ai_slim/pydantic_ai/durable_exec/prefect/)
until Pydantic AI v3, when it moves here. Both paths give the same objects, so imports from
`pydantic_ai.durable_exec.prefect` keep working until v3, and switching between them doesn't change
any recorded flow or task name. The deprecated `PrefectAgent`, `PrefectFunctionToolset`, and
`PrefectMCPToolset` are not re-exported here; they stay importable from the core path until v3.

## Installation

uv:

```bash
uv add "pydantic-ai-harness[prefect]" "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-harness[prefect]" "pydantic-ai-slim[openai]"
```

## Quick start

Give the agent a `name` and call `agent.run()` inside your own `@flow`:

```python {test="skip"}
from prefect import flow

from pydantic_ai import Agent
from pydantic_ai_harness.prefect import PrefectDurability

agent = Agent('openai:gpt-5', name='geography', capabilities=[PrefectDurability()])


@flow
async def answer(question: str) -> str:
    result = await agent.run(question)
    return result.output
```

The [Prefect docs](https://pydantic.dev/docs/ai/capabilities/durable_execution/prefect/) cover task
configuration, caching, streaming, and the rest of the integration.
