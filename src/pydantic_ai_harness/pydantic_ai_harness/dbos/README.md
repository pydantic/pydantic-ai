# DBOS Durability

`DBOSDurability` makes an agent run durable on [DBOS](https://www.dbos.dev). Run the agent inside a
DBOS workflow and its model requests and MCP communication run as DBOS steps, so a run that crashes
mid-way recovers from its last completed step. Outside a workflow the capability is transparent.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/dbos/)

`pydantic_ai_harness.dbos` is the import path for Pydantic AI's DBOS integration. The code lives in
[`pydantic_ai.durable_exec.dbos`](https://github.com/pydantic/pydantic-ai/tree/main/pydantic_ai_slim/pydantic_ai/durable_exec/dbos/)
until Pydantic AI v3, when it moves here. Both paths give the same objects, so imports from
`pydantic_ai.durable_exec.dbos` keep working until v3, and switching between them doesn't change any
recorded workflow or step name. The deprecated `DBOSAgent` is not re-exported here; it stays
importable from the core path until v3.

## Installation

uv:

```bash
uv add "pydantic-ai-harness[dbos]" "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-harness[dbos]" "pydantic-ai-slim[openai]"
```

## Quick start

Give the agent a `name`, construct it before `DBOS.launch()`, and call `agent.run()` inside your own
`@DBOS.workflow`:

```python {test="skip"}
from dbos import DBOS, DBOSConfig

from pydantic_ai import Agent
from pydantic_ai_harness.dbos import DBOSDurability

dbos_config: DBOSConfig = {'name': 'geography', 'system_database_url': 'sqlite:///dbos.sqlite'}
DBOS(config=dbos_config)

agent = Agent('openai:gpt-5', name='geography', capabilities=[DBOSDurability()])


@DBOS.workflow()
async def answer(question: str) -> str:
    result = await agent.run(question)
    return result.output
```

The [DBOS docs](https://pydantic.dev/docs/ai/capabilities/durable_execution/dbos/) cover tools with
side effects, step configuration, child workflows, streaming, and the rest of the integration.
