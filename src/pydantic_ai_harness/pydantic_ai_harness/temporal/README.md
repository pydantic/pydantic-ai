# Temporal Durability

`TemporalDurability` makes an agent run durable on [Temporal](https://temporal.io). Run the agent
inside a Temporal workflow and its model requests, tool calls, and MCP server communication run as
Temporal activities, so a worker that crashes mid-run resumes where it left off. Outside a workflow
the capability is transparent.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/temporal/)

`pydantic_ai_harness.temporal` is the import path for Pydantic AI's Temporal integration. The code
lives in [`pydantic_ai.durable_exec.temporal`](https://github.com/pydantic/pydantic-ai/tree/main/pydantic_ai_slim/pydantic_ai/durable_exec/temporal/)
until Pydantic AI v3, when it moves here. Both paths give the same objects, so imports from
`pydantic_ai.durable_exec.temporal` keep working until v3, and switching between them doesn't change
any recorded activity name. The deprecated `TemporalAgent` is not re-exported here; it stays
importable from the core path until v3.

## Installation

uv:

```bash
uv add "pydantic-ai-harness[temporal]" "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-harness[temporal]" "pydantic-ai-slim[openai]"
```

## Quick start

Give the agent a `name`, list it on a workflow, and connect with `PydanticAIPlugin`, which registers
the agent's activities with the worker:

```python {test="skip"}
from temporalio import workflow

from pydantic_ai import Agent
from pydantic_ai_harness.temporal import PydanticAIWorkflow, TemporalDurability

agent = Agent('openai:gpt-5', name='geography', capabilities=[TemporalDurability()])


@workflow.defn
class GeographyWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await agent.run(prompt)
        return result.output
```

The [Temporal docs](https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/) cover
the worker and client setup, activity configuration, streaming, child workflows, and the rest of
the integration.
