---
title: Run an agent as a durable workflow
description: Make model and tool calls replay-safe so long-running work survives worker restarts.
---

# Run an agent as a durable workflow

Attach `TemporalDurability` and invoke the agent inside a workflow when the whole run must survive process or infrastructure failure.

```bash
pip/uv-add "pydantic-ai[temporal]"
```

```python {test="skip"}
from temporalio import workflow

from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch, WebSearch
from pydantic_ai.durable_exec.temporal import PydanticAIWorkflow, TemporalDurability

agent = Agent(
    'openai:gpt-5.6-sol',
    name='release_researcher',
    instructions='Research the release, preserve source links, and return a concise brief.',
    capabilities=[WebSearch(), WebFetch(), TemporalDurability()],
)


@workflow.defn
class ReleaseResearchWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, release: str) -> str:
        result = await agent.run(f'Research the operational risks in {release}.')
        return result.output
```

Durability comes from starting this workflow through a Temporal client and worker; attaching the capability alone does not persist an ordinary `agent.run()`. Keep the agent name and toolset IDs stable because they become part of replay history.
