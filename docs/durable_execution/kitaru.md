---
description: "Record Pydantic AI agent runs to Kitaru and replay them against new prompts, models or tools. Kitaru's current adapter records and replays runs; it does not resume a crashed run."
---

# Recording and Replay with Kitaru

[Kitaru](https://docs.zenml.io/kitaru) records agent runs as sessions and replays them against your next change, so you can test a new prompt, model or tool against what the agent already did. Its Pydantic AI adapter is the separate [`kitaru-pydantic-ai`](https://pypi.org/project/kitaru-pydantic-ai/) package, maintained by the Kitaru team, rather than part of `pydantic_ai.durable_exec`.

!!! warning "Kitaru no longer provides durable execution"
    Kitaru 0.21 and earlier shipped a durable execution adapter at `kitaru.adapters.pydantic_ai`, with flows, checkpoints and `@hitl_tool`. Kitaru 0.22.0 removed it, along with the `kitaru[pydantic-ai]` extra. The current adapter records runs and replays them: if the process dies mid-run, the session stays `in_progress` with what was recorded so far, and the run is not resumed. For runs that survive crashes and restarts, use one of the other [durable execution engines](./overview.md).

## Record an agent

Install the adapter next to Kitaru:

```bash
uv add kitaru-pydantic-ai
```

The adapter supports a pinned range of `pydantic-ai-slim` versions that can lag the latest Pydantic AI release; check [its PyPI page](https://pypi.org/project/kitaru-pydantic-ai/) before upgrading.

Register the agent with `kitaru agent register`, then wrap your [`Agent`][pydantic_ai.agent.Agent] in `KitaruAgent` with the registered agent's ID:

```python {title="kitaru_agent.py" test="skip" lint="skip"}
import os
import uuid

from kitaru_pydantic_ai import KitaruAgent
from pydantic_ai import Agent

agent = Agent('{{ openai:small }}', name='support-agent')
support = KitaruAgent(agent, agent_id=uuid.UUID(os.environ['KITARU_AGENT_ID']))

result = support.run_sync('Refund order #4821.')
print(result.output)
```

`KitaruAgent` is a [`WrapperAgent`][pydantic_ai.agent.WrapperAgent]: `run`, `run_sync`, `iter`, tools, output types and capabilities behave as on the wrapped agent. Each run becomes a Kitaru session that records its model requests (messages, token usage and estimated cost) and tool calls (arguments and results). The adapter needs a reachable Kitaru server, found through `KITARU_API_URL` and an API token or `kitaru login` credentials; if the server is unreachable, the run fails at session creation instead of running unrecorded.

## Replay a run

You do not change the script to replay it. A Kitaru worker runs it as a replay task: the recorded inputs replace the prompt in your script, model overrides apply at each model request, and a tool policy decides whether each tool call returns its recorded result, returns a fixed value, or runs for real. When a replay cannot satisfy a tool call under its policy, the adapter raises `ToolPolicyMissError` (or `ToolPolicyError` for a recorded tool failure) from `kitaru_pydantic_ai`.

See the [Kitaru Pydantic AI adapter guide](https://docs.zenml.io/kitaru/adapters/pydantic-ai) for the recording lifecycle, replay overrides, [tool policies](https://docs.zenml.io/kitaru/guides/tool-policies), cost estimation and failure behavior.
