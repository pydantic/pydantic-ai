---
title: Repair Tool Arguments
description: Repair malformed JSON tool arguments before schema validation.
---

# Repair Tool Arguments

Repair malformed JSON tool arguments when a model produces trailing commas, single-quoted keys,
or incomplete JSON. Use this capability on its own or through [Coder](coder.md), which includes it.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

Install `pydantic-ai-harness`; no extra is required for repair. This example uses a local test model.

```bash
pip/uv-add pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.repair_tool_arguments import RepairToolArguments

agent = Agent(TestModel(), capabilities=[RepairToolArguments()])

@agent.tool_plain
async def greet(name: str) -> str:
    return f'Hello, {name}!'

result = agent.run_sync('Greet someone')
print(result.output)
```

## Validation and safety

`RepairToolArguments` uses `json-repair` in `before_tool_validate`, before normal Pydantic AI tool
schema validation. It applies to all tools on the agent, including tools supplied by other capabilities.
Valid JSON strings and already-parsed arguments pass through unchanged. Missing fields and invalid
field types still follow normal validation and retry behavior. If the repair parser raises `ValueError`
or `RecursionError`, the original arguments go through normal validation.

Repair is heuristic: malformed input can be ambiguous, and inferred strings may differ from the model's
intent. The capability does not supply a schema to the repair library or bypass tool validation,
approval, or execution checks. It has no configuration options.

## Telemetry

Each repair attempt emits a `repair_tool_arguments` span through `ctx.tracer`. Valid JSON and parsed
arguments do not emit this span. The span contains no arguments or tool contents, including when
`trace_include_content` is enabled. Core still emits its normal tool spans.

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/repair_tool_arguments/).

## API reference

::: pydantic_ai_harness.repair_tool_arguments.RepairToolArguments
