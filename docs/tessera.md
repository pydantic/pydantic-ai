# Tessera: Tool-Call Gating for PydanticAI Agents

[Tessera](https://github.com/kenithphilip/Tessera) is an
Apache-2.0 library of two security primitives for LLM agent
systems: signed trust labels on context segments with
taint-tracking at the tool-call boundary, and schema-enforced
dual-LLM execution. It pairs naturally with PydanticAI's
schema-first model: both treat free-form strings as the unsafe
escape hatch.

## Why this is in scope for PydanticAI users

Indirect prompt injection is the OWASP Agentic ASI01 vector.
Any PydanticAI agent that ingests retrieved context (web pages,
RAG chunks, MCP tool outputs) is exposed: an attacker who
controls a fetched document can plant instructions that alter
the agent's behaviour on the next sensitive tool call. The
`output_type=BookingDecision` discipline closes the
model-output channel; Tessera closes the tool-input channel.

## Install

```bash
pip install pydantic-ai tessera-mesh[pydantic-ai]
```

## Wire the guard

```python
import os
import secrets

from tessera.adapters.pydantic_ai import MeshPydanticAIGuard
from tessera.policy import Policy
from tessera.signing import HMACSigner

from pydantic_ai import Agent, Tool


# Tool that the guard will gate.
def book_hotel(city: str, nights: int) -> str:
    return f'Booked {nights} nights in {city}'

# 1. Build a Tessera policy that requires USER trust on book_hotel.
policy = Policy()
policy.require('book_hotel', level='USER')

# 2. Wrap the agent with the Tessera guard.
agent = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    tools=[Tool(book_hotel)],
)
guard = MeshPydanticAIGuard(
    policy=policy,
    signer=HMACSigner(os.environ.get('TESSERA_KEY', secrets.token_bytes(32))),
)
guard.attach(agent)

# 3. Run normally. The guard transparently labels segments and
#    denies sensitive tool calls when the active context contains
#    untrusted segments.
result = agent.run_sync(
    'Look up reviews for The Plaza Hotel and book me 2 nights '
    'if the rating is above 4 stars.'
)
print(result.output)
```

## What changes for the agent

- **Benign tasks** behave identically. Tessera's policy decision
  is `Allow` when every context segment carries `trust_level >=
  USER`.
- **Injected tool outputs** demote the context's minimum trust
  to `UNTRUSTED`. The next `book_hotel` call is denied; the
  agent receives a structured error and can either retry without
  the tainted context or surface the failure.

## Pairing with `output_type`

Tessera's `WorkerReport` (the schema-enforced dual-LLM contract)
uses the same idea PydanticAI's `output_type` does: forbid
free-form strings on the safety-critical channel. Recommending
one doesn't replace the other; they apply to different channels
(tool outputs vs. agent outputs) and are stronger together.

## Reference

- Tessera repo: <https://github.com/kenithphilip/Tessera>
- PydanticAI adapter source:
  [`tessera/adapters/pydantic_ai.py`](https://github.com/kenithphilip/Tessera/blob/main/src/tessera/adapters/pydantic_ai.py)
- Threat model: <https://github.com/kenithphilip/Tessera/blob/main/SECURITY.md>
- Adapter test:
  [`tests/test_pydantic_ai_adapter.py`](https://github.com/kenithphilip/Tessera/blob/main/tests/test_pydantic_ai_adapter.py)
