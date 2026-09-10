# Pydantic AI vs Vercel AI SDK

Choosing an agent framework and you're down to
[Pydantic AI](../agent.md) and the Vercel AI SDK? They sit on different sides of the language line (TypeScript vs Python) — if your service doesn't bind you to either, this page is the tiebreaker — the answer first, then code
you can run in seconds.

## Pydantic AI fits if you need

- a **Python 3.10+** agent
- **result semantics declared by the framework** — a transform, a schema, native parts — not interpreted by your handler
- the Python-side checklist: deps boundary, budgets, resumable cancellation

## Why the answers differ

Their loop is the norm for TypeScript; ours is a typed contract for Python. Where it shows: what a response *becomes* is part of the agent's type, not a post-processing decision.

## See it work

Say the shape of your result should be your framework's job, not your handler's puzzle.

In the AI SDK, `generateText` hands you a result object that your code interprets (SDK 5).

Your side, runs offline:

```python {title="output_transports.py"}
"""Wire semantics are explicit: the output transport decides how a response
becomes your result — a transform, or a dict validated against a schema."""
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.output import StructuredDict, TextOutput

async def model(messages, info):
    return ModelResponse(parts=[TextPart('hello world')])

async def model_json(messages, info):
    return ModelResponse(parts=[TextPart('{"n": 7}')])

# A transform applied to the model's text:
def upper(t: str) -> str:
    return t.upper()

agent = Agent(FunctionModel(model), output_type=TextOutput(upper))
print('TextOutput(fn):', agent.run_sync('q').output)

# A JSON schema the response must satisfy:
schema = {
    'type': 'object',
    'properties': {'n': {'type': 'integer'}},
    'required': ['n'],
}
agent2 = Agent(FunctionModel(model_json), output_type=StructuredDict(schema))
out = agent2.run_sync('q').output
print(f'StructuredDict: {out!r} is a {type(out).__name__}')
assert agent.run_sync('q').output == 'HELLO WORLD'
assert out == {'n': 7}


```

```text
TextOutput(fn): HELLO WORLD
StructuredDict: {'n': 7} is a dict
```

**Notice:** Here a response *becomes* what you declared it to be — a transform or a schema-validated dict — and that type flows into tests and evals.

## The details

| What you get | Vercel AI SDK | Pydantic AI |
|---|---|---|
|---|---|---|
| Language/runtime | TS/JS + your framework (Next/Express) | Python 3.10+, asyncio-native |
| Result shape | `generateText` returns data; your handler decides | Output **transports** are explicit: a transform, a schema, a tool, native parts (proven below) |
| Cancellation | `AbortSignal` — the JS norm, forwarded into providers | Typed: `ctx.cancel()`, thread-safe token, catchable `RunCancelled` with resumable history |
| Extensions | Providers, tool sets, experimental agents | Capabilities: one unit, deferrable, serializable |
| Durable | Your infrastructure | Six engine wraps on the public interface |

## If this answer doesn't fit you

If your app is TypeScript, Vercel AI SDK is the ecosystem default with real UI ergonomics — streaming, tool loops, skills protocol — and that's not nothing. We can't compete with JS comfort from here. This page is for the Python side: where the result's shape should be the framework's job, and where the loop has seams.

---

---

*Versions: Vercel AI SDK 5; Pydantic AI 2.42.0 — 2026-09-10. Snippets re-executed by this repository's tests.*
