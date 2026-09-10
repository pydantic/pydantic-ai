# Pydantic AI vs Vercel AI SDK

**Vercel AI SDK, at its best:** the JS/TS standard for AI apps — `generateText`, tool loops,
streaming UI helpers, `AbortSignal` as the cancellation norm, and provider-agnostic adapters.

**Pydantic AI, at its best:** a typed Python loop where the wire semantics of the result are
explicit, cancellation is a typed resumable outcome, and every seam is a type.

*Verified against Vercel AI SDK 5 (specVersion `v2`; 2026-09-10). Pydantic AI claims below are
self-contained scripts — offline, no API keys — re-executed by this repository's test suite.*

## Quick comparison

| What you get | Vercel AI SDK | Pydantic AI |
|---|---|---|
| Language/runtime | TS/JS + your framework (Next/Express) | Python 3.11+, asyncio-native |
| Result shape | `generateText` returns data; your handler decides | Output **transports** are explicit: a transform, a schema, a tool, native parts (proven below) |
| Cancellation | `AbortSignal` — the JS norm, forwarded into providers | Typed: `ctx.cancel()`, thread-safe token, catchable `RunCancelled` with resumable history |
| Extensions | Providers, tool sets, experimental agents | Capabilities: one unit, deferrable, serializable |
| Durable | Your infrastructure | Six engine wraps on the public interface |

## Prove it yourself

Their `generateText` hands you a result object; ours lets you declare what a response *becomes* —
a transform or a schema-validated dict:

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

```text
TextOutput(fn): HELLO WORLD
StructuredDict: {'n': 7} is a dict
```

The shape of your result is part of the agent's type — the same type that flows into tests and evals.

## Key differences

**Their best:** it is the standard for TS teams with real UI ergonomics (streaming, AgentCost usage,
skills protocol, approvals) on the platform you already deploy.

**Ours:** the Python loop is typed end to end — explicit output transports, a deps boundary, budgets
that halt before side effects, cancellation that resumes. If your stack is Python (or your team is),
the seams are stronger than what the JS norm provides.

## When to choose Vercel AI SDK

Your app is TypeScript and you want the ecosystem default — streaming UI, tool loops, provider
adapters.

## When to choose Pydantic AI

Your agent is Python and the result semantics matter: a response that is explicitly transformed
(`HELLO WORLD`) or validated (`{'n': 7}` is a dict) rather than an object you interpret.

## Summary

Their loop is the norm for TS; ours is a typed contract for Python. `TextOutput(fn)` → HELLO WORLD
StructuredDict: {'n': 7} is a dict; `StructuredDict` → {'n': 7} is a dict.

*Vercel AI SDK behavior pinned (docs + probe); records in the
[framework-comparison series](https://github.com/pydantic/pydantic-ai-notes). Pydantic AI verified
on 2.42.0, 2026-09-10.*