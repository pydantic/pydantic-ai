"""Fixture exercising eight common Pydantic AI v1.x deprecations.

Importable on v1.100.0; importing + the `__main__` block together fire each
deprecation warning. Each block is annotated with `# EXPECTED:` naming the v2
replacement so a migration evaluator can grade deterministically. No real
network or subprocess I/O happens at import time.
"""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_evals import Case, Dataset


# ---------------------------------------------------------------------------
# 1. `Agent.to_a2a()`  →  fasta2a.pydantic_ai.agent_to_a2a
#    EXPECTED v2: from fasta2a.pydantic_ai import agent_to_a2a; app = agent_to_a2a(agent)
# ---------------------------------------------------------------------------
def build_a2a_app(agent: Agent) -> object:
    return agent.to_a2a()  # type: ignore[deprecated]


# ---------------------------------------------------------------------------
# 2. `agent.iter(...).stream_responses()` (plural, yields tuple)
#    EXPECTED v2: `.stream_response()` (singular, yields ModelResponse)
# ---------------------------------------------------------------------------
async def consume_stream(agent: Agent, prompt: str) -> int:
    n = 0
    async with agent.iter(prompt) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for _msg, _is_last in stream.stream_responses():  # type: ignore[deprecated]
                        n += 1
    return n


# ---------------------------------------------------------------------------
# 3. `from pydantic_ai.mcp import MCPServerStdio`
#    EXPECTED v2: from pydantic_ai.mcp import MCPToolset (+ StdioTransport for arbitrary commands)
# ---------------------------------------------------------------------------
def build_mcp_server() -> object:
    from pydantic_ai.mcp import MCPServerStdio  # type: ignore[deprecated]

    return MCPServerStdio('uv', args=['run', 'my_mcp.py'])


# ---------------------------------------------------------------------------
# 4. `Agent('openai:gpt-4o', ...)` — string form is fine on v1, but on v2 the
#    `openai:` prefix flips to the Responses API.
#    EXPECTED v2: 'openai-chat:gpt-4o' to preserve v1 Chat-Completions semantics.
# ---------------------------------------------------------------------------
def build_openai_agent() -> Agent:
    return Agent('openai:gpt-4o', system_prompt='You are helpful.')


# ---------------------------------------------------------------------------
# 5. `Agent('gateway/gemini:...')` — deprecated gateway prefix
#    EXPECTED v2: 'gateway/google-cloud:gemini-1.5-pro'
# ---------------------------------------------------------------------------
def build_gateway_agent() -> Agent:
    return Agent('gateway/gemini:gemini-1.5-pro')


# ---------------------------------------------------------------------------
# 6. `result.usage()` method call
#    EXPECTED v2: `result.usage` (property)
# ---------------------------------------------------------------------------
async def get_tokens(agent: Agent, prompt: str) -> int:
    result = await agent.run(prompt)
    usage = result.usage()  # type: ignore[operator]
    return usage.input_tokens + usage.output_tokens


# ---------------------------------------------------------------------------
# 7. `Dataset(...)` without `name=`
#    EXPECTED v2: `Dataset(name='...', cases=[...], evaluators=[...])`
# ---------------------------------------------------------------------------
def build_dataset() -> Dataset[str, str, None]:
    return Dataset(  # type: ignore[call-arg]
        cases=[Case(name='c1', inputs='hello', expected_output='hi')],
        evaluators=[],
    )


# ---------------------------------------------------------------------------
# 8. `Agent(instrument=True)`
#    EXPECTED v2: `Agent(..., capabilities=[Instrumentation()])`
# ---------------------------------------------------------------------------
def build_instrumented_agent() -> Agent:
    return Agent('openai:gpt-4o', instrument=True)  # type: ignore[deprecated]


if __name__ == '__main__':  # pragma: no cover
    # Importing + constructing the agents fires most warnings; the streaming /
    # usage paths need a real (or fake) model to fire. Gate behind __main__ so
    # graders can `import v1_sample` cheaply.
    a = build_openai_agent()
    build_gateway_agent()
    build_mcp_server()
    build_dataset()
    build_instrumented_agent()
    asyncio.run(consume_stream(a, 'hi'))
    asyncio.run(get_tokens(a, 'hi'))
    build_a2a_app(a)
