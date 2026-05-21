"""Golden v2 form of `v1_sample.py`.

Imports, signatures, and call shapes match what v2.0.0b1 exposes. Verified
against the `v2-migration-skills` branch (forked from main, post-deprecation
but pre-v2-removal) — both v1 and v2 forms exist there, so each replacement is
checked against the actual definition.
"""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_evals import Case, Dataset


# ---------------------------------------------------------------------------
# 1. Agent.to_a2a() → external fasta2a.pydantic_ai bridge
# ---------------------------------------------------------------------------
def build_a2a_app(agent: Agent) -> object:
    # Requires: pip install 'fasta2a[pydantic-ai]>=0.6.1'
    from fasta2a.pydantic_ai import agent_to_a2a

    return agent_to_a2a(agent)


# ---------------------------------------------------------------------------
# 2. stream_responses() (tuple) → stream_response() (ModelResponse, singular)
# ---------------------------------------------------------------------------
async def consume_stream(agent: Agent, prompt: str) -> int:
    n = 0
    async with agent.iter(prompt) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for _msg in stream.stream_response():
                        n += 1
    return n


# ---------------------------------------------------------------------------
# 3. MCPServerStdio → MCPToolset
# ---------------------------------------------------------------------------
def build_mcp_server() -> object:
    from fastmcp.client.transports import StdioTransport

    from pydantic_ai.mcp import MCPToolset

    return MCPToolset(StdioTransport(command='uv', args=['run', 'my_mcp.py']))


# ---------------------------------------------------------------------------
# 4. 'openai:gpt-4o' → 'openai-chat:gpt-4o' (preserve Chat-Completions semantics).
#    To opt into the new Responses-API default, keep 'openai:gpt-4o'; that's a
#    separate decision from the upgrade itself.
# ---------------------------------------------------------------------------
def build_openai_agent() -> Agent:
    return Agent('openai-chat:gpt-4o', system_prompt='You are helpful.')


# ---------------------------------------------------------------------------
# 5. 'gateway/gemini:...' → 'gateway/google-cloud:...'
# ---------------------------------------------------------------------------
def build_gateway_agent() -> Agent:
    return Agent('gateway/google-cloud:gemini-1.5-pro')


# ---------------------------------------------------------------------------
# 6. result.usage() method → result.usage property
# ---------------------------------------------------------------------------
async def get_tokens(agent: Agent, prompt: str) -> int:
    result = await agent.run(prompt)
    usage = result.usage
    return usage.input_tokens + usage.output_tokens


# ---------------------------------------------------------------------------
# 7. Dataset(...) → Dataset(name=..., ...)
# ---------------------------------------------------------------------------
def build_dataset() -> Dataset[str, str, None]:
    return Dataset(
        name='greetings',
        cases=[Case(name='c1', inputs='hello', expected_output='hi')],
        evaluators=[],
    )


# ---------------------------------------------------------------------------
# 8. Agent(instrument=True) → capabilities=[Instrumentation()]
# ---------------------------------------------------------------------------
def build_instrumented_agent() -> Agent:
    return Agent('openai-chat:gpt-4o', capabilities=[Instrumentation()])


if __name__ == '__main__':  # pragma: no cover
    a = build_openai_agent()
    build_gateway_agent()
    build_mcp_server()
    build_dataset()
    build_instrumented_agent()
    asyncio.run(consume_stream(a, 'hi'))
    asyncio.run(get_tokens(a, 'hi'))
    build_a2a_app(a)
