"""What the sandboxes' `mounts=` tests share: a bucket to mount, and an agent whose one tool runs a command."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import S3Mount


def bucket(key: str = 'the-key-id') -> S3Mount:
    return S3Mount(
        path='data',
        url='s3://bucket/',
        access_key_id=key,
        secret_access_key='the-secret-key',
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        endpoint='http://minio:9000',
    )


def probing(capability: AbstractCapability[str]) -> Agent[str, str]:
    """An agent that runs `echo contained` in `capability`'s workspace, so the first operation mounts."""
    agent = Agent(TestModel(call_tools=['probe']), deps_type=str, capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[str]) -> str:
        return (await ctx.workspace.run(['echo', 'contained'])).stdout

    return agent
