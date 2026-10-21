"""CodeMode limit refusal across Prefect flow retries and cached nested task results."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import CodeMode
from tests.conftest import try_import

with try_import() as imports_successful:
    from prefect import flow
    from prefect.settings import PREFECT_LOCAL_STORAGE_PATH, temporary_settings

    from pydantic_ai.durable_exec.prefect import PrefectDurability

pytestmark = [
    pytest.mark.xdist_group(name='prefect'),
    pytest.mark.skipif(not imports_successful(), reason='prefect not installed'),
]


class TestCodeMode:
    @pytest.mark.parametrize(
        'code',
        [
            'await add(a=3, b=4)\nawait add(a=5, b=6)',
            'import asyncio\nawait asyncio.gather(add(a=3, b=4), add(a=5, b=6))',
        ],
        ids=['sequential', 'gather'],
    )
    async def test_tool_calls_limit_refuses_after_flow_retry(
        self, prefect_test_server: None, tmp_path: Path, code: str
    ) -> None:
        """Retry cached nested tool results without repeating side effects or losing the terminal refusal."""
        additions: list[tuple[int, int]] = []
        limit_errors: list[str] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': code}, tool_call_id='limit')])

        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            additions.append((a, b))
            return a + b

        agent = Agent(
            FunctionModel(model),
            name='code_mode_prefect_limit',
            tools=[add],
            capabilities=[CodeMode(), PrefectDurability()],
        )

        @flow(retries=1, retry_delay_seconds=0)
        async def run() -> None:
            try:
                await agent.run('Add twice', usage_limits=UsageLimits(tool_calls_limit=2))
            except UsageLimitExceeded as error:
                limit_errors.append(str(error))
                raise

        with (
            temporary_settings({PREFECT_LOCAL_STORAGE_PATH: tmp_path}),
            pytest.raises(UsageLimitExceeded, match=r'tool_calls_limit of 2 \(tool_calls=3\)'),
        ):
            await run()

        assert len(limit_errors) == 2
        assert limit_errors[0] == limit_errors[1]
        assert additions == [(3, 4)]
