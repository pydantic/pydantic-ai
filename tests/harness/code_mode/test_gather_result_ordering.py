"""Settled gather batches expose the same failure for every pending-ID order."""

from itertools import permutations
from typing import Literal

import pytest
from pydantic_monty import AsyncFutureSnapshot

from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import CodeMode


class TestCodeModeGatherResultOrdering:
    @pytest.mark.parametrize('pending_order', list(permutations(range(3))))
    @pytest.mark.parametrize('barrier', [False, True], ids=['parallel', 'pre-resolved'])
    @pytest.mark.parametrize('outcomes', ['errors', 'mixed', 'successes'])
    async def test_settled_batch_uses_call_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
        pending_order: tuple[int, ...],
        barrier: bool,
        outcomes: Literal['errors', 'mixed', 'successes'],
    ) -> None:
        # Force the real snapshot's unordered input, as in the sequential gather regressions.
        original = AsyncFutureSnapshot.pending_call_ids
        pending_batches: list[list[int]] = []

        def pending_ids(snapshot: AsyncFutureSnapshot) -> list[int]:
            ids = sorted(original.__get__(snapshot, AsyncFutureSnapshot))
            assert len(ids) == 3
            reordered = [ids[index] for index in pending_order]
            pending_batches.append(reordered)
            return reordered

        monkeypatch.setattr(AsyncFutureSnapshot, 'pending_call_ids', property(pending_ids))
        started: list[int | str] = []

        async def call(index: int) -> int:
            started.append(index)
            if outcomes == 'errors' or (outcomes == 'mixed' and index != 0):
                raise ValueError(f'error-{index}')
            return index

        async def checkpoint() -> None:
            started.append('checkpoint')

        prefix = 'a = call(index=0)\nb = call(index=1)\n'
        if barrier:
            prefix += 'checkpoint()\n'
        code = (
            'import asyncio\n'
            + prefix
            + 'try:\n'
            + '    result = await asyncio.gather(a, b, call(index=2))\n'
            + 'except ValueError as error:\n'
            + '    result = str(error)\n'
            + 'result'
        )
        if outcomes == 'successes':
            code += '\n[result, await asyncio.gather(call(index=3), call(index=4), call(index=5))]'

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('run_code', {'code': code})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(respond),
            tools=[call, Tool(checkpoint, sequential=True)],
            capabilities=[CodeMode()],
        )
        result = await agent.run('go')
        returns = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
        ]
        expected = {'errors': 'error-0', 'mixed': 'error-1', 'successes': [[0, 1, 2], [3, 4, 5]]}[outcomes]
        assert returns == [expected]
        assert len(pending_batches) == (2 if outcomes == 'successes' else 1)
        expected_starts: list[int | str] = [0, 1, 'checkpoint', 2] if barrier else [0, 1, 2]
        if outcomes == 'successes':
            expected_starts += [3, 4, 5]
        assert started == expected_starts
