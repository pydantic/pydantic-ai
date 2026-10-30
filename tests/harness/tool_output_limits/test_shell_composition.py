"""Exercise the documented Shell and ToolOutputLimits stacking recipe."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.tool_output_limits import (
    Band,
    Spill,
    ToolOutputLimits,
    Truncate,
    TruncationStrategy,
)


@pytest.mark.parametrize('output_chars', [6_000, 25_000, 120_000], ids=['truncate', 'spill', 'native-cap-then-spill'])
async def test_shell_stacking_recipe(tmp_path: Path, output_chars: int):
    executable = Path(sys.executable).as_posix()
    command = f'"{executable}" -c "import sys; sys.stdout.write(\'x\' * {output_chars}); sys.exit(7)"'

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('run_command', {'command': command})])

    tail = Truncate(max_chars=4_000, strategy=TruncationStrategy.tail)
    agent = Agent(
        FunctionModel(respond),
        capabilities=[
            Shell(allowed_commands=[executable], max_output_chars=100_000),
            ToolOutputLimits(
                bands=[],
                per_tool={
                    'run_command': [
                        Band(over=20_000, action=Spill(then=tail)),
                        Band(over=4_000, action=tail),
                    ],
                },
            ),
        ],
    )
    result = await agent.run('run the command', workspace=LocalWorkspaceBackend(tmp_path))
    returns = [part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)]
    assert len(returns) == 1
    part = returns[0]
    assert isinstance(part.content, str)
    assert part.content.endswith('[exit code: 7]')
    if output_chars == 6_000:
        assert len(part.content) == 4_000
        assert part.content.startswith('[... output truncated, showing last 3952 chars]\n')
        assert part.metadata is None
    else:
        assert 'read_tool_result' in part.content
        assert part.metadata is not None
        handle = part.metadata['overflow_handle']
        assert handle.startswith(f'{tmp_path}/.pydantic-ai-harness/tool-output/')
        stored = Path(handle).read_text(encoding='utf-8')
        original = f'[stdout]\n{"x" * output_chars}\n[exit code: 7]'
        if output_chars == 25_000:
            assert stored == original
        else:
            assert len(stored) == 100_000
            assert stored.startswith('[... output truncated, showing last ')
            assert stored.endswith('[exit code: 7]')
            assert stored != original
