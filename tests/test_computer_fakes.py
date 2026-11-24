"""`FakeComputer`, the `computer` tool the model adapter tests use, returns results in the shape Pydantic AI Harness's tool does.

Not a VCR test: the fake tool and `FunctionModel` make no provider request, so a cassette would record nothing.
"""

from __future__ import annotations

from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.computer_use import Click, ComputerReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from .computer_fakes import FakeComputer
from .workspace_fakes import InMemoryProvider


def look_then_click(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if any(isinstance(part, ComputerReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('done')])
    actions = [{'type': 'screenshot'}, {'type': 'click', 'x': 1, 'y': 2}]
    return ModelResponse(parts=[ToolCallPart('computer', {'actions': actions})])


def computer_result(messages: list[ModelMessage]) -> object:
    (part,) = [part for message in messages for part in message.parts if isinstance(part, ComputerReturnPart)]
    return part.metadata


async def test_the_fake_computer_runs_actions_and_can_fail_them() -> None:
    computer = FakeComputer()
    result = await Agent(FunctionModel(look_then_click), capabilities=[computer, InMemoryProvider().capability()]).run(
        'Click it.'
    )
    assert computer.actions == [Click(x=1, y=2)]
    assert computer_result(result.all_messages()) == {'performed': 2}

    computer = FakeComputer(native=False, fail_acting=True)
    result = await Agent(FunctionModel(look_then_click), capabilities=[computer, InMemoryProvider().capability()]).run(
        'Click it.'
    )
    assert computer.actions == []
    assert computer_result(result.all_messages()) == {'performed': 1, 'error': 'The action failed.'}
