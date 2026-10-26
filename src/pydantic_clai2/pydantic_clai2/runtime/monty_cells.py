"""Run the user's own `run_code` cells in Monty, code mode's sandbox, between turns, through code mode itself.

A cell runs as a run of the session's agent whose model is scripted: it calls `run_code` once with
the user's code and stops. So the cell gets everything the model's cells get, the session's tools,
sandbox mounts, resource limits, and the Monty state that code mode keeps with the session, without
a second way to drive the sandbox. The run is kept out of the conversation; the next prompt tells the
model what the user ran, since its variables are in Monty now.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

MONTY_CELL = 'clai2_monty_cell'
"""Run metadata marking a user's Monty cell, so observers can tell it from a turn."""

RUN_CODE = 'run_code'
NOTE_PREFIX = 'Between turns, the user ran this cell in Monty, the sandbox your `run_code` runs in'
"""How each note to the model starts, so a UI that shows the cells already can leave the notes out."""
NOTE_CHARS = 4_000
"""The most of a cell's result the next prompt repeats to the model."""


def is_monty_cell(ctx: RunContext[object]) -> bool:
    """Whether `ctx` is a run of a user's Monty cell rather than of a prompt."""
    return bool(ctx.metadata and ctx.metadata.get(MONTY_CELL))


@dataclass(kw_only=True)
class MontyCell:
    """A scripted model that runs one `run_code` cell with the user's code, then keeps what it returned."""

    code: str
    returned: str | None = None
    failed: bool = False

    def model(self) -> FunctionModel:
        return FunctionModel(self.respond, stream_function=self.stream, model_name='monty-cell')

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        for part in request.parts:
            if isinstance(part, ToolReturnPart) and part.outcome == 'retried':
                # A cell that fails is reported, not retried: the user wrote it, not a model.
                self.returned, self.failed = part.model_response_str(wrap_if_error=False), True
                return ModelResponse(parts=[TextPart('done')])
            if isinstance(part, ToolReturnPart) and part.tool_name == RUN_CODE:
                self.returned = part.model_response_str()
                return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(RUN_CODE, {'code': self.code})])

    async def stream(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        [part] = self.respond(messages, info).parts
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    def report(self) -> str:
        """What the cell printed and returned, as the terminal and browser show it."""
        returned = self.returned or ''
        try:
            value: object = json.loads(returned)
        except ValueError:
            return returned
        if not isinstance(value, dict):
            return returned
        printed = value.get('output')  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        result = value.get('result')  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        lines = [printed.rstrip('\n')] if isinstance(printed, str) and printed else []
        if result is not None:
            lines.append(f'-> {result if isinstance(result, str) else json.dumps(result)}')
        return '\n'.join(lines)

    def note(self) -> str:
        """What the next prompt tells the model about this cell."""
        report = self.report()
        if len(report) > NOTE_CHARS:
            report = report[:NOTE_CHARS] + f'\n... {len(report) - NOTE_CHARS:,} more characters'
        outcome = 'It failed' if self.failed else 'It returned'
        return (
            f'{NOTE_PREFIX}, so its state is there now:\n'
            f'```python\n{self.code}\n```\n{outcome}:\n{report or "(nothing)"}'
        )
