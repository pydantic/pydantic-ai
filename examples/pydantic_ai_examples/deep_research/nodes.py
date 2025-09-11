from dataclasses import dataclass
from functools import cached_property
from typing import Any, NewType, cast, get_args, get_origin

from pydantic import TypeAdapter
from pydantic_core import to_json

from pydantic_ai import Agent, models

NodeId = NewType('NodeId', str)


class Node[StateT, InputT, OutputT]:
    id: NodeId
    _output_type: OutputT

    async def run(self, state: StateT, inputs: InputT) -> OutputT:
        raise NotImplementedError


class TypeUnion[T]:
    pass


@dataclass(init=False)
class Prompt[InputT, OutputT](Node[Any, InputT, OutputT]):
    input_type: type[InputT]
    output_type: type[TypeUnion[OutputT]] | type[OutputT]
    prompt: str
    model: models.Model | models.KnownModelName | str = 'openai:gpt-4o'

    @cached_property
    def agent(self) -> Agent[None, OutputT]:
        input_json_schema = to_json(
            TypeAdapter(self.input_type).json_schema(), indent=2
        ).decode()
        instructions = '\n'.join(
            [
                'You will receive messages matching the following JSON schema:',
                input_json_schema,
                '',
                'Generate output based on the following instructions:',
                self.prompt,
            ]
        )
        output_type = self.output_type
        if get_origin(output_type) is TypeUnion:
            output_type = get_args(self.output_type)[0]
        return Agent(
            model=self.model,
            output_type=cast(type[OutputT], output_type),
            instructions=instructions,
        )

    async def run(self, state: Any, inputs: InputT) -> OutputT:
        result = await self.agent.run(to_json(inputs, indent=2).decode())
        return result.output
