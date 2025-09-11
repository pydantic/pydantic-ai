from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, overload

from pydantic import TypeAdapter
from pydantic_core import to_json
from pydantic_graph.v2.id_types import NodeId
from pydantic_graph.v2.step import StepContext
from pydantic_graph.v2.util import TypeOrTypeExpression, unpack_type_expression

from pydantic_ai import Agent, models


@dataclass(init=False)
class Prompt[InputT, OutputT]:
    input_type: type[InputT]
    output_type: type[Any]
    output_selector: Callable[[InputT, Any], OutputT] | None
    prompt: str
    model: models.Model | models.KnownModelName | str = 'openai:gpt-4o'

    @overload
    def __init__(
        self,
        *,
        input_type: TypeOrTypeExpression[InputT],
        output_type: TypeOrTypeExpression[OutputT],
        prompt: str,
        model: models.Model | models.KnownModelName | str = 'openai:gpt-4o',
    ) -> None: ...
    @overload
    def __init__[IntermediateT](
        self,
        *,
        input_type: TypeOrTypeExpression[InputT],
        output_type: TypeOrTypeExpression[IntermediateT],
        output_transform: Callable[[InputT, IntermediateT], OutputT],
        prompt: str,
        model: models.Model | models.KnownModelName | str = 'openai:gpt-4o',
    ) -> None: ...
    def __init__(
        self,
        *,
        input_type: TypeOrTypeExpression[InputT],
        output_type: TypeOrTypeExpression[Any],
        output_transform: Callable[[InputT, Any], OutputT] | None = None,
        prompt: str,
        model: models.Model | models.KnownModelName | str = 'openai:gpt-4o',
    ):
        self.input_type = unpack_type_expression(input_type)
        self.output_type = unpack_type_expression(output_type)
        self.output_transform = output_transform
        self.prompt = prompt
        self.model = model

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
        return Agent(
            model=self.model,
            output_type=self.output_type,
            instructions=instructions,
        )

    async def __call__(self, ctx: StepContext[Any, InputT]) -> OutputT:
        result = self.agent.run_sync(to_json(ctx.inputs, indent=2).decode())
        output = result.output
        if self.output_transform:
            output = self.output_transform(ctx.inputs, output)
        return output


@dataclass
class Interruption[StopT, ResumeT]:
    value: StopT
    next_node: (
        NodeId  # This is the node this walk should resume from after the interruption
    )
    graph_state: Any = None  # TODO: Need a way to pass the graph state ...?
