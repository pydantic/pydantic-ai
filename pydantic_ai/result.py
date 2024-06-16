from dataclasses import dataclass
from typing import Any, AsyncIterable, Generic, Self, TypedDict, TypeVar

import pydantic_core
from pydantic import TypeAdapter
from pydantic.json_schema import JsonSchemaValue

from . import _utils, messages


@dataclass
class Cost:
    """Cost of a run."""

    total_cost: int


ResultData = TypeVar('ResultData')


@dataclass
class RunResult(Generic[ResultData]):
    """Result of a run."""

    response: ResultData
    message_history: list[messages.Message]
    cost: Cost

    def message_history_json(self) -> str:
        """Return the history of messages as a JSON string."""
        return pydantic_core.to_json(self.message_history).decode()


@dataclass
class RunStreamResult(Generic[ResultData]):
    """Result of an async streamed run."""

    # history: History
    cost: Cost
    _streamed: str = ''

    async def stream(self) -> AsyncIterable[str]:
        """Iterate through the result."""
        raise NotImplementedError()

    async def response(self) -> ResultData:
        """Access the combined result - basically the chunks yielded by `stream` concatenated together and validated."""
        raise NotImplementedError()


@dataclass
class ResultSchema(Generic[ResultData]):
    """Model the final response from an agent run.

    Similar to `Retriever` but for the final response
    """

    name: str
    description: str
    type_adapter: TypeAdapter[Any]
    json_schema: JsonSchemaValue
    allow_plain_message: bool
    outer_typed_dict: bool
    retries: int

    @classmethod
    def build(cls, response_type: type[ResultData], name: str, description: str, retries: int) -> Self | None:
        """Build a ResponseModel dataclass from a response type."""
        if response_type is str:
            return None

        if _utils.is_model_like(response_type):
            type_adapter = TypeAdapter(response_type)
            outer_typed_dict = False
        else:
            # noinspection PyTypedDict
            response_data_typed_dict = TypedDict('response_data_typed_dict', {'response': response_type})  # noqa
            type_adapter = TypeAdapter(response_data_typed_dict)
            outer_typed_dict = True

        return cls(
            name=name,
            description=description,
            type_adapter=type_adapter,
            json_schema=type_adapter.json_schema(),
            allow_plain_message=_utils.allow_plain_str(response_type),
            outer_typed_dict=outer_typed_dict,
            retries=retries,
        )

    def validate(self, message: messages.FunctionCall) -> ResultData:
        """Validate a message."""
        # TODO retries
        return self.type_adapter.validate_json(message['arguments'])
