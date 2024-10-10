from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any, Generic, Self, TypedDict, TypeVar, Union

from pydantic import TypeAdapter, ValidationError

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
        return messages.MessagesTypeAdapter.dump_json(self.message_history).decode()


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
    json_schema: _utils.ObjectJsonSchema
    allow_plain_message: bool
    outer_typed_dict: bool
    max_retries: int
    _current_retry: int = 0

    @classmethod
    def build(cls, response_type: type[ResultData], name: str, description: str, retries: int) -> Union[Self, None]:
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
            json_schema=_utils.check_object_json_schema(type_adapter.json_schema()),
            allow_plain_message=_utils.allow_plain_str(response_type),
            outer_typed_dict=outer_typed_dict,
            max_retries=retries,
        )

    def validate(self, message: messages.FunctionCall) -> _utils.Either[ResultData, messages.FunctionRetry]:
        """Validate a message."""
        try:
            result = self.type_adapter.validate_json(message.arguments)
        except ValidationError as e:
            self._current_retry += 1
            if self._current_retry > self.max_retries:
                raise
            else:
                m = messages.FunctionRetry(
                    function_id=message.function_id, function_name=message.function_name, content=e.errors()
                )
                return _utils.Either(right=m)
        else:
            return _utils.Either(left=result)
