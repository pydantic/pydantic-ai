from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import TypeAdapter, ValidationError
from typing_extensions import Self, TypedDict

from . import _utils, messages

ResultData = TypeVar('ResultData')


@dataclass
class Cost:
    """Cost of a run."""

    total_cost: int


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
class ResultSchema(Generic[ResultData]):
    """Model the final response from an agent run.

    Similar to `Retriever` but for the final result of running an agent.
    """

    name: str
    description: str
    type_adapter: TypeAdapter[Any]
    json_schema: _utils.ObjectJsonSchema
    allow_text_result: bool
    outer_typed_dict: bool

    @classmethod
    def build(cls, response_type: type[ResultData], name: str, description: str) -> Self | None:
        """Build a ResultSchema dataclass from a response type."""
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
            allow_text_result=_utils.allow_plain_str(response_type),
            outer_typed_dict=outer_typed_dict,
        )

    def validate(self, tool_call: messages.ToolCall) -> ResultData:
        """Validate a result message.

        Returns:
            Either the validated result data (left) or a retry message (right).
        """
        from ._result import ToolRetryError

        try:
            if isinstance(tool_call.args, messages.ArgsJson):
                result = self.type_adapter.validate_json(tool_call.args.args_json)
            else:
                result = self.type_adapter.validate_python(tool_call.args.args_object)
        except ValidationError as e:
            m = messages.ToolRetry(
                tool_name=tool_call.tool_name,
                content=e.errors(include_url=False),
                tool_id=tool_call.tool_id,
            )
            raise ToolRetryError(m) from e
        else:
            if self.outer_typed_dict:
                result = result['response']
            return result
