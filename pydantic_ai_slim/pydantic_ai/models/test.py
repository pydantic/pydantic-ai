from __future__ import annotations as _annotations

import re
import string
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import InitVar, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal

import pydantic_core

from .. import _utils
from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from ..result import Usage
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
)
from .function import _estimate_string_tokens, _estimate_usage  # pyright: ignore[reportPrivateUsage]


@dataclass
class _TextResult:
    """A private wrapper class to tag a result that came from the custom_result_text field."""

    value: str | None


@dataclass
class _FunctionToolResult:
    """A wrapper class to tag a result that came from the custom_result_args field."""

    value: Any | None


@dataclass
class TestModel(Model):
    """A model specifically for testing purposes.

    This will (by default) call all tools in the agent, then return a tool response if possible,
    otherwise a plain response.

    How useful this model is will vary significantly.

    Apart from `__init__` derived by the `dataclass` decorator, all methods are private or match those
    of the base class.
    """

    # NOTE: Avoid test discovery by pytest.
    __test__ = False

    call_tools: list[str] | Literal['all'] = 'all'
    """List of tools to call. If `'all'`, all tools will be called."""
    custom_result_text: str | None = None
    """If set, this text is returned as the final result."""
    custom_result_args: Any | None = None
    """If set, these args will be passed to the result tool."""
    seed: int = 0
    """Seed for generating random data."""
    last_model_request_parameters: ModelRequestParameters | None = field(default=None, init=False)
    """The last ModelRequestParameters passed to the model in a request.

    The ModelRequestParameters contains information about the function and result tools available during request handling.

    This is set when a request is made, so will reflect the function tools from the last step of the last run.
    """
    _model_name: str = field(default='test', repr=False)
    _system: str | None = field(default=None, repr=False)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        self.last_model_request_parameters = model_request_parameters

        model_response = self._request(messages, model_settings, model_request_parameters)
        usage = _estimate_usage([*messages, model_response])
        return model_response, usage

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        self.last_model_request_parameters = model_request_parameters

        model_response = self._request(messages, model_settings, model_request_parameters)
        yield TestStreamedResponse(
            _model_name=self._model_name, _structured_response=model_response, _messages=messages
        )

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str | None:
        """The system / model provider."""
        return self._system

    def gen_tool_args(self, tool_def: ToolDefinition) -> Any:
        return _JsonSchemaTestData(tool_def.parameters_json_schema, self.seed).generate()

    def _get_tool_calls(self, model_request_parameters: ModelRequestParameters) -> list[tuple[str, ToolDefinition]]:
        if self.call_tools == 'all':
            return [(r.name, r) for r in model_request_parameters.function_tools]
        else:
            function_tools_lookup = {t.name: t for t in model_request_parameters.function_tools}
            tools_to_call = (function_tools_lookup[name] for name in self.call_tools)
            return [(r.name, r) for r in tools_to_call]

    def _get_result(self, model_request_parameters: ModelRequestParameters) -> _TextResult | _FunctionToolResult:
        if self.custom_result_text is not None:
            assert model_request_parameters.allow_text_result, (
                'Plain response not allowed, but `custom_result_text` is set.'
            )
            assert self.custom_result_args is None, 'Cannot set both `custom_result_text` and `custom_result_args`.'
            return _TextResult(self.custom_result_text)
        elif self.custom_result_args is not None:
            assert model_request_parameters.result_tools is not None, (
                'No result tools provided, but `custom_result_args` is set.'
            )
            result_tool = model_request_parameters.result_tools[0]

            if k := result_tool.outer_typed_dict_key:
                return _FunctionToolResult({k: self.custom_result_args})
            else:
                return _FunctionToolResult(self.custom_result_args)
        elif model_request_parameters.allow_text_result:
            return _TextResult(None)
        elif model_request_parameters.result_tools:
            return _FunctionToolResult(None)
        else:
            return _TextResult(None)

    def _request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        tool_calls = self._get_tool_calls(model_request_parameters)
        result = self._get_result(model_request_parameters)
        result_tools = model_request_parameters.result_tools

        # if there are tools, the first thing we want to do is call all of them
        if tool_calls and not any(isinstance(m, ModelResponse) for m in messages):
            return ModelResponse(
                parts=[ToolCallPart(name, self.gen_tool_args(args)) for name, args in tool_calls],
                model_name=self._model_name,
            )

        if messages:
            last_message = messages[-1]
            assert isinstance(last_message, ModelRequest), 'Expected last message to be a `ModelRequest`.'

            # check if there are any retry prompts, if so retry them
            new_retry_names = {p.tool_name for p in last_message.parts if isinstance(p, RetryPromptPart)}
            if new_retry_names:
                # Handle retries for both function tools and result tools
                # Check function tools first
                retry_parts: list[ModelResponsePart] = [
                    ToolCallPart(name, self.gen_tool_args(args)) for name, args in tool_calls if name in new_retry_names
                ]
                # Check result tools
                if result_tools:
                    retry_parts.extend(
                        [
                            ToolCallPart(
                                tool.name,
                                result.value
                                if isinstance(result, _FunctionToolResult) and result.value is not None
                                else self.gen_tool_args(tool),
                            )
                            for tool in result_tools
                            if tool.name in new_retry_names
                        ]
                    )
                return ModelResponse(parts=retry_parts, model_name=self._model_name)

        if isinstance(result, _TextResult):
            if (response_text := result.value) is None:
                # build up details of tool responses
                output: dict[str, Any] = {}
                for message in messages:
                    if isinstance(message, ModelRequest):
                        for part in message.parts:
                            if isinstance(part, ToolReturnPart):
                                output[part.tool_name] = part.content
                if output:
                    return ModelResponse(
                        parts=[TextPart(pydantic_core.to_json(output).decode())], model_name=self._model_name
                    )
                else:
                    return ModelResponse(parts=[TextPart('success (no tool calls)')], model_name=self._model_name)
            else:
                return ModelResponse(parts=[TextPart(response_text)], model_name=self._model_name)
        else:
            assert result_tools, 'No result tools provided'
            custom_result_args = result.value
            result_tool = result_tools[self.seed % len(result_tools)]
            if custom_result_args is not None:
                return ModelResponse(
                    parts=[ToolCallPart(result_tool.name, custom_result_args)], model_name=self._model_name
                )
            else:
                response_args = self.gen_tool_args(result_tool)
                return ModelResponse(parts=[ToolCallPart(result_tool.name, response_args)], model_name=self._model_name)


@dataclass
class TestStreamedResponse(StreamedResponse):
    """A structured response that streams test data."""

    _model_name: str
    _structured_response: ModelResponse
    _messages: InitVar[Iterable[ModelMessage]]
    _timestamp: datetime = field(default_factory=_utils.now_utc, init=False)

    def __post_init__(self, _messages: Iterable[ModelMessage]):
        self._usage = _estimate_usage(_messages)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for i, part in enumerate(self._structured_response.parts):
            if isinstance(part, TextPart):
                text = part.content
                *words, last_word = text.split(' ')
                words = [f'{word} ' for word in words]
                words.append(last_word)
                if len(words) == 1 and len(text) > 2:
                    mid = len(text) // 2
                    words = [text[:mid], text[mid:]]
                self._usage += _get_string_usage('')
                yield self._parts_manager.handle_text_delta(vendor_part_id=i, content='')
                for word in words:
                    self._usage += _get_string_usage(word)
                    yield self._parts_manager.handle_text_delta(vendor_part_id=i, content=word)
            else:
                yield self._parts_manager.handle_tool_call_part(
                    vendor_part_id=i, tool_name=part.tool_name, args=part.args, tool_call_id=part.tool_call_id
                )

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp


_chars = string.ascii_letters + string.digits + string.punctuation


class _JsonSchemaTestData:
    """Generate data that matches a JSON schema.

    This tries to generate the minimal viable data for the schema.
    """

    def __init__(self, schema: _utils.ObjectJsonSchema, seed: int = 0):
        self.schema = schema
        self.defs = schema.get('$defs', {})
        self.seed = seed

    def generate(self) -> Any:
        """Generate data for the JSON schema."""
        return self._gen_any(self.schema)

    def _gen_any(self, schema: dict[str, Any]) -> Any:
        """Generate data for any JSON Schema."""
        if const := schema.get('const'):
            return const
        elif enum := schema.get('enum'):
            return enum[self.seed % len(enum)]
        elif examples := schema.get('examples'):
            return examples[self.seed % len(examples)]
        elif ref := schema.get('$ref'):
            key = re.sub(r'^#/\$defs/', '', ref)
            js_def = self.defs[key]
            return self._gen_any(js_def)
        elif any_of := schema.get('anyOf'):
            return self._gen_any(any_of[self.seed % len(any_of)])

        type_ = schema.get('type')
        if type_ is None:
            # if there's no type or ref, we can't generate anything
            return self._char()
        elif type_ == 'object':
            return self._object_gen(schema)
        elif type_ == 'string':
            return self._str_gen(schema)
        elif type_ == 'integer':
            return self._int_gen(schema)
        elif type_ == 'number':
            return float(self._int_gen(schema))
        elif type_ == 'boolean':
            return self._bool_gen()
        elif type_ == 'array':
            return self._array_gen(schema)
        elif type_ == 'null':
            return None
        else:
            raise NotImplementedError(f'Unknown type: {type_}, please submit a PR to extend JsonSchemaTestData!')

    def _object_gen(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Generate data for a JSON Schema object."""
        required = set(schema.get('required', []))

        data: dict[str, Any] = {}
        if properties := schema.get('properties'):
            for key, value in properties.items():
                if key in required:
                    data[key] = self._gen_any(value)

        if addition_props := schema.get('additionalProperties'):
            add_prop_key = 'additionalProperty'
            while add_prop_key in data:
                add_prop_key += '_'
            if addition_props is True:
                data[add_prop_key] = self._char()
            else:
                data[add_prop_key] = self._gen_any(addition_props)

        return data

    def _str_gen(self, schema: dict[str, Any]) -> str:
        """Generate a string from a JSON Schema string."""
        min_len = schema.get('minLength')
        if min_len is not None:
            return self._char() * min_len

        if schema.get('maxLength') == 0:
            return ''

        if fmt := schema.get('format'):
            if fmt == 'date':
                return (date(2024, 1, 1) + timedelta(days=self.seed)).isoformat()

        return self._char()

    def _int_gen(self, schema: dict[str, Any]) -> int:
        """Generate an integer from a JSON Schema integer."""
        maximum = schema.get('maximum')
        if maximum is None:
            exc_max = schema.get('exclusiveMaximum')
            if exc_max is not None:
                maximum = exc_max - 1

        minimum = schema.get('minimum')
        if minimum is None:
            exc_min = schema.get('exclusiveMinimum')
            if exc_min is not None:
                minimum = exc_min + 1

        if minimum is not None and maximum is not None:
            return minimum + self.seed % (maximum - minimum)
        elif minimum is not None:
            return minimum + self.seed
        elif maximum is not None:
            return maximum - self.seed
        else:
            return self.seed

    def _bool_gen(self) -> bool:
        """Generate a boolean from a JSON Schema boolean."""
        return bool(self.seed % 2)

    def _array_gen(self, schema: dict[str, Any]) -> list[Any]:
        """Generate an array from a JSON Schema array."""
        data: list[Any] = []
        unique_items = schema.get('uniqueItems')
        if prefix_items := schema.get('prefixItems'):
            for item in prefix_items:
                data.append(self._gen_any(item))
                if unique_items:
                    self.seed += 1

        items_schema = schema.get('items', {})
        min_items = schema.get('minItems', 0)
        if min_items > len(data):
            for _ in range(min_items - len(data)):
                data.append(self._gen_any(items_schema))
                if unique_items:
                    self.seed += 1
        elif items_schema:
            # if there is an `items` schema, add an item unless it would break `maxItems` rule
            max_items = schema.get('maxItems')
            if max_items is None or max_items > len(data):
                data.append(self._gen_any(items_schema))
                if unique_items:
                    self.seed += 1

        return data

    def _char(self) -> str:
        """Generate a character on the same principle as Excel columns, e.g. a-z, aa-az..."""
        chars = len(_chars)
        s = ''
        rem = self.seed // chars
        while rem > 0:
            s += _chars[(rem - 1) % chars]
            rem //= chars
        s += _chars[self.seed % chars]
        return s


def _get_string_usage(text: str) -> Usage:
    response_tokens = _estimate_string_tokens(text)
    return Usage(response_tokens=response_tokens, total_tokens=response_tokens)
