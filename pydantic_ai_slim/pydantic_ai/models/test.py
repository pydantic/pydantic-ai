from __future__ import annotations as _annotations

import re
import string
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import InitVar, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal

import pydantic_core
from typing_extensions import assert_never

from .. import _utils
from .._run_context import RunContext
from ..exceptions import UserError
from ..messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)
from ..profiles import ModelProfileSpec
from ..settings import ModelSettings
from ..tools import ToolDefinition
from ..usage import RequestUsage
from . import Model, ModelRequestParameters, StreamedResponse
from .function import _estimate_string_tokens, _estimate_usage  # pyright: ignore[reportPrivateUsage]


@dataclass
class _WrappedTextOutput:
    """A private wrapper class to tag an output that came from the custom_output_text field."""

    value: str | None


@dataclass(init=False)
class _WrappedToolOutput:
    """A wrapper class to tag an output that came from the custom_output_args field."""

    value: dict[str, Any] | None

    def __init__(self, value: Any | None):
        self.value = pydantic_core.to_jsonable_python(value)


@dataclass(init=False)
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
    custom_output_text: str | None = None
    """If set, this text is returned as the final output."""
    custom_output_args: Any | None = None
    """If set, these args will be passed to the output tool."""
    seed: int = 0
    """Seed for generating random data."""
    last_model_request_parameters: ModelRequestParameters | None = field(default=None, init=False)
    """The last ModelRequestParameters passed to the model in a request.

    The ModelRequestParameters contains information about the function and output tools available during request handling.

    This is set when a request is made, so will reflect the function tools from the last step of the last run.
    """
    _model_name: str = field(default='test', repr=False)
    _system: str = field(default='test', repr=False)

    def __init__(
        self,
        *,
        call_tools: list[str] | Literal['all'] = 'all',
        custom_output_text: str | None = None,
        custom_output_args: Any | None = None,
        seed: int = 0,
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize TestModel with optional settings and profile."""
        self.call_tools = call_tools
        self.custom_output_text = custom_output_text
        self.custom_output_args = custom_output_args
        self.seed = seed
        self.last_model_request_parameters = None
        self._model_name = 'test'
        self._system = 'test'
        super().__init__(settings=settings, profile=profile)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        self.last_model_request_parameters = model_request_parameters
        model_response = self._request(messages, model_settings, model_request_parameters)
        model_response.usage = _estimate_usage([*messages, model_response])
        return model_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        self.last_model_request_parameters = model_request_parameters

        model_response = self._request(messages, model_settings, model_request_parameters)
        yield TestStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _structured_response=model_response,
            _messages=messages,
            _provider_name=self._system,
        )

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
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

    def _get_output(self, model_request_parameters: ModelRequestParameters) -> _WrappedTextOutput | _WrappedToolOutput:
        if self.custom_output_text is not None:
            assert model_request_parameters.output_mode != 'tool', (
                'Plain response not allowed, but `custom_output_text` is set.'
            )
            assert self.custom_output_args is None, 'Cannot set both `custom_output_text` and `custom_output_args`.'
            return _WrappedTextOutput(self.custom_output_text)
        elif self.custom_output_args is not None:
            assert model_request_parameters.output_tools is not None, (
                'No output tools provided, but `custom_output_args` is set.'
            )
            output_tool = model_request_parameters.output_tools[0]

            if k := output_tool.outer_typed_dict_key:
                return _WrappedToolOutput({k: self.custom_output_args})
            else:
                return _WrappedToolOutput(self.custom_output_args)
        elif model_request_parameters.allow_text_output:
            return _WrappedTextOutput(None)
        elif model_request_parameters.output_tools:
            return _WrappedToolOutput(None)
        else:
            return _WrappedTextOutput(None)

    def _request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if model_request_parameters.builtin_tools:
            raise UserError('TestModel does not support built-in tools')

        tool_calls = self._get_tool_calls(model_request_parameters)
        output_wrapper = self._get_output(model_request_parameters)
        output_tools = model_request_parameters.output_tools

        # if there are tools, the first thing we want to do is call all of them
        if tool_calls and not any(isinstance(m, ModelResponse) for m in messages):
            return ModelResponse(
                parts=[
                    ToolCallPart(name, self.gen_tool_args(args), tool_call_id=f'pyd_ai_tool_call_id__{name}')
                    for name, args in tool_calls
                ],
                model_name=self._model_name,
            )

        if messages:  # pragma: no branch
            last_message = messages[-1]
            assert isinstance(last_message, ModelRequest), 'Expected last message to be a `ModelRequest`.'

            # check if there are any retry prompts, if so retry them
            new_retry_names = {p.tool_name for p in last_message.parts if isinstance(p, RetryPromptPart)}
            if new_retry_names:
                # Handle retries for both function tools and output tools
                # Check function tools first
                retry_parts: list[ModelResponsePart] = [
                    ToolCallPart(name, self.gen_tool_args(args)) for name, args in tool_calls if name in new_retry_names
                ]
                # Check output tools
                if output_tools:
                    retry_parts.extend(
                        [
                            ToolCallPart(
                                tool.name,
                                output_wrapper.value
                                if isinstance(output_wrapper, _WrappedToolOutput) and output_wrapper.value is not None
                                else self.gen_tool_args(tool),
                                tool_call_id=f'pyd_ai_tool_call_id__{tool.name}',
                            )
                            for tool in output_tools
                            if tool.name in new_retry_names
                        ]
                    )
                return ModelResponse(parts=retry_parts, model_name=self._model_name)

        if isinstance(output_wrapper, _WrappedTextOutput):
            if (response_text := output_wrapper.value) is None:
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
            assert output_tools, 'No output tools provided'
            custom_output_args = output_wrapper.value
            output_tool = output_tools[self.seed % len(output_tools)]
            if custom_output_args is not None:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            output_tool.name,
                            custom_output_args,
                            tool_call_id=f'pyd_ai_tool_call_id__{output_tool.name}',
                        )
                    ],
                    model_name=self._model_name,
                )
            else:
                response_args = self.gen_tool_args(output_tool)
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            output_tool.name,
                            response_args,
                            tool_call_id=f'pyd_ai_tool_call_id__{output_tool.name}',
                        )
                    ],
                    model_name=self._model_name,
                )


@dataclass
class TestStreamedResponse(StreamedResponse):
    """A structured response that streams test data."""

    _model_name: str
    _structured_response: ModelResponse
    _messages: InitVar[Iterable[ModelMessage]]
    _provider_name: str
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
                maybe_event = self._parts_manager.handle_text_delta(vendor_part_id=i, content='')
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event
                for word in words:
                    self._usage += _get_string_usage(word)
                    maybe_event = self._parts_manager.handle_text_delta(vendor_part_id=i, content=word)
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
            elif isinstance(part, ToolCallPart):
                yield self._parts_manager.handle_tool_call_part(
                    vendor_part_id=i, tool_name=part.tool_name, args=part.args, tool_call_id=part.tool_call_id
                )
            elif isinstance(part, BuiltinToolCallPart | BuiltinToolReturnPart):  # pragma: no cover
                # NOTE: These parts are not generated by TestModel, but we need to handle them for type checking
                assert False, f'Unexpected part type in TestModel: {type(part).__name__}'
            elif isinstance(part, ThinkingPart):  # pragma: no cover
                # NOTE: There's no way to reach this part of the code, since we don't generate ThinkingPart on TestModel.
                assert False, "This should be unreachable — we don't generate ThinkingPart on TestModel."
            elif isinstance(part, FilePart):  # pragma: no cover
                # NOTE: There's no way to reach this part of the code, since we don't generate FilePart on TestModel.
                assert False, "This should be unreachable — we don't generate FilePart on TestModel."
            else:
                assert_never(part)

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

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

    def generate(self) -> dict[str, Any]:
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


def _get_string_usage(text: str) -> RequestUsage:
    response_tokens = _estimate_string_tokens(text)
    return RequestUsage(output_tokens=response_tokens)
