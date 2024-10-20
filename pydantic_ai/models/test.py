"""
Utilities for testing apps build with pydantic_ai, specifically by using a model based which calls
local functions.
"""

from __future__ import annotations as _annotations

import re
import string
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pydantic_core

from .. import _utils
from ..messages import LLMMessage, LLMResponse, LLMToolCalls, Message, ToolCall, ToolRetry, ToolReturn
from . import AbstractToolDefinition, AgentModel, Model


@dataclass
class TestModel(Model):
    """
    A model specifically for testing purposes.

    This will (by default) call all retrievers in the agent model, then return a tool response if possible,
    otherwise a plain response.

    How useful this function will be is unknown, it may be useless, it may require significant changes to be useful.
    """

    # NOTE: Avoid test discovery by pytest.
    __test__ = False

    call_retrievers: list[str] | Literal['all'] = 'all'
    custom_result_text: str | None = None
    custom_result_args: Any | None = None

    def agent_model(
        self,
        retrievers: Mapping[str, AbstractToolDefinition],
        allow_text_result: bool,
        result_tool: AbstractToolDefinition | None,
    ) -> AgentModel:
        if self.call_retrievers == 'all':
            retriever_calls = [(r.name, r) for r in retrievers.values()]
        else:
            retrievers_to_call = (retrievers[name] for name in self.call_retrievers)
            retriever_calls = [(r.name, r) for r in retrievers_to_call]

        if self.custom_result_text is not None:
            assert allow_text_result, 'Plain response not allowed, but `custom_result_text` is set.'
            assert self.custom_result_args is None, 'Cannot set both `custom_result_text` and `custom_result_args`.'
            result: _utils.Either[str | None, Any | None] = _utils.Either(left=self.custom_result_text)
        elif self.custom_result_args is not None:
            assert result_tool is not None, 'No result tool name provided, but `custom_result_args` is set.'

            if k := result_tool.outer_typed_dict_key:
                result = _utils.Either(right={k: self.custom_result_args})
            else:
                result = _utils.Either(right=self.custom_result_args)
        elif result_tool is not None:
            result = _utils.Either(right=None)
        else:
            result = _utils.Either(left=None)
        return TestAgentModel(retriever_calls, result, result_tool)

    def name(self) -> str:
        return 'test-model'


@dataclass
class TestAgentModel(AgentModel):
    # NOTE: Avoid test discovery by pytest.
    __test__ = False

    retriever_calls: list[tuple[str, AbstractToolDefinition]]
    # left means the text is plain text; right means it's a function call
    result: _utils.Either[str | None, Any | None]
    result_tool: AbstractToolDefinition | None
    step: int = 0
    last_message_count: int = 0

    async def request(self, messages: list[Message]) -> LLMMessage:
        if self.step == 0:
            calls = [ToolCall.from_object(name, self.gen_retriever_args(args)) for name, args in self.retriever_calls]
            self.step += 1
            self.last_message_count = len(messages)
            return LLMToolCalls(calls=calls)

        new_messages = messages[self.last_message_count :]
        self.last_message_count = len(messages)
        new_retry_names = {m.tool_name for m in new_messages if isinstance(m, ToolRetry)}
        if new_retry_names:
            calls = [
                ToolCall.from_object(name, self.gen_retriever_args(args))
                for name, args in self.retriever_calls
                if name in new_retry_names
            ]
            self.step += 1
            return LLMToolCalls(calls=calls)
        else:
            if response_text := self.result.left:
                self.step += 1
                if response_text.value is None:
                    # build up details of retriever responses
                    output: dict[str, Any] = {}
                    for message in messages:
                        if isinstance(message, ToolReturn):
                            output[message.tool_name] = message.content
                    return LLMResponse(content=pydantic_core.to_json(output).decode())
                else:
                    return LLMResponse(content=response_text.value)
            else:
                assert self.result_tool is not None, 'No result tool name provided'
                custom_result_args = self.result.right
                if custom_result_args is not None:
                    self.step += 1
                    return LLMToolCalls(calls=[ToolCall.from_object(self.result_tool.name, custom_result_args)])
                else:
                    response_args = self.gen_retriever_args(self.result_tool)
                    self.step += 1
                    return LLMToolCalls(calls=[ToolCall.from_object(self.result_tool.name, response_args)])

    def gen_retriever_args(self, tool_def: AbstractToolDefinition) -> Any:
        """Generate arguments for a retriever."""
        return _JsonSchemaTestData(tool_def.json_schema, self.step).generate()


_chars = string.ascii_letters + string.digits + string.punctuation


class _JsonSchemaTestData:
    """
    Generate data that matches a JSON schema.

    This tries to generate the minimal viable data for the schema.
    """

    def __init__(self, schema: _utils.ObjectJsonSchema, seed: int = 0):
        self.schema = schema
        self.defs = schema.get('$defs', {})
        self.seed = seed

    def generate(self) -> Any:
        """Generate data for the JSON schema."""
        return self._gen_any(self.schema)  # pyright: ignore[reportArgumentType]

    def _gen_any(self, schema: dict[str, Any]) -> Any:
        """Generate data for any JSON Schema."""
        if const := schema.get('const'):
            return const
        elif enum := schema.get('enum'):
            return enum[0]
        elif examples := schema.get('examples'):
            return examples[0]
        elif ref := schema.get('$ref'):
            key = re.sub(r'^#/\$defs/', '', ref)
            js_def = self.defs[key]
            return self._gen_any(js_def)
        elif any_of := schema.get('anyOf'):
            if {'type': 'null'} in any_of:
                return None
            else:
                return self._gen_any(any_of[0])

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
        else:
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
