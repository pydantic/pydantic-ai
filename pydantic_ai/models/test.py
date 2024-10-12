"""
Utilities for testing apps build with pydantic_ai, specifically by using a model based which calls
local functions.
"""

from __future__ import annotations as _annotations

import json
import re
import string
from dataclasses import dataclass
from typing import Any, Literal

from .. import _utils
from ..messages import FunctionCall, LLMFunctionCalls, LLMMessage, LLMResponse, Message
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
    custom_response_text: str | None = None
    custom_response_args: Any | None = None

    def agent_model(self, allow_plain_response: bool, tools: list[AbstractToolDefinition]) -> AgentModel:
        if self.call_retrievers == 'all':
            retriever_calls = [(r.name, gen_retriever_args(r)) for r in tools if r.name != 'response']
        else:
            lookup = {r.name: r for r in tools}
            retriever_calls = [(name, gen_retriever_args(lookup[name])) for name in self.call_retrievers]

        if self.custom_response_text is not None:
            if not allow_plain_response:
                raise ValueError('Plain response not allowed, but `custom_response_text` is set.')
            final_response: _utils.Either[str, str] = _utils.Either(left=self.custom_response_text)
        elif self.custom_response_args is not None:
            response_def = next((r for r in tools if r.name == 'response'), None)
            if response_def is None:
                raise ValueError('Custom response arguments provided, but no response tool found.')
            final_response = _utils.Either(right=self.custom_response_args)
        else:
            if response_def := next((r for r in tools if r.name == 'response'), None):
                final_response = _utils.Either(right=gen_retriever_args(response_def))
            else:
                final_response = _utils.Either(left='Final response')
        return TestAgentModel(retriever_calls, final_response)


@dataclass
class TestAgentModel(AgentModel):
    # NOTE: Avoid test discovery by pytest.
    __test__ = False

    retriever_calls: list[tuple[str, str]]
    # left means the final response is plain text, right means it's a function call
    final_response: _utils.Either[str, str]
    step: int = 0

    async def request(self, messages: list[Message]) -> LLMMessage:
        if self.step == 0:
            self.step += 1
            return LLMFunctionCalls(
                calls=[
                    FunctionCall(function_id=name, function_name=name, arguments=args)
                    for name, args in self.retriever_calls
                ]
            )
        elif self.step == 1:
            self.step += 1
            if response_text := self.final_response.left:
                return LLMResponse(content=response_text)
            else:
                response_args = self.final_response.right
                return LLMFunctionCalls(
                    calls=[FunctionCall(function_id='response', function_name='response', arguments=response_args)]
                )
        else:
            raise ValueError('Invalid step')


def gen_retriever_args(tool_def: AbstractToolDefinition) -> str:
    """Generate arguments for a retriever."""
    return _JsonSchemaTestData(tool_def.json_schema).generate_json()


_chars = string.ascii_letters + string.digits + string.punctuation


class _JsonSchemaTestData:
    """
    Generate data that matches a JSON schema.

    This tries to generate the minimal viable data for the schema.
    """

    def __init__(self, schema: _utils.ObjectJsonSchema):
        self.schema = schema
        self.defs = schema.get('$defs', {})
        self.seed = 0

    def generate(self) -> Any:
        """Generate data for the JSON schema."""
        return self._gen_any(self.schema)  # pyright: ignore[reportArgumentType]

    def generate_json(self) -> str:
        return json.dumps(self.generate())

    def _gen_any(self, schema: dict[str, Any]) -> Any:
        """Generate data for any JSON Schema."""
        if const := schema.get('const'):
            return const
        elif enum := schema.get('enum'):
            return enum[0]
        elif examples := schema.get('examples'):
            return examples[0]

        type_ = schema.get('type')
        if type_ is None:
            if ref := schema.get('$ref'):
                key = re.sub(r'^#/\$defs/', '', ref)
                js_def = self.defs[key]
                return self._gen_any(js_def)
            else:
                # if there's no type or ref, we can't generate anything
                return self._char()

        if type_ == 'object':
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
                if unique_items:
                    self.seed += 1
                data.append(self._gen_any(item))

        items_schema = schema.get('items', {})
        min_items = schema.get('minItems', 0)
        if min_items > len(data):
            for _ in range(min_items - len(data)):
                if unique_items:
                    self.seed += 1
                data.append(self._gen_any(items_schema))
        elif items_schema:
            # if there is an `items` schema, add an item if minItems doesn't require it
            # unless it would break `maxItems` rule
            max_items = schema.get('maxItems')
            if max_items is None or max_items > len(data):
                if unique_items:
                    self.seed += 1
                data.append(self._gen_any(items_schema))

        return data

    def _char(self) -> str:
        """Generate a character on the same principle as Excel columns, e.g. a-z, aa-az..."""
        chars = len(_chars)
        s = ''
        rem = self.seed // chars
        while rem > 0:
            s += _chars[rem % chars]
            rem //= chars
        s += _chars[self.seed % chars]
        return s
