"""A minimal Python host for the Rust core.

The host does the HTTP and runs the tools. The core decides everything else: what to send, when to call a
tool, when the run is over.
"""

from __future__ import annotations

import ctypes
import inspect
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
from pydantic import ValidationError, validate_call
from pydantic_core import to_jsonable_python

from pydantic_ai import ModelRetry, Tool
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

__all__ = ['Agent', 'CoreError', 'RunResult', 'call_core']

_LIBRARY_NAMES = {'darwin': 'libpydantic_ai_core.dylib', 'win32': 'pydantic_ai_core.dll'}
_LIBRARY = Path(__file__).parent.parent / 'target/release' / _LIBRARY_NAMES.get(sys.platform, 'libpydantic_ai_core.so')

_core = ctypes.CDLL(str(_LIBRARY))
_core.pydantic_ai_core_call.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
_core.pydantic_ai_core_call.restype = ctypes.c_void_p
_core.pydantic_ai_core_dealloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_core.pydantic_ai_core_dealloc.restype = None


class CoreError(Exception):
    """The core rejected a call, or ended the run with an error."""


def call_core(request: dict[str, Any]) -> Any:
    """Send one JSON request to the core and return what it answers."""
    data = json.dumps(request).encode()
    response_len = ctypes.c_size_t()
    response_ptr = _core.pydantic_ai_core_call(data, len(data), ctypes.byref(response_len))
    try:
        response = json.loads(ctypes.string_at(response_ptr, response_len.value))
    finally:
        _core.pydantic_ai_core_dealloc(response_ptr, response_len.value)
    if 'error' in response:
        raise CoreError(response['error'])
    return response['ok']


@dataclass(frozen=True)
class RunResult:
    """The outcome of a run."""

    output: str
    messages: list[ModelMessage]
    usage: dict[str, int]


@dataclass
class Agent:
    """An agent whose run loop, request building and response parsing live in the Rust core."""

    model: str
    instructions: str | None = None
    base_url: str = 'https://api.openai.com/v1'
    api_key: str | None = None
    http_client: httpx2.AsyncClient = field(default_factory=httpx2.AsyncClient)
    _tools: dict[str, Callable[..., Any]] = field(default_factory=dict[str, Callable[..., Any]])

    def tool_plain(self, function: Callable[..., Any]) -> Callable[..., Any]:
        """Register a function the model can call."""
        self._tools[function.__name__] = function
        return function

    async def run(self, user_prompt: str, *, message_history: list[ModelMessage] | None = None) -> RunResult:
        """Run the agent until the model answers with text."""
        tool_defs = [Tool(function).tool_def for function in self._tools.values()]
        event: dict[str, Any] = {
            'type': 'start',
            'model': self.model,
            'instructions': self.instructions,
            'user_prompt': user_prompt,
            'tools': [
                {
                    'name': tool_def.name,
                    'description': tool_def.description,
                    'parameters_json_schema': tool_def.parameters_json_schema,
                    'strict': tool_def.strict,
                }
                for tool_def in tool_defs
            ],
            'message_history': ModelMessagesTypeAdapter.dump_python(message_history or [], mode='json'),
        }
        state: str | None = None
        while True:
            step = call_core({'fn': 'run_step', 'state': state, 'event': event})
            state, action = step['state'], step['action']
            if action['type'] == 'send_request':
                event = await self._send(action['request'])
            elif action['type'] == 'call_tools':
                event = {'type': 'tool_results', 'results': [await self._call_tool(call) for call in action['calls']]}
            else:
                messages = ModelMessagesTypeAdapter.validate_python(action['messages'])
                return RunResult(output=action['output'], messages=messages, usage=action['usage'])

    async def _send(self, request: dict[str, str]) -> dict[str, Any]:
        api_key = self.api_key or os.environ['OPENAI_API_KEY']
        response = await self.http_client.request(
            request['method'],
            self.base_url + request['path'],
            content=request['body'],
            headers={'authorization': f'Bearer {api_key}', 'content-type': 'application/json'},
        )
        return {'type': 'response', 'status': response.status_code, 'body': response.text}

    async def _call_tool(self, call: dict[str, str]) -> dict[str, Any]:
        result: dict[str, Any] = {'tool_call_id': call['tool_call_id']}
        function = self._tools.get(call['tool_name'])
        if function is None:
            return {**result, 'retry': f'Unknown tool name: {call["tool_name"]!r}.'}
        try:
            value = validate_call(function)(**json.loads(call['args'] or '{}'))
            if inspect.isawaitable(value):
                value = await value
        except ModelRetry as retry:
            return {**result, 'retry': retry.message}
        except (ValidationError, json.JSONDecodeError) as invalid:
            return {**result, 'retry': str(invalid)}
        return {**result, 'content': to_jsonable_python(value)}
