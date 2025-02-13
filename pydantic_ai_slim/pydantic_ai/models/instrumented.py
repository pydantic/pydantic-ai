from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import logfire_api

from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..settings import ModelSettings
from ..usage import Usage
from . import ModelRequestParameters, StreamedResponse
from .wrapper import WrapperModel

MODEL_SETTING_ATTRIBUTES: tuple[
    Literal[
        'max_tokens',
        'top_p',
        'seed',
        'temperature',
        'presence_penalty',
        'frequency_penalty',
    ],
    ...,
] = (
    'max_tokens',
    'top_p',
    'seed',
    'temperature',
    'presence_penalty',
    'frequency_penalty',
)

NOT_GIVEN = object()


@dataclass
class InstrumentedModel(WrapperModel):
    """Model which is instrumented with logfire."""

    logfire_instance: logfire_api.Logfire = logfire_api.DEFAULT_LOGFIRE_INSTANCE

    def __post_init__(self):
        self.logfire_instance = self.logfire_instance.with_settings(custom_scope_suffix='pydantic_ai')

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        with self._instrument(messages, model_settings) as finish:
            response, usage = await super().request(messages, model_settings, model_request_parameters)
            finish(response, usage)
            return response, usage

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        with self._instrument(messages, model_settings) as finish:
            response_stream: StreamedResponse | None = None
            try:
                async with super().request_stream(
                    messages, model_settings, model_request_parameters
                ) as response_stream:
                    yield response_stream
            finally:
                if response_stream:
                    finish(response_stream.get(), response_stream.usage())

    @contextmanager
    def _instrument(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ):
        operation = 'chat'
        model_name = self.model_name
        span_name = f'{operation} {model_name}'
        system = getattr(self.wrapped, 'system', '') or self.wrapped.__class__.__name__.removesuffix('Model').lower()
        system = {'google-gla': 'gemini', 'google-vertex': 'vertex_ai', 'mistral': 'mistral_ai'}.get(system, system)
        # TODO Missing attributes:
        #  - server.address: requires a Model.base_url abstract method or similar
        #  - server.port: to parse from the base_url
        #  - error.type: unclear if we should do something here or just always rely on span exceptions
        #  - gen_ai.request.stop_sequences/top_k: model_settings doesn't include these
        attributes: dict[str, Any] = {
            'gen_ai.operation.name': operation,
            'gen_ai.system': system,
            'gen_ai.request.model': model_name,
        }

        if model_settings:
            for key in MODEL_SETTING_ATTRIBUTES:
                if (value := model_settings.get(key, NOT_GIVEN)) is not NOT_GIVEN:
                    attributes[f'gen_ai.request.{key}'] = value

        emit_event = partial(self._emit_event, system)

        with self.logfire_instance.span(span_name, **attributes) as span:
            if span.is_recording():
                for message in messages:
                    if isinstance(message, ModelRequest):
                        for part in message.parts:
                            event_name, body = _request_part_body(part)
                            if event_name:
                                emit_event(event_name, body)
                    elif isinstance(message, ModelResponse):
                        for body in _response_bodies(message):
                            emit_event('gen_ai.assistant.message', body)

            def finish(response: ModelResponse, usage: Usage):
                if not span.is_recording():
                    return

                for response_body in _response_bodies(response):
                    if response_body:
                        emit_event(
                            'gen_ai.choice',
                            {
                                # TODO finish_reason
                                'index': 0,
                                'message': response_body,
                            },
                        )
                span.set_attributes(
                    {
                        k: v
                        for k, v in {
                            # TODO finish_reason (https://github.com/open-telemetry/semantic-conventions/issues/1277), id
                            #  https://github.com/pydantic/pydantic-ai/issues/886
                            'gen_ai.response.model': response.model_name or model_name,
                            'gen_ai.usage.input_tokens': usage.request_tokens,
                            'gen_ai.usage.output_tokens': usage.response_tokens,
                        }.items()
                        if v is not None
                    }
                )

            yield finish

    def _emit_event(self, system: str, event_name: str, body: dict[str, Any]) -> None:
        self.logfire_instance.info(event_name, **{'gen_ai.system': system}, **body)


def _request_part_body(part: ModelRequestPart) -> tuple[str, dict[str, Any]]:
    if isinstance(part, SystemPromptPart):
        return 'gen_ai.system.message', {'content': part.content}
    elif isinstance(part, UserPromptPart):
        return 'gen_ai.user.message', {'content': part.content}
    elif isinstance(part, ToolReturnPart):
        return 'gen_ai.tool.message', {'content': part.content, 'id': part.tool_call_id}
    elif isinstance(part, RetryPromptPart):
        if part.tool_name is None:
            return 'gen_ai.user.message', {'content': part.model_response()}
        else:
            return 'gen_ai.tool.message', {'content': part.model_response(), 'id': part.tool_call_id}
    else:
        return '', {}


def _response_bodies(message: ModelResponse) -> list[dict[str, Any]]:
    body: dict[str, Any] = {}
    result = [body]
    for part in message.parts:
        if isinstance(part, ToolCallPart):
            body.setdefault('tool_calls', []).append(
                {
                    'id': part.tool_call_id,
                    'type': 'function',  # TODO https://github.com/pydantic/pydantic-ai/issues/888
                    'function': {
                        'name': part.tool_name,
                        'arguments': part.args,
                    },
                }
            )
        elif isinstance(part, TextPart):
            if body.get('content'):
                body = {}
                result.append(body)
            body['content'] = part.content

    return result
