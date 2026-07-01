"""Experimental Gemini model backed by Google's Interactions API.

This is an **experimental, proof-of-concept** model that talks to Google's new
[Interactions API](https://ai.google.dev/gemini-api/docs/interactions)
(`client.interactions.create`) instead of the classic `generateContent` endpoint.

Limitations (subject to change — this is a draft, see <https://github.com/pydantic/pydantic-ai/issues/6192>):

- **Gemini API (GLA) only.** The Interactions API is not yet available on Vertex AI or the
  Pydantic AI Gateway, so this model only works with the `google` / `google-gla` provider.
- **Gemini 2.5 / 3.x only.** The endpoint rejects Gemini 2.0 and older model families.
- Text in, text + tool-calls out. Multimodal user content, native/builtin tools, structured
  and tool output modes, safety settings, and token counting are **not** supported yet and
  raise a clear error rather than silently degrading.

It is intentionally not wired into provider inference, `KnownModelName`, or the public package
exports; import it explicitly via
`from pydantic_ai.models.google_interactions import GoogleInteractionsModel`.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from typing_extensions import assert_never

from .. import _utils, usage
from .._run_context import RunContext
from ..exceptions import ModelAPIError, ModelHTTPError, UserError
from ..messages import (
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..settings import ModelSettings, ThinkingEffort
from . import ModelRequestParameters, StreamedResponse, check_allow_model_requests
from .google import (
    _GEMINI_API_PROVIDER_NAMES,  # pyright: ignore[reportPrivateUsage]
    GoogleModel,
    GoogleModelName,
    GoogleModelSettings,
)

try:
    from google.genai import errors

    # `google.genai.interactions` rebuilds `__all__` dynamically, so pyright can't see this
    # re-export even though it's the public import surface for the Interactions types.
    from google.genai.interactions import Interaction  # pyright: ignore[reportPrivateImportUsage]
except ImportError as _import_error:
    raise ImportError(
        'Please install `google-genai>=2.0` to use the experimental GoogleInteractionsModel, '
        'you can use the `google` optional group — `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error


# The Interactions API reports an interaction-level `status` instead of a per-candidate finish
# reason, and drops the fine-grained `MAX_TOKENS` / `SAFETY` / `RECITATION` granularity.
# `requires_action` means a function call is pending (normal control flow), which maps to `tool_call`.
_STATUS_TO_FINISH_REASON: dict[str, FinishReason | None] = {
    'completed': 'stop',
    'incomplete': 'length',
    'budget_exceeded': 'length',
    'failed': 'error',
    'requires_action': 'tool_call',
    'cancelled': None,
    'in_progress': None,
}

_THINKING_EFFORT_TO_LEVEL: dict[ThinkingEffort, str] = {
    'minimal': 'minimal',
    'low': 'low',
    'medium': 'medium',
    'high': 'high',
    'xhigh': 'high',  # no higher level available on the Interactions API
}


@dataclass(init=False)
class GoogleInteractionsModel(GoogleModel):
    """Experimental Gemini model using Google's Interactions API (`client.interactions.create`).

    Subclasses [`GoogleModel`][pydantic_ai.models.google.GoogleModel] to reuse its `__init__`,
    provider wiring, client, and profile, but overrides the request/response mapping to speak the
    Interactions `steps` timeline instead of `generateContent`.

    **Experimental and GLA-only.** See the [module docstring][pydantic_ai.models.google_interactions]
    for the full list of limitations. Tracks <https://github.com/pydantic/pydantic-ai/issues/6192>.
    """

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(model_settings, model_request_parameters)
        settings = cast(GoogleModelSettings, model_settings or {})
        self._check_supported(settings, model_request_parameters)

        system_instruction, input_steps = self._build_input(messages, model_request_parameters)
        tools = self._build_function_tools(model_request_parameters)
        generation_config = self._build_generation_config(settings, model_request_parameters)

        try:
            interaction = await self.client.aio.interactions.create(
                model=self._model_name,
                store=False,
                input=input_steps,
                system_instruction=system_instruction,
                tools=tools,
                generation_config=generation_config,
            )
        except errors.APIError as e:
            raise _map_api_error(e, self._model_name) from e

        if not isinstance(interaction, Interaction):  # pragma: no cover
            raise ModelAPIError(model_name=self._model_name, message='Expected a non-streamed Interaction response')
        return self._process_interaction(interaction)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(model_settings, model_request_parameters)
        settings = cast(GoogleModelSettings, model_settings or {})
        self._check_supported(settings, model_request_parameters)

        system_instruction, input_steps = self._build_input(messages, model_request_parameters)
        tools = self._build_function_tools(model_request_parameters)
        generation_config = self._build_generation_config(settings, model_request_parameters)

        try:
            stream = await self.client.aio.interactions.create(
                model=self._model_name,
                store=False,
                stream=True,
                input=input_steps,
                system_instruction=system_instruction,
                tools=tools,
                generation_config=generation_config,
            )
        except errors.APIError as e:
            raise _map_api_error(e, self._model_name) from e

        if isinstance(stream, Interaction):  # pragma: no cover
            raise ModelAPIError(model_name=self._model_name, message='Expected a streamed Interaction response')
        try:
            yield GoogleInteractionsStreamedResponse(
                model_request_parameters=model_request_parameters,
                _model_name=self._model_name,
                _provider_name=self.system,
                _provider_url=self.base_url,
                _response=stream,
            )
        finally:
            await stream.close()

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> usage.RequestUsage:
        raise NotImplementedError(
            'The Interactions API does not expose a token-counting endpoint, so '
            '`count_tokens` is not supported by the experimental GoogleInteractionsModel.'
        )

    def _check_supported(self, settings: GoogleModelSettings, model_request_parameters: ModelRequestParameters) -> None:
        """Raise a clear error for inputs this proof-of-concept does not support yet."""
        if self.system not in _GEMINI_API_PROVIDER_NAMES:
            raise UserError(
                'GoogleInteractionsModel only supports the Gemini API (GLA) provider (`google` / `google-gla`); '
                f'the Interactions API is not available on {self.system!r}. Use `GoogleModel` instead.'
            )
        if settings.get('google_safety_settings'):
            raise UserError('`google_safety_settings` is not supported by the experimental GoogleInteractionsModel.')
        if model_request_parameters.native_tools:
            raise UserError('Native (builtin) tools are not supported by the experimental GoogleInteractionsModel yet.')
        if model_request_parameters.output_mode != 'text':
            raise UserError(
                'Structured and tool output modes are not supported by the experimental GoogleInteractionsModel yet; '
                'use plain text output.'
            )

    def _build_input(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Map pydantic-ai message history to Interactions `input` steps + `system_instruction`."""
        system_parts: list[str] = []
        input_steps: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        system_parts.append(part.content)
                    elif isinstance(part, UserPromptPart):
                        input_steps.append(self._user_input_step(part))
                    elif isinstance(part, ToolReturnPart):
                        input_steps.append(
                            {
                                'type': 'function_result',
                                'call_id': part.tool_call_id,
                                'name': part.tool_name,
                                'result': [{'type': 'text', 'text': part.model_response_str()}],
                            }
                        )
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            input_steps.append(
                                {'type': 'user_input', 'content': [{'type': 'text', 'text': part.model_response()}]}
                            )
                        else:
                            input_steps.append(
                                {
                                    'type': 'function_result',
                                    'call_id': part.tool_call_id,
                                    'name': part.tool_name,
                                    'result': [{'type': 'text', 'text': part.model_response()}],
                                    'is_error': True,
                                }
                            )
                    else:
                        assert_never(part)
            elif isinstance(message, ModelResponse):
                input_steps.extend(self._model_response_steps(message))
            else:
                assert_never(message)

        if instruction_parts := self._get_instruction_parts(messages, model_request_parameters):
            system_parts.extend(p.content for p in instruction_parts)

        system_instruction = '\n\n'.join(system_parts) if system_parts else None
        return system_instruction, input_steps

    def _user_input_step(self, part: UserPromptPart) -> dict[str, Any]:
        if isinstance(part.content, str):
            text = part.content
        else:
            texts: list[str] = []
            for item in part.content:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, TextContent):
                    texts.append(item.content)
                else:
                    raise UserError(
                        'Multimodal user content is not supported by the experimental GoogleInteractionsModel yet; '
                        f'got {type(item).__name__}.'
                    )
            text = ''.join(texts)
        return {'type': 'user_input', 'content': [{'type': 'text', 'text': text}]}

    def _model_response_steps(self, message: ModelResponse) -> list[dict[str, Any]]:
        """Map a stored `ModelResponse` back into Interactions input steps, echoing thought signatures.

        Each `function_call` step's `signature` MUST be echoed back or the Interactions API returns
        a 400, so we recover it from `ToolCallPart.provider_details['thought_signature']` (and the
        thought signature from `ThinkingPart.signature`).
        """
        steps: list[dict[str, Any]] = []
        for part in message.parts:
            if isinstance(part, TextPart):
                if part.content:
                    steps.append({'type': 'model_output', 'content': [{'type': 'text', 'text': part.content}]})
            elif isinstance(part, ThinkingPart):
                step: dict[str, Any] = {'type': 'thought'}
                if part.signature:
                    step['signature'] = part.signature
                if part.content:
                    step['summary'] = [{'type': 'text', 'text': part.content}]
                # Only echo a thought step that carries a signature or summary; a bare thought is meaningless.
                if len(step) > 1:
                    steps.append(step)
            elif isinstance(part, ToolCallPart):
                step = {
                    'type': 'function_call',
                    'id': part.tool_call_id,
                    'name': part.tool_name,
                    'arguments': part.args_as_dict(),
                }
                if part.provider_details and (signature := part.provider_details.get('thought_signature')):
                    step['signature'] = signature
                steps.append(step)
            else:
                # Native-tool, file, and compaction parts are unsupported in this proof-of-concept and
                # are rejected on the request path before we ever store them, so drop them here.
                continue
        return steps

    def _build_function_tools(self, model_request_parameters: ModelRequestParameters) -> list[dict[str, Any]] | None:
        tools = [
            {
                'type': 'function',
                'name': tool.name,
                'description': tool.description or '',
                'parameters': tool.parameters_json_schema,
            }
            for tool in model_request_parameters.function_tools
        ]
        return tools or None

    def _build_generation_config(
        self, settings: GoogleModelSettings, model_request_parameters: ModelRequestParameters
    ) -> dict[str, Any]:
        generation_config: dict[str, Any] = {}
        if (temperature := settings.get('temperature')) is not None:
            generation_config['temperature'] = temperature
        if (top_p := settings.get('top_p')) is not None:
            generation_config['top_p'] = top_p
        if (seed := settings.get('seed')) is not None:
            generation_config['seed'] = seed
        if (stop_sequences := settings.get('stop_sequences')) is not None:
            generation_config['stop_sequences'] = stop_sequences
        if (max_tokens := settings.get('max_tokens')) is not None:
            generation_config['max_output_tokens'] = max_tokens
        if (presence_penalty := settings.get('presence_penalty')) is not None:
            generation_config['presence_penalty'] = presence_penalty
        if (frequency_penalty := settings.get('frequency_penalty')) is not None:
            generation_config['frequency_penalty'] = frequency_penalty

        thinking = model_request_parameters.thinking
        if thinking is False:
            generation_config['thinking_level'] = 'minimal'
        elif thinking is True:
            generation_config['thinking_summaries'] = 'auto'
        elif thinking is not None:
            generation_config['thinking_level'] = _THINKING_EFFORT_TO_LEVEL[thinking]
            generation_config['thinking_summaries'] = 'auto'

        return generation_config

    def _process_interaction(self, interaction: Interaction) -> ModelResponse:
        step_dicts = [step.model_dump(mode='python', exclude_none=True) for step in interaction.steps or []]
        parts = _steps_to_parts(step_dicts, self.system)

        status = str(interaction.status)
        finish_reason = _STATUS_TO_FINISH_REASON.get(status)
        provider_details: dict[str, Any] | None = {'status': status} if status else None

        usage_dict = interaction.usage.model_dump(exclude_none=True) if interaction.usage else None

        return ModelResponse(
            parts=parts,
            model_name=interaction.model or self._model_name,
            usage=_usage_from_dict(usage_dict),
            provider_response_id=interaction.id,
            provider_details=provider_details,
            provider_name=self.system,
            provider_url=self.base_url,
            finish_reason=finish_reason,
        )


def _map_api_error(e: errors.APIError, model_name: str) -> ModelAPIError:
    if (status_code := e.code) >= 400:
        return ModelHTTPError(
            status_code=status_code,
            model_name=model_name,
            body=cast(Any, e.details),  # pyright: ignore[reportUnknownMemberType]
        )
    return ModelAPIError(model_name=model_name, message=str(e))


def _steps_to_parts(steps: list[dict[str, Any]], provider_name: str) -> list[ModelResponsePart]:
    """Convert response `steps` (as dicts) to `ModelResponse` parts. Echoed input steps are skipped."""
    parts: list[ModelResponsePart] = []
    for step in steps:
        step_type = step.get('type')
        if step_type == 'thought':
            summary = _join_text(step.get('summary'))
            parts.append(ThinkingPart(content=summary, signature=step.get('signature'), provider_name=provider_name))
        elif step_type == 'model_output':
            text = _join_text(step.get('content'))
            if text:
                parts.append(TextPart(content=text))
        elif step_type == 'function_call':
            signature = step.get('signature')
            provider_details = {'thought_signature': signature} if signature else None
            parts.append(
                ToolCallPart(
                    tool_name=step.get('name', ''),
                    args=step.get('arguments'),
                    tool_call_id=step.get('id', ''),
                    provider_name=provider_name if provider_details else None,
                    provider_details=provider_details,
                )
            )
        else:
            # `user_input` / `function_result` echo the request; native-tool and unknown steps
            # aren't produced by this proof-of-concept. Skip them.
            continue
    return parts


def _join_text(contents: list[dict[str, Any]] | None) -> str:
    """Join the `text` of `text`-typed content blocks (thought summary / model output)."""
    if not contents:
        return ''
    return ''.join(c['text'] for c in contents if c.get('type') == 'text' and c.get('text'))


def _usage_from_dict(metadata: dict[str, Any] | None) -> usage.RequestUsage:
    if not metadata:
        return usage.RequestUsage()
    details: dict[str, int] = {}
    for key in ('total_thought_tokens', 'total_tool_use_tokens', 'total_tokens'):
        if value := metadata.get(key):
            details[key] = value
    return usage.RequestUsage(
        input_tokens=metadata.get('total_input_tokens') or 0,
        output_tokens=metadata.get('total_output_tokens') or 0,
        cache_read_tokens=metadata.get('total_cached_tokens') or 0,
        details=details,
    )


@dataclass
class GoogleInteractionsStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for the experimental Interactions API model."""

    _model_name: GoogleModelName
    _provider_name: str
    _provider_url: str
    _response: AsyncIterator[Any]
    _timestamp: datetime = field(default_factory=_utils.now_utc)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        # Interactions streams index-keyed step deltas; `step.start` tells us each index's step type
        # (and, for `function_call`, its id / name / signature) so `step.delta` events can be routed.
        step_types: dict[int, str] = {}
        try:
            async for event in self._response:
                data: dict[str, Any] = event.model_dump(mode='python')
                event_type = data.get('event_type')
                if event_type == 'step.start':
                    index = data['index']
                    step: dict[str, Any] = data.get('step') or {}
                    step_type = step.get('type', '')
                    step_types[index] = step_type
                    if step_type == 'function_call':
                        signature = step.get('signature')
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=index,
                            tool_name=step.get('name'),
                            tool_call_id=step.get('id'),
                            provider_name=self._provider_name if signature else None,
                            provider_details={'thought_signature': signature} if signature else None,
                        )
                        if maybe_event is not None:
                            yield maybe_event
                elif event_type == 'step.delta':
                    index = data['index']
                    for e in self._handle_delta(index, data.get('delta') or {}, step_types):
                        yield e
                elif event_type == 'interaction.completed':
                    interaction: dict[str, Any] = data.get('interaction') or {}
                    self._usage = _usage_from_dict(interaction.get('usage'))
                    if interaction_id := interaction.get('id'):
                        self.provider_response_id = interaction_id
                    if status := interaction.get('status'):
                        status = str(status)
                        self.provider_details = {**(self.provider_details or {}), 'status': status}
                        self.finish_reason = _STATUS_TO_FINISH_REASON.get(status)
        except errors.APIError as e:
            raise _map_api_error(e, self._model_name) from e

    def _handle_delta(
        self, index: int, delta: dict[str, Any], step_types: dict[int, str]
    ) -> Iterator[ModelResponseStreamEvent]:
        delta_type = delta.get('type')
        if delta_type == 'text':
            text = delta.get('text') or ''
            if step_types.get(index) == 'thought':
                yield from self._parts_manager.handle_thinking_delta(vendor_part_id=index, content=text)
            else:
                yield from self._parts_manager.handle_text_delta(vendor_part_id=index, content=text)
        elif delta_type == 'thought_summary':
            yield from self._parts_manager.handle_thinking_delta(
                vendor_part_id=index, content=_join_text(delta.get('content'))
            )
        elif delta_type == 'thought_signature':
            yield from self._parts_manager.handle_thinking_delta(
                vendor_part_id=index, signature=delta.get('signature'), provider_name=self._provider_name
            )
        elif delta_type == 'arguments_delta':
            maybe_event = self._parts_manager.handle_tool_call_delta(vendor_part_id=index, args=delta.get('arguments'))
            if maybe_event is not None:
                yield maybe_event

    @property
    def model_name(self) -> GoogleModelName:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def provider_url(self) -> str:
        return self._provider_url

    @property
    def timestamp(self) -> datetime:
        return self._timestamp
