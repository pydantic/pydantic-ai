from __future__ import annotations as _annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, cast

from typing_extensions import assert_never

from pydantic_ai.exceptions import UserError

from .. import ModelHTTPError, usage
from .._utils import generate_tool_call_id as _generate_tool_call_id, guard_tool_call_id as _guard_tool_call_id
from ..messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..profiles import ModelProfileSpec
from ..providers import Provider, infer_provider
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import Model, ModelRequestParameters, check_allow_model_requests

try:
    from cohere import (
        AssistantChatMessageV2,
        AssistantMessageV2ContentItem,
        AsyncClientV2,
        ChatFinishReason,
        ChatMessageV2,
        SystemChatMessageV2,
        TextAssistantMessageV2ContentItem,
        ThinkingAssistantMessageV2ContentItem,
        ToolCallV2,
        ToolCallV2Function,
        ToolChatMessageV2,
        ToolV2,
        ToolV2Function,
        UserChatMessageV2,
        V2ChatResponse,
    )
    from cohere.core.api_error import ApiError
    from cohere.v2.client import OMIT
except ImportError as _import_error:
    raise ImportError(
        'Please install `cohere` to use the Cohere model, '
        'you can use the `cohere` optional group — `pip install "pydantic-ai-slim[cohere]"`'
    ) from _import_error

LatestCohereModelNames = Literal[
    'c4ai-aya-expanse-32b',
    'c4ai-aya-expanse-8b',
    'command-nightly',
    'command-r-08-2024',
    'command-r-plus-08-2024',
    'command-r7b-12-2024',
]
"""Latest Cohere models."""

CohereModelName = str | LatestCohereModelNames
"""Possible Cohere model names.

Since Cohere supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [Cohere's docs](https://docs.cohere.com/v2/docs/models) for a list of all available models.
"""

_FINISH_REASON_MAP: dict[ChatFinishReason, FinishReason] = {
    'COMPLETE': 'stop',
    'STOP_SEQUENCE': 'stop',
    'MAX_TOKENS': 'length',
    'TOOL_CALL': 'tool_call',
    'ERROR': 'error',
}


class CohereModelSettings(ModelSettings, total=False):
    """Settings used for a Cohere model request."""

    # ALL FIELDS MUST BE `cohere_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    # This class is a placeholder for any future cohere-specific settings


@dataclass(init=False)
class CohereModel(Model):
    """A model that uses the Cohere API.

    Internally, this uses the [Cohere Python client](
    https://github.com/cohere-ai/cohere-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncClientV2 = field(repr=False)

    _model_name: CohereModelName = field(repr=False)
    _provider: Provider[AsyncClientV2] = field(repr=False)

    def __init__(
        self,
        model_name: CohereModelName,
        *,
        provider: Literal['cohere'] | Provider[AsyncClientV2] = 'cohere',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an Cohere model.

        Args:
            model_name: The name of the Cohere model to use. List of model names
                available [here](https://docs.cohere.com/docs/models#command).
            provider: The provider to use for authentication and API access. Can be either the string
                'cohere' or an instance of `Provider[AsyncClientV2]`. If not provided, a new provider will be
                created using the other parameters.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        client_wrapper = self.client._client_wrapper  # type: ignore
        return str(client_wrapper.get_base_url())

    @property
    def model_name(self) -> CohereModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._chat(messages, cast(CohereModelSettings, model_settings or {}), model_request_parameters)
        model_response = self._process_response(response)
        return model_response

    async def _chat(
        self,
        messages: list[ModelMessage],
        model_settings: CohereModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> V2ChatResponse:
        tools = self._get_tools(model_request_parameters)

        if model_request_parameters.builtin_tools:
            raise UserError('Cohere does not support built-in tools')

        cohere_messages = self._map_messages(messages, model_request_parameters)
        try:
            return await self.client.chat(
                model=self._model_name,
                messages=cohere_messages,
                tools=tools or OMIT,
                max_tokens=model_settings.get('max_tokens', OMIT),
                stop_sequences=model_settings.get('stop_sequences', OMIT),
                temperature=model_settings.get('temperature', OMIT),
                p=model_settings.get('top_p', OMIT),
                seed=model_settings.get('seed', OMIT),
                presence_penalty=model_settings.get('presence_penalty', OMIT),
                frequency_penalty=model_settings.get('frequency_penalty', OMIT),
            )
        except ApiError as e:
            if (status_code := e.status_code) and status_code >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover

    def _process_response(self, response: V2ChatResponse) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        parts: list[ModelResponsePart] = []
        if response.message.content is not None:
            for content in response.message.content:
                if content.type == 'text':
                    parts.append(TextPart(content=content.text))
                elif content.type == 'thinking':  # pragma: no branch
                    parts.append(ThinkingPart(content=content.thinking))
        for c in response.message.tool_calls or []:
            if c.function and c.function.name and c.function.arguments:  # pragma: no branch
                parts.append(
                    ToolCallPart(
                        tool_name=c.function.name,
                        args=c.function.arguments,
                        tool_call_id=c.id or _generate_tool_call_id(),
                    )
                )

        raw_finish_reason = response.finish_reason
        provider_details = {'finish_reason': raw_finish_reason}
        finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

        return ModelResponse(
            parts=parts,
            usage=_map_usage(response),
            model_name=self._model_name,
            provider_name=self._provider.name,
            finish_reason=finish_reason,
            provider_details=provider_details,
        )

    def _map_messages(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> list[ChatMessageV2]:
        """Just maps a `pydantic_ai.Message` to a `cohere.ChatMessageV2`."""
        cohere_messages: list[ChatMessageV2] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                cohere_messages.extend(self._map_user_message(message))
            elif isinstance(message, ModelResponse):
                texts: list[str] = []
                thinking: list[str] = []
                tool_calls: list[ToolCallV2] = []
                for item in message.parts:
                    if isinstance(item, TextPart):
                        texts.append(item.content)
                    elif isinstance(item, ThinkingPart):
                        thinking.append(item.content)
                    elif isinstance(item, ToolCallPart):
                        tool_calls.append(self._map_tool_call(item))
                    elif isinstance(item, BuiltinToolCallPart | BuiltinToolReturnPart):  # pragma: no cover
                        # This is currently never returned from cohere
                        pass
                    elif isinstance(item, FilePart):  # pragma: no cover
                        # Files generated by models are not sent back to models that don't themselves generate files.
                        pass
                    else:
                        assert_never(item)

                message_param = AssistantChatMessageV2(role='assistant')
                if texts or thinking:
                    contents: list[AssistantMessageV2ContentItem] = []
                    if thinking:
                        contents.append(ThinkingAssistantMessageV2ContentItem(thinking='\n\n'.join(thinking)))
                    if texts:  # pragma: no branch
                        contents.append(TextAssistantMessageV2ContentItem(text='\n\n'.join(texts)))
                    message_param.content = contents
                if tool_calls:
                    message_param.tool_calls = tool_calls
                cohere_messages.append(message_param)
            else:
                assert_never(message)
        if instructions := self._get_instructions(messages, model_request_parameters):
            cohere_messages.insert(0, SystemChatMessageV2(role='system', content=instructions))
        return cohere_messages

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolV2]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> ToolCallV2:
        return ToolCallV2(
            id=_guard_tool_call_id(t=t),
            type='function',
            function=ToolCallV2Function(
                name=t.tool_name,
                arguments=t.args_as_json_str(),
            ),
        )

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ToolV2:
        return ToolV2(
            type='function',
            function=ToolV2Function(
                name=f.name,
                description=f.description,
                parameters=f.parameters_json_schema,
            ),
        )

    @classmethod
    def _map_user_message(cls, message: ModelRequest) -> Iterable[ChatMessageV2]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield SystemChatMessageV2(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                if isinstance(part.content, str):
                    yield UserChatMessageV2(role='user', content=part.content)
                else:
                    raise RuntimeError('Cohere does not yet support multi-modal inputs.')
            elif isinstance(part, ToolReturnPart):
                yield ToolChatMessageV2(
                    role='tool',
                    tool_call_id=_guard_tool_call_id(t=part),
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield UserChatMessageV2(role='user', content=part.model_response())  # pragma: no cover
                else:
                    yield ToolChatMessageV2(
                        role='tool',
                        tool_call_id=_guard_tool_call_id(t=part),
                        content=part.model_response(),
                    )
            else:
                assert_never(part)


def _map_usage(response: V2ChatResponse) -> usage.RequestUsage:
    u = response.usage
    if u is None:
        return usage.RequestUsage()
    else:
        details: dict[str, int] = {}
        if u.billed_units is not None:
            if u.billed_units.input_tokens:  # pragma: no branch
                details['input_tokens'] = int(u.billed_units.input_tokens)
            if u.billed_units.output_tokens:
                details['output_tokens'] = int(u.billed_units.output_tokens)
            if u.billed_units.search_units:  # pragma: no cover
                details['search_units'] = int(u.billed_units.search_units)
            if u.billed_units.classifications:  # pragma: no cover
                details['classifications'] = int(u.billed_units.classifications)

        request_tokens = int(u.tokens.input_tokens) if u.tokens and u.tokens.input_tokens else 0
        response_tokens = int(u.tokens.output_tokens) if u.tokens and u.tokens.output_tokens else 0
        return usage.RequestUsage(
            input_tokens=request_tokens,
            output_tokens=response_tokens,
            details=details,
        )
