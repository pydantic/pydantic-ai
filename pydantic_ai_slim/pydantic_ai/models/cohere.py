from __future__ import annotations as _annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import chain
from typing import Literal, Union, cast

from cohere import TextAssistantMessageContentItem
from httpx import AsyncClient as AsyncHTTPClient
from typing_extensions import assert_never

from .. import ModelHTTPError, result
from .._utils import guard_tool_call_id as _guard_tool_call_id
from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    Model,
    ModelRequestParameters,
    check_allow_model_requests,
)

try:
    from cohere import (
        AssistantChatMessageV2,
        AsyncClientV2,
        ChatMessageV2,
        ChatResponse,
        SystemChatMessageV2,
        ToolCallV2,
        ToolCallV2Function,
        ToolChatMessageV2,
        ToolV2,
        ToolV2Function,
        UserChatMessageV2,
    )
    from cohere.core.api_error import ApiError
    from cohere.v2.client import OMIT
except ImportError as _import_error:
    raise ImportError(
        'Please install `cohere` to use the Cohere model, '
        "you can use the `cohere` optional group — `pip install 'pydantic-ai-slim[cohere]'`"
    ) from _import_error

LatestCohereModelNames = Literal[
    'c4ai-aya-expanse-32b',
    'c4ai-aya-expanse-8b',
    'command',
    'command-light',
    'command-light-nightly',
    'command-nightly',
    'command-r',
    'command-r-03-2024',
    'command-r-08-2024',
    'command-r-plus',
    'command-r-plus-04-2024',
    'command-r-plus-08-2024',
    'command-r7b-12-2024',
]
"""Latest Cohere models."""

CohereModelName = Union[str, LatestCohereModelNames]
"""Possible Cohere model names.

Since Cohere supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [Cohere's docs](https://docs.cohere.com/v2/docs/models) for a list of all available models.
"""


class CohereModelSettings(ModelSettings):
    """Settings used for a Cohere model request."""

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
    _system: str | None = field(default='cohere', repr=False)

    def __init__(
        self,
        model_name: CohereModelName,
        *,
        api_key: str | None = None,
        cohere_client: AsyncClientV2 | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        """Initialize an Cohere model.

        Args:
            model_name: The name of the Cohere model to use. List of model names
                available [here](https://docs.cohere.com/docs/models#command).
            api_key: The API key to use for authentication, if not provided, the
                `CO_API_KEY` environment variable will be used if available.
            cohere_client: An existing Cohere async client to use. If provided,
                `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        self._model_name: CohereModelName = model_name
        if cohere_client is not None:
            assert http_client is None, 'Cannot provide both `cohere_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `cohere_client` and `api_key`'
            self.client = cohere_client
        else:
            self.client = AsyncClientV2(api_key=api_key, httpx_client=http_client)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, result.Usage]:
        check_allow_model_requests()
        response = await self._chat(messages, cast(CohereModelSettings, model_settings or {}), model_request_parameters)
        return self._process_response(response), _map_usage(response)

    @property
    def model_name(self) -> CohereModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str | None:
        """The system / model provider."""
        return self._system

    async def _chat(
        self,
        messages: list[ModelMessage],
        model_settings: CohereModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ChatResponse:
        tools = self._get_tools(model_request_parameters)
        cohere_messages = list(chain(*(self._map_message(m) for m in messages)))
        try:
            return await self.client.chat(
                model=self._model_name,
                messages=cohere_messages,
                tools=tools or OMIT,
                max_tokens=model_settings.get('max_tokens', OMIT),
                temperature=model_settings.get('temperature', OMIT),
                p=model_settings.get('top_p', OMIT),
                seed=model_settings.get('seed', OMIT),
                presence_penalty=model_settings.get('presence_penalty', OMIT),
                frequency_penalty=model_settings.get('frequency_penalty', OMIT),
            )
        except ApiError as e:
            if (status_code := e.status_code) and status_code >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise

    def _process_response(self, response: ChatResponse) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        parts: list[ModelResponsePart] = []
        if response.message.content is not None and len(response.message.content) > 0:
            # While Cohere's API returns a list, it only does that for future proofing
            # and currently only one item is being returned.
            choice = response.message.content[0]
            parts.append(TextPart(choice.text))
        for c in response.message.tool_calls or []:
            if c.function and c.function.name and c.function.arguments:
                parts.append(
                    ToolCallPart(
                        tool_name=c.function.name,
                        args=c.function.arguments,
                        tool_call_id=c.id,
                    )
                )
        return ModelResponse(parts=parts, model_name=self._model_name)

    def _map_message(self, message: ModelMessage) -> Iterable[ChatMessageV2]:
        """Just maps a `pydantic_ai.Message` to a `cohere.ChatMessageV2`."""
        if isinstance(message, ModelRequest):
            yield from self._map_user_message(message)
        elif isinstance(message, ModelResponse):
            texts: list[str] = []
            tool_calls: list[ToolCallV2] = []
            for item in message.parts:
                if isinstance(item, TextPart):
                    texts.append(item.content)
                elif isinstance(item, ToolCallPart):
                    tool_calls.append(self._map_tool_call(item))
                else:
                    assert_never(item)
            message_param = AssistantChatMessageV2(role='assistant')
            if texts:
                message_param.content = [TextAssistantMessageContentItem(text='\n\n'.join(texts))]
            if tool_calls:
                message_param.tool_calls = tool_calls
            yield message_param
        else:
            assert_never(message)

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolV2]:
        tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
        if model_request_parameters.result_tools:
            tools += [self._map_tool_definition(r) for r in model_request_parameters.result_tools]
        return tools

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> ToolCallV2:
        return ToolCallV2(
            id=_guard_tool_call_id(t=t, model_source='Cohere'),
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
                    tool_call_id=_guard_tool_call_id(t=part, model_source='Cohere'),
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield UserChatMessageV2(role='user', content=part.model_response())
                else:
                    yield ToolChatMessageV2(
                        role='tool',
                        tool_call_id=_guard_tool_call_id(t=part, model_source='Cohere'),
                        content=part.model_response(),
                    )
            else:
                assert_never(part)


def _map_usage(response: ChatResponse) -> result.Usage:
    usage = response.usage
    if usage is None:
        return result.Usage()
    else:
        details: dict[str, int] = {}
        if usage.billed_units is not None:
            if usage.billed_units.input_tokens:
                details['input_tokens'] = int(usage.billed_units.input_tokens)
            if usage.billed_units.output_tokens:
                details['output_tokens'] = int(usage.billed_units.output_tokens)
            if usage.billed_units.search_units:
                details['search_units'] = int(usage.billed_units.search_units)
            if usage.billed_units.classifications:
                details['classifications'] = int(usage.billed_units.classifications)

        request_tokens = int(usage.tokens.input_tokens) if usage.tokens and usage.tokens.input_tokens else None
        response_tokens = int(usage.tokens.output_tokens) if usage.tokens and usage.tokens.output_tokens else None
        return result.Usage(
            request_tokens=request_tokens,
            response_tokens=response_tokens,
            total_tokens=(request_tokens or 0) + (response_tokens or 0),
            details=details,
        )
