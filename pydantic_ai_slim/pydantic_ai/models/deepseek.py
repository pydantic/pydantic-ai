from __future__ import annotations as _annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Literal, cast

from pydantic import BaseModel, JsonValue, ValidationError
from typing_extensions import override

from .. import usage
from .._utils import is_str_dict
from ..messages import ModelResponse, ModelResponseStreamEvent, NativeToolCallPart, NativeToolReturnPart
from ..native_tools import AbstractNativeTool, WebFetchTool, WebSearchTool
from ..profiles import ModelProfileSpec, merge_profile
from ..providers import Provider
from ..providers.deepseek import DeepSeekModelName, DeepSeekModelProfile
from ..settings import ModelSettings
from . import ModelRequestParameters

try:
    from openai import AsyncOpenAI
    from openai.types import responses

    from .openai import OpenAIResponsesModel, OpenAIResponsesStreamedResponse
    from .openai._response_fields import map_usage
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the DeepSeek models, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = ('DeepSeekResponsesModel', 'DeepSeekResponsesStreamedResponse')

_WEB_TOOL_KEY = 'deepseek_web_tool'
"""`NativeToolReturnPart.provider_details` key holding the `kind` and `signature` a web tool call is sent back with."""


class _WebToolItem(BaseModel):
    """A `web_search_call` item from one of DeepSeek's web tools, as received and as sent back."""

    type: Literal['web_search_call']
    id: str
    kind: str
    status: Literal['in_progress', 'searching', 'completed', 'failed']
    action: dict[str, JsonValue] | None = None
    page: dict[str, JsonValue] | None = None
    signature: str | None = None
    error: dict[str, JsonValue] | None = None


def _web_tool_parts(
    item: responses.ResponseFunctionWebSearch, provider_name: str
) -> tuple[NativeToolCallPart, NativeToolReturnPart] | None:
    """The parts for a `web_search_call` item from one of DeepSeek's web tools, or `None` for a standard item."""
    wire = item.model_dump(mode='json', exclude_unset=True, warnings=False)
    if 'kind' not in wire:
        return None
    parsed = _WebToolItem.model_validate(wire)
    name = WebFetchTool.kind if parsed.kind == 'read_link' else WebSearchTool.kind
    action = parsed.action or {}
    content = {
        'status': parsed.status,
        'sources': action.get('sources'),
        'page': parsed.page,
        'error': parsed.error,
    }
    return (
        NativeToolCallPart(
            tool_name=name,
            args={key: value for key, value in action.items() if key != 'sources'},
            tool_call_id=parsed.id,
            id=parsed.id,
            provider_name=provider_name,
        ),
        NativeToolReturnPart(
            tool_name=name,
            content={key: value for key, value in content.items() if value is not None},
            tool_call_id=parsed.id,
            outcome='failed' if parsed.status == 'failed' else 'success',
            provider_name=provider_name,
            provider_details={_WEB_TOOL_KEY: parsed.model_dump(include={'kind', 'signature'}, exclude_none=True)},
        ),
    )


def _map_usage(
    response: responses.Response, provider_name: str, provider_url: str, model_name: str
) -> usage.RequestUsage:
    """Map a response's usage, with the non-zero web tool call counts of `usage.server_tool_use` in its details.

    DeepSeek bills none of these calls, only the tokens their results add to the context:
    https://api-docs.deepseek.com/quick_start/pricing/.
    """
    request_usage = map_usage(response, provider_name, provider_url, model_name, bills_web_searches=False)
    counts = (response.usage.model_extra or {}).get('server_tool_use') if response.usage else None
    if is_str_dict(counts):
        request_usage.details.update(
            {
                key: count
                for key in ('tool_search', 'tool_fetch', 'tool_image_search')
                if isinstance(count := counts.get(key), int) and not isinstance(count, bool) and count > 0
            }
        )
    return request_usage


class DeepSeekResponsesModel(OpenAIResponsesModel):
    """A model that uses DeepSeek's Responses API.

    With a model whose profile sets
    [`deepseek_supports_web_tools`][pydantic_ai.providers.deepseek.DeepSeekModelProfile.deepseek_supports_web_tools],
    as `deepseek-flash`'s does, [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool] and
    [`WebFetchTool`][pydantic_ai.native_tools.WebFetchTool] run as DeepSeek's server-side `deepseek_web_search` and
    `deepseek_read_url` tools. Their signed results are kept in the message history and sent back on later requests.
    With any other model, it behaves like [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel].

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    def __init__(
        self,
        model_name: DeepSeekModelName | str,
        *,
        provider: Literal['deepseek'] | Provider[AsyncOpenAI] = 'deepseek',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a DeepSeek Responses model.

        Args:
            model_name: The name of the DeepSeek model to use.
            provider: The provider to use. Defaults to `'deepseek'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)

    @classmethod
    @override
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        return super().supported_native_tools() | {WebFetchTool}

    @cached_property
    @override
    def profile(self) -> DeepSeekModelProfile:
        profile = cast(DeepSeekModelProfile, super().profile)
        if profile.get('deepseek_supports_web_tools', False):
            # DeepSeek's `url_citation` offsets span the cited text: its content holds no citation markers.
            update = DeepSeekModelProfile(openai_responses_url_citation_content_offsets=True)
        else:
            update = DeepSeekModelProfile(
                supported_native_tools=profile.get('supported_native_tools', frozenset()) - {WebFetchTool}
            )
        return cast(DeepSeekModelProfile, merge_profile(profile, update))

    @override
    def unsupported_native_tool_settings(self, native_tool: AbstractNativeTool) -> frozenset[str]:
        """Return `native_tool`'s restricting settings that the model can't apply.

        DeepSeek's web tools take none of them. Otherwise, as for `OpenAIResponsesModel`.
        """
        if self.profile.get('deepseek_supports_web_tools', False) and isinstance(
            native_tool, WebSearchTool | WebFetchTool
        ):
            return native_tool.restricting_settings
        return super().unsupported_native_tool_settings(native_tool)

    @override
    def _get_native_tools(self, model_request_parameters: ModelRequestParameters) -> list[responses.ToolParam]:
        if not self.profile.get('deepseek_supports_web_tools', False):
            return super()._get_native_tools(model_request_parameters)
        native_tools = model_request_parameters.native_tools
        other_tools = [tool for tool in native_tools if not isinstance(tool, WebSearchTool | WebFetchTool)]
        # The SDK's tool union doesn't declare DeepSeek's server-side tools.
        return [
            *(
                cast(
                    responses.ToolParam,
                    {'type': 'deepseek_read_url' if isinstance(tool, WebFetchTool) else 'deepseek_web_search'},
                )
                for tool in native_tools
                if isinstance(tool, WebSearchTool | WebFetchTool)
            ),
            *super()._get_native_tools(replace(model_request_parameters, native_tools=other_tools)),
        ]

    @override
    def _map_web_search_call(
        self, item: responses.ResponseFunctionWebSearch
    ) -> tuple[NativeToolCallPart, NativeToolReturnPart]:
        if self.profile.get('deepseek_supports_web_tools', False) and (parts := _web_tool_parts(item, self.system)):
            return parts
        return super()._map_web_search_call(item)

    @override
    def _map_usage(self, response: responses.Response) -> usage.RequestUsage:
        if not self.profile.get('deepseek_supports_web_tools', False):
            return super()._map_usage(response)
        return _map_usage(response, self._provider.name, self._provider.base_url, self.model_name)

    @property
    @override
    def _streamed_response_cls(self) -> type[OpenAIResponsesStreamedResponse]:
        if self.profile.get('deepseek_supports_web_tools', False):
            return DeepSeekResponsesStreamedResponse
        return super()._streamed_response_cls

    @override
    def _map_native_tool_call_replay(
        self, item: NativeToolCallPart, message: ModelResponse
    ) -> list[responses.ResponseInputItemParam] | None:
        result = next(
            (
                part
                for part in message.parts
                if isinstance(part, NativeToolReturnPart)
                and (part.tool_name, part.tool_call_id) == (item.tool_name, item.tool_call_id)
                and _WEB_TOOL_KEY in (part.provider_details or {})
            ),
            None,
        )
        if (
            not self.profile.get('deepseek_supports_web_tools', False)
            or result is None
            or not self.accepts_stateless_artifacts(message, item)
        ):
            return super()._map_native_tool_call_replay(item, message)
        metadata = (result.provider_details or {}).get(_WEB_TOOL_KEY)
        content = result.content
        if not is_str_dict(metadata) or not is_str_dict(content):
            return []
        action = item.args_as_dict()
        if 'sources' in content:
            action = {**action, 'sources': content['sources']}
        # History may come from storage or a UI client, so only the web tool's own fields are restored.
        try:
            replay = _WebToolItem.model_validate(
                {
                    'type': 'web_search_call',
                    'id': item.tool_call_id,
                    'action': action,
                    'status': content.get('status'),
                    'page': content.get('page'),
                    'error': content.get('error'),
                    'kind': metadata.get('kind'),
                    'signature': metadata.get('signature'),
                }
            )
        except ValidationError:
            return []
        # The SDK's input item union doesn't declare DeepSeek's extra fields.
        return [cast(responses.ResponseInputItemParam, replay.model_dump(mode='json', exclude_none=True))]


@dataclass
class DeepSeekResponsesStreamedResponse(OpenAIResponsesStreamedResponse):
    """Implementation of `StreamedResponse` for DeepSeek's Responses API with its web tools."""

    @override
    def _map_web_search_call_added(self, item: responses.ResponseFunctionWebSearch) -> ModelResponseStreamEvent:
        if (parts := _web_tool_parts(item, self.provider_name)) is None:
            return super()._map_web_search_call_added(item)
        return self._parts_manager.handle_part(vendor_part_id=f'{item.id}-call', part=replace(parts[0], args=None))

    @override
    def _map_web_search_call_done(
        self, item: responses.ResponseFunctionWebSearch
    ) -> Iterator[ModelResponseStreamEvent]:
        if (parts := _web_tool_parts(item, self.provider_name)) is None:
            yield from super()._map_web_search_call_done(item)
            return
        call_part, return_part = parts
        if self._parts_manager.get_part_by_vendor_id(f'{item.id}-call') is None:
            yield self._parts_manager.handle_part(vendor_part_id=f'{item.id}-call', part=call_part)
        else:
            yield from self._parts_manager.handle_tool_call_delta_events(
                vendor_part_id=f'{item.id}-call', args=call_part.args
            )
        yield self._parts_manager.handle_part(vendor_part_id=f'{item.id}-return', part=return_part)

    @override
    def _map_usage(self, response: responses.Response) -> usage.RequestUsage:
        return _map_usage(response, self._provider_name, self._provider_url, self.model_name)
