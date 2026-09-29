"""`BedrockConverseModel` with request assembly, response parsing and streaming done by babel's `bedrock-converse` codec."""

from __future__ import annotations as _annotations

import dataclasses
from collections.abc import AsyncIterator, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

from llm_transform.capabilities import Capabilities, capabilities_for
from llm_transform.media import MEDIA_URL_OK
from llm_transform.registry import decode_response, encode, stream_step

from ...messages import InstructionPart, ModelMessage, ModelResponse, ModelResponseStreamEvent
from ...profiles import ModelProfile, merge_profile
from ...providers.bedrock import BedrockModelProfile
from .. import ModelRequestParameters
from ..bedrock import (
    _FINISH_REASON_MAP,  # pyright: ignore[reportPrivateUsage]
    BedrockConverseModel,
    BedrockModelSettings,
    BedrockStreamedResponse,
    _AsyncIteratorWrapper,  # pyright: ignore[reportPrivateUsage]
    _insert_cache_point_before_trailing_documents,  # pyright: ignore[reportPrivateUsage]
    _map_api_errors,  # pyright: ignore[reportPrivateUsage]
    _map_usage,  # pyright: ignore[reportPrivateUsage]
)
from ._adapters import (
    download_url_media,
    fold_stream_emits,
    ir_to_model_response,
    messages_to_ir,
    reconcile_ir,
    uploaded_files,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        MessageUnionTypeDef,
        SystemContentBlockTypeDef,
    )

__all__ = ('BabelBedrockConverseModel', 'BabelBedrockStreamedResponse')

# Converse takes S3 objects by reference (`s3Location`), which babel encodes from an `s3://` URL, so
# such a URL is never downloaded.
_PASSTHROUGH_SCHEMES = frozenset({'s3'})


class BabelBedrockStreamedResponse(BedrockStreamedResponse):
    """`BedrockStreamedResponse` whose events are folded by babel's `bedrock-converse` `stream_step`."""

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        with _map_api_errors(self._model_name, self._model_id_namespace):
            if self._provider_response_id is not None:
                self.provider_response_id = self._provider_response_id
            ignore_leading_whitespace = self._model_profile.get('ignore_streamed_leading_whitespace', False)
            state: Any = {}
            async for chunk in _AsyncIteratorWrapper(self._event_stream):
                match chunk:
                    case {'messageStop': message_stop}:
                        raw_finish_reason = message_stop['stopReason']
                        self.provider_details = {**(self.provider_details or {}), 'finish_reason': raw_finish_reason}
                        self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)
                    case {'metadata': metadata}:
                        if 'usage' in metadata:
                            self._usage += _map_usage(
                                metadata['usage'], self._provider_name, self._provider_url, self._model_name
                            )
                        if 'trace' in metadata:
                            self.provider_details = {**(self.provider_details or {}), 'trace': metadata['trace']}
                    case _:
                        pass
                result = stream_step('bedrock-converse', state, chunk)
                state = result['state']
                for event in fold_stream_emits(
                    result['emit'],
                    self._parts_manager,
                    self,
                    provider_name=self._provider_name,
                    ignore_leading_whitespace=ignore_leading_whitespace,
                ):
                    yield event


class BabelBedrockConverseModel(BedrockConverseModel):
    """[`BedrockConverseModel`][pydantic_ai.models.bedrock.BedrockConverseModel] mapped by babel's `bedrock-converse` codec.

    Construct it exactly like `BedrockConverseModel`. The provider, boto3 client, settings and
    guardrail configuration behave as they do there; only the translation between the message
    history and the Converse wire is babel's. The model family facts the profile states are handed
    to babel as capability data and applied the way the native model applies them: the block a tool
    result takes (`bedrock_tool_result_format`), whether it carries a `status`
    (`bedrock_supports_tool_result_status`), whether a conversation may open with an assistant turn
    (`bedrock_supports_leading_assistant_message`, otherwise a `.` user turn is prepended) and
    whether thinking parts are replayed (`bedrock_send_back_thinking_parts`). The
    `bedrock_cache_instructions` and `bedrock_cache_messages` breakpoints are placed as the native
    model places them when the profile supports prompt caching.

    Converse takes no media by URL except S3 objects, so every other `FileUrl` is downloaded and
    inlined as bytes. The raw stop reason and a guardrail trace are recorded in `provider_details`
    as the native model records them.
    """

    @cached_property
    def profile(self) -> BedrockModelProfile:
        # babel carries one request-level system prompt, so a mid-conversation `SystemPromptPart` is
        # delivered the way `Model.prepare_messages` delivers it to any wire with no inline system
        # role: as `<system>`-tagged user text, in place.
        return cast(
            BedrockModelProfile, merge_profile(super().profile, ModelProfile(supports_inline_system_prompts=False))
        )

    @property
    def _streamed_response_cls(self) -> type[BedrockStreamedResponse]:
        return BabelBedrockStreamedResponse

    async def _map_messages(
        self,
        messages: Sequence[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        model_settings: BedrockModelSettings | None,
    ) -> tuple[list[SystemContentBlockTypeDef], list[MessageUnionTypeDef]]:
        for file in uploaded_files(messages):
            self._validate_uploaded_file_provider(file)
        messages = await download_url_media(
            messages, MEDIA_URL_OK['bedrock-converse'], passthrough_schemes=_PASSTHROUGH_SCHEMES
        )
        instruction_parts = self._get_instruction_parts(messages, model_request_parameters) or []
        ir = messages_to_ir(
            messages,
            model_name=self.model_name,
            provider_name=self._provider.name,
            instruction_parts=instruction_parts,
        )
        ir = reconcile_ir(ir, _capabilities(self.profile))
        encoded = encode('bedrock-converse', ir)
        system = cast(list['SystemContentBlockTypeDef'], encoded.get('system') or [])
        bedrock_messages = cast(list['MessageUnionTypeDef'], encoded['messages'])
        if self.profile.get('bedrock_supports_prompt_caching', False):
            self._add_cache_points(system, bedrock_messages, instruction_parts, model_settings or {})
        return system, bedrock_messages

    def _add_cache_points(
        self,
        system: list[SystemContentBlockTypeDef],
        messages: list[MessageUnionTypeDef],
        instruction_parts: Sequence[InstructionPart],
        settings: BedrockModelSettings,
    ) -> None:
        """Place the `bedrock_cache_instructions` and `bedrock_cache_messages` breakpoints.

        As in the native model: the instructions breakpoint follows the last static instruction, or
        the whole system prompt when every instruction is static, and the messages breakpoint ends
        the last user message, ahead of any trailing documents Converse will not cache after.
        """
        if system and (cache_instructions := settings.get('bedrock_cache_instructions')):
            cache_point = cast('SystemContentBlockTypeDef', self._get_cache_point(cache_instructions))
            static_count = sum(1 for part in instruction_parts if not part.dynamic)
            if static_count < len(instruction_parts):
                index = len(system) - len(instruction_parts) + static_count
                if index > 0:
                    system.insert(index, cache_point)
            else:
                system.append(cache_point)
        if messages and (cache_messages := settings.get('bedrock_cache_messages')):
            last_user_content = self._get_last_user_message_content(messages)
            if last_user_content is not None:
                _insert_cache_point_before_trailing_documents(last_user_content, self._get_cache_point(cache_messages))

    async def _process_response(self, response: ConverseResponseTypeDef) -> ModelResponse:
        raw_finish_reason = response['stopReason']
        details: dict[str, Any] = {'finish_reason': raw_finish_reason}
        if 'trace' in response:
            details['trace'] = response['trace']
        return ir_to_model_response(
            decode_response('bedrock-converse', cast(dict[str, Any], response)),
            fmt='bedrock-converse',
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            usage=_map_usage(response['usage'], self._provider.name, self._provider.base_url, self.model_name),
            model_name=self.model_name,
            provider_response_id=response.get('ResponseMetadata', {}).get('RequestId'),
            provider_details=details,
            finish_reason=_FINISH_REASON_MAP.get(raw_finish_reason),
        )


def _capabilities(profile: BedrockModelProfile) -> Capabilities:
    """The Converse facts babel ships, with the family facts the profile states.

    The profile is what a user overrides in Pydantic AI, so it is the source of these facts here
    rather than babel's own family table; both were read from the same provider behaviour.
    """
    return dataclasses.replace(
        capabilities_for('bedrock-converse'),
        reasoning=profile.get('bedrock_send_back_thinking_parts', False),
        tool_result_block=profile.get('bedrock_tool_result_format', 'text'),
        tool_result_status=profile.get('bedrock_supports_tool_result_status', True),
        conversation_may_start_with_assistant=profile.get('bedrock_supports_leading_assistant_message', False),
    )
