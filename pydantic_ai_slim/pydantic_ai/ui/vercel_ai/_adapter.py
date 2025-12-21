"""Vercel AI adapter for handling requests."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

from pydantic import TypeAdapter
from typing_extensions import assert_never

from ...messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    VideoUrl,
)
from ...output import OutputDataT
from ...tools import AgentDepsT
from .. import MessagesBuilder, UIAdapter, UIEventStream
from ._event_stream import VercelAIEventStream
from ._utils import dump_provider_metadata, load_provider_metadata
from .request_types import (
    DataUIPart,
    DynamicToolInputAvailablePart,
    DynamicToolInputStreamingPart,
    DynamicToolOutputAvailablePart,
    DynamicToolOutputErrorPart,
    DynamicToolUIPart,
    FileUIPart,
    ProviderMetadata,
    ReasoningUIPart,
    RequestData,
    SourceDocumentUIPart,
    SourceUrlUIPart,
    StepStartUIPart,
    TextUIPart,
    ToolInputAvailablePart,
    ToolInputStreamingPart,
    ToolOutputAvailablePart,
    ToolOutputErrorPart,
    ToolUIPart,
    UIMessage,
    UIMessagePart,
)
from .response_types import BaseChunk

if TYPE_CHECKING:
    pass


__all__ = ['VercelAIAdapter']

request_data_ta: TypeAdapter[RequestData] = TypeAdapter(RequestData)


@dataclass
class VercelAIAdapter(UIAdapter[RequestData, UIMessage, BaseChunk, AgentDepsT, OutputDataT]):
    """UI adapter for the Vercel AI protocol."""

    @classmethod
    def build_run_input(cls, body: bytes) -> RequestData:
        """Build a Vercel AI run input object from the request body."""
        return request_data_ta.validate_json(body)

    def build_event_stream(self) -> UIEventStream[RequestData, BaseChunk, AgentDepsT, OutputDataT]:
        """Build a Vercel AI event stream transformer."""
        return VercelAIEventStream(self.run_input, accept=self.accept)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the Vercel AI run input."""
        return self.load_messages(self.run_input.messages)

    @classmethod
    def load_messages(cls, messages: Sequence[UIMessage]) -> list[ModelMessage]:  # noqa: C901
        """Transform Vercel AI messages into Pydantic AI messages."""
        builder = MessagesBuilder()

        for msg in messages:
            if msg.role == 'system':
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        builder.add(SystemPromptPart(content=part.text))
                    else:  # pragma: no cover
                        raise ValueError(f'Unsupported system message part type: {type(part)}')
            elif msg.role == 'user':
                user_prompt_content: str | list[UserContent] = []
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        user_prompt_content.append(part.text)
                    elif isinstance(part, FileUIPart):
                        try:
                            file = BinaryContent.from_data_uri(part.url)
                        except ValueError:
                            media_type_prefix = part.media_type.split('/', 1)[0]
                            match media_type_prefix:
                                case 'image':
                                    file = ImageUrl(url=part.url, media_type=part.media_type)
                                case 'video':
                                    file = VideoUrl(url=part.url, media_type=part.media_type)
                                case 'audio':
                                    file = AudioUrl(url=part.url, media_type=part.media_type)
                                case _:
                                    file = DocumentUrl(url=part.url, media_type=part.media_type)
                        user_prompt_content.append(file)
                    else:  # pragma: no cover
                        raise ValueError(f'Unsupported user message part type: {type(part)}')

                if user_prompt_content:  # pragma: no branch
                    if len(user_prompt_content) == 1 and isinstance(user_prompt_content[0], str):
                        user_prompt_content = user_prompt_content[0]
                    builder.add(UserPromptPart(content=user_prompt_content))

            elif msg.role == 'assistant':
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        part_metadata = load_provider_metadata(part.provider_metadata)
                        builder.add(
                            TextPart(
                                content=part.text,
                                id=part_metadata.get('id'),
                                provider_details=part_metadata.get('provider_details'),
                            )
                        )
                    elif isinstance(part, ReasoningUIPart):
                        part_metadata = load_provider_metadata(part.provider_metadata)
                        builder.add(
                            ThinkingPart(
                                content=part.text,
                                id=part_metadata.get('id'),
                                signature=part_metadata.get('signature'),
                                provider_name=part_metadata.get('provider_name'),
                                provider_details=part_metadata.get('provider_details'),
                            )
                        )
                    elif isinstance(part, FileUIPart):
                        try:
                            file = BinaryContent.from_data_uri(part.url)
                        except ValueError as e:  # pragma: no cover
                            # We don't yet handle non-data-URI file URLs returned by assistants, as no Pydantic AI models do this.
                            raise ValueError(
                                'Vercel AI integration can currently only handle assistant file parts with data URIs.'
                            ) from e
                        part_metadata = load_provider_metadata(part.provider_metadata)
                        builder.add(
                            FilePart(
                                content=file,
                                id=part_metadata.get('id'),
                                provider_name=part_metadata.get('provider_name'),
                                provider_details=part_metadata.get('provider_details'),
                            )
                        )
                    elif isinstance(part, ToolUIPart | DynamicToolUIPart):
                        if isinstance(part, DynamicToolUIPart):
                            tool_name = part.tool_name
                            builtin_tool = False
                        else:
                            tool_name = part.type.removeprefix('tool-')
                            builtin_tool = part.provider_executed

                        tool_call_id = part.tool_call_id

                        args: str | dict[str, Any] | None = part.input

                        if isinstance(args, str):
                            try:
                                parsed = json.loads(args)
                                if isinstance(parsed, dict):
                                    args = cast(dict[str, Any], parsed)
                            except json.JSONDecodeError:
                                pass
                        elif isinstance(args, dict) or args is None:
                            pass
                        else:
                            assert_never(args)

                        part_metadata = {}
                        part_id = provider_name = provider_details = None
                        if not isinstance(part, (DynamicToolInputStreamingPart, ToolInputStreamingPart)):
                            part_metadata = load_provider_metadata(part.call_provider_metadata)
                            part_id = part_metadata.get('id')
                            provider_name = part_metadata.get('provider_name')
                            provider_details = part_metadata.get('provider_details')

                        if builtin_tool:
                            # For builtin tools, the call and return metadata are combined in the ToolOutputAvailablePart.
                            # We extract and return them to the respective parts for pydantic_ai.
                            call_meta = return_meta = {}
                            if isinstance(part, ToolOutputAvailablePart):
                                call_meta, return_meta = cls._load_builtin_tool_meta(part_metadata)
                                call_meta = call_meta or {}
                                return_meta = return_meta or {}

                            call_id = call_meta.get('id') or part_id
                            call_provider_name = call_meta.get('provider_name') or provider_name
                            call_provider_details = call_meta.get('provider_details') or provider_details

                            return_provider_name = return_meta.get('provider_name') or provider_name
                            return_provider_details = return_meta.get('provider_details') or provider_details

                            call_part = BuiltinToolCallPart(
                                tool_name=tool_name,
                                tool_call_id=tool_call_id,
                                args=args,
                                id=call_id,
                                provider_name=call_provider_name,
                                provider_details=call_provider_details,
                            )
                            builder.add(call_part)

                            if isinstance(part, ToolOutputAvailablePart | ToolOutputErrorPart):
                                if part.state == 'output-available':
                                    output = part.output
                                else:
                                    output = {'error_text': part.error_text, 'is_error': True}

                                builder.add(
                                    BuiltinToolReturnPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        content=output,
                                        provider_name=return_provider_name,
                                        provider_details=return_provider_details,
                                    )
                                )
                        else:
                            builder.add(
                                ToolCallPart(
                                    tool_name=tool_name,
                                    tool_call_id=tool_call_id,
                                    args=args,
                                    id=part_id,
                                    provider_details=provider_details,
                                )
                            )

                            if part.state == 'output-available':
                                builder.add(
                                    ToolReturnPart(tool_name=tool_name, tool_call_id=tool_call_id, content=part.output)
                                )
                            elif part.state == 'output-error':
                                builder.add(
                                    RetryPromptPart(
                                        tool_name=tool_name, tool_call_id=tool_call_id, content=part.error_text
                                    )
                                )
                    elif isinstance(part, DataUIPart):  # pragma: no cover
                        # Contains custom data that shouldn't be sent to the model
                        pass
                    elif isinstance(part, SourceUrlUIPart):  # pragma: no cover
                        # TODO: Once we support citations: https://github.com/pydantic/pydantic-ai/issues/3126
                        pass
                    elif isinstance(part, SourceDocumentUIPart):  # pragma: no cover
                        # TODO: Once we support citations: https://github.com/pydantic/pydantic-ai/issues/3126
                        pass
                    elif isinstance(part, StepStartUIPart):  # pragma: no cover
                        # Nothing to do here
                        pass
                    else:
                        assert_never(part)
            else:
                assert_never(msg.role)

        return builder.messages

    @staticmethod
    def _dump_builtin_tool_meta(
        call_provider_metadata: ProviderMetadata | None, return_provider_metadata: ProviderMetadata | None
    ) -> ProviderMetadata | None:
        """Builtin tool call and return provider metadata use special keys (call_meta and return_meta)."""
        return dump_provider_metadata(call_meta=call_provider_metadata, return_meta=return_provider_metadata)

    @staticmethod
    def _load_builtin_tool_meta(
        provider_metadata: ProviderMetadata,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Builtin tool call and return provider metadata use special keys (call_meta and return_meta)."""
        return provider_metadata.get('call_meta'), provider_metadata.get('return_meta')

    @staticmethod
    def _dump_request_message(msg: ModelRequest) -> tuple[list[UIMessagePart], list[UIMessagePart]]:
        """Convert a ModelRequest into a UIMessage."""
        system_ui_parts: list[UIMessagePart] = []
        user_ui_parts: list[UIMessagePart] = []

        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                system_ui_parts.append(TextUIPart(text=part.content, state='done'))
            elif isinstance(part, UserPromptPart):
                user_ui_parts.extend(_convert_user_prompt_part(part))
            elif isinstance(part, ToolReturnPart):
                # Tool returns are merged into the tool call in the assistant message
                pass
            elif isinstance(part, RetryPromptPart):
                if part.tool_name:
                    # Tool-related retries are handled when processing ToolCallPart in ModelResponse
                    pass
                else:
                    # Non-tool retries (e.g., output validation errors) become user text
                    user_ui_parts.append(TextUIPart(text=part.model_response(), state='done'))
            else:
                assert_never(part)

        return system_ui_parts, user_ui_parts

    @classmethod
    def _dump_response_message(
        cls, msg: ModelResponse, tool_results: dict[str, ToolReturnPart | RetryPromptPart]
    ) -> list[UIMessagePart]:
        """Convert a ModelResponse into a UIMessage."""
        ui_parts: list[UIMessagePart] = []

        # For builtin tools, returns can be in the same ModelResponse as calls
        local_builtin_returns: dict[str, BuiltinToolReturnPart] = {
            part.tool_call_id: part for part in msg.parts if isinstance(part, BuiltinToolReturnPart)
        }

        for part in msg.parts:
            if isinstance(part, BuiltinToolReturnPart):
                continue
            elif isinstance(part, TextPart):
                # Combine consecutive text parts
                if ui_parts and isinstance(ui_parts[-1], TextUIPart):
                    ui_parts[-1].text += part.content
                else:
                    provider_metadata = dump_provider_metadata(id=part.id, provider_details=part.provider_details)
                    ui_parts.append(TextUIPart(text=part.content, state='done', provider_metadata=provider_metadata))
            elif isinstance(part, ThinkingPart):
                provider_metadata = dump_provider_metadata(
                    id=part.id,
                    signature=part.signature,
                    provider_name=part.provider_name,
                    provider_details=part.provider_details,
                )
                ui_parts.append(ReasoningUIPart(text=part.content, state='done', provider_metadata=provider_metadata))
            elif isinstance(part, FilePart):
                ui_parts.append(
                    FileUIPart(
                        url=part.content.data_uri,
                        media_type=part.content.media_type,
                        provider_metadata=dump_provider_metadata(
                            id=part.id, provider_name=part.provider_name, provider_details=part.provider_details
                        ),
                    )
                )
            elif isinstance(part, BuiltinToolCallPart):
                call_provider_metadata = dump_provider_metadata(
                    id=part.id, provider_name=part.provider_name, provider_details=part.provider_details
                )

                if builtin_return := local_builtin_returns.get(part.tool_call_id):
                    # Builtin tool calls in pydantic_ai are represented by two parts:
                    #   1. BuiltinToolCallPart (the tool request) -> part
                    #   2. BuiltinToolReturnPart (the tool's output) -> builtin_return
                    # The Vercel AI SDK only has a single ToolOutputAvailablePart.
                    # So, we combine all relevant metadata here so that, when we later convert back from Vercel AI to pydantic_ai,
                    # we can properly reconstruct both the call and return parts with their original metadata.
                    # Note: This extra metadata handling is only needed for built-in tools, since normal tool returns
                    # (ToolReturnPart) do not include provider metadata.

                    call_meta = dump_provider_metadata(
                        wrapper_key=None,
                        id=part.id,
                        provider_name=part.provider_name,
                        provider_details=part.provider_details,
                    )
                    return_meta = dump_provider_metadata(
                        wrapper_key=None,
                        provider_name=builtin_return.provider_name,
                        provider_details=builtin_return.provider_details,
                    )
                    combined_provider_meta = cls._dump_builtin_tool_meta(call_meta, return_meta)

                    content = builtin_return.model_response_str()
                    ui_parts.append(
                        ToolOutputAvailablePart(
                            type=f'tool-{part.tool_name}',
                            tool_call_id=part.tool_call_id,
                            input=part.args_as_json_str(),
                            output=content,
                            state='output-available',
                            provider_executed=True,
                            call_provider_metadata=combined_provider_meta,
                        )
                    )
                else:
                    ui_parts.append(
                        ToolInputAvailablePart(
                            type=f'tool-{part.tool_name}',
                            tool_call_id=part.tool_call_id,
                            input=part.args_as_json_str(),
                            state='input-available',
                            provider_executed=True,
                            call_provider_metadata=call_provider_metadata,
                        )
                    )
            elif isinstance(part, ToolCallPart):
                tool_result = tool_results.get(part.tool_call_id)
                call_provider_metadata = dump_provider_metadata(id=part.id, provider_details=part.provider_details)

                if isinstance(tool_result, ToolReturnPart):
                    content = tool_result.model_response_str()
                    ui_parts.append(
                        DynamicToolOutputAvailablePart(
                            tool_name=part.tool_name,
                            tool_call_id=part.tool_call_id,
                            input=part.args_as_json_str(),
                            output=content,
                            state='output-available',
                            call_provider_metadata=call_provider_metadata,
                        )
                    )
                elif isinstance(tool_result, RetryPromptPart):
                    error_text = tool_result.model_response()
                    ui_parts.append(
                        DynamicToolOutputErrorPart(
                            tool_name=part.tool_name,
                            tool_call_id=part.tool_call_id,
                            input=part.args_as_json_str(),
                            error_text=error_text,
                            state='output-error',
                            call_provider_metadata=call_provider_metadata,
                        )
                    )
                else:
                    ui_parts.append(
                        DynamicToolInputAvailablePart(
                            tool_name=part.tool_name,
                            tool_call_id=part.tool_call_id,
                            input=part.args_as_json_str(),
                            state='input-available',
                            call_provider_metadata=call_provider_metadata,
                        )
                    )
            else:
                assert_never(part)

        return ui_parts

    @classmethod
    def dump_messages(
        cls,
        messages: Sequence[ModelMessage],
    ) -> list[UIMessage]:
        """Transform Pydantic AI messages into Vercel AI messages.

        Args:
            messages: A sequence of ModelMessage objects to convert

        Returns:
            A list of UIMessage objects in Vercel AI format
        """
        tool_results: dict[str, ToolReturnPart | RetryPromptPart] = {}

        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        tool_results[part.tool_call_id] = part
                    elif isinstance(part, RetryPromptPart) and part.tool_name:
                        tool_results[part.tool_call_id] = part

        result: list[UIMessage] = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                system_ui_parts, user_ui_parts = cls._dump_request_message(msg)
                if system_ui_parts:
                    result.append(UIMessage(id=str(uuid.uuid4()), role='system', parts=system_ui_parts))

                if user_ui_parts:
                    result.append(UIMessage(id=str(uuid.uuid4()), role='user', parts=user_ui_parts))

            elif isinstance(  # pragma: no branch
                msg, ModelResponse
            ):
                ui_parts: list[UIMessagePart] = cls._dump_response_message(msg, tool_results)
                if ui_parts:  # pragma: no branch
                    result.append(UIMessage(id=str(uuid.uuid4()), role='assistant', parts=ui_parts))
            else:
                assert_never(msg)

        return result


def _convert_user_prompt_part(part: UserPromptPart) -> list[UIMessagePart]:
    """Convert a UserPromptPart to a list of UI message parts."""
    ui_parts: list[UIMessagePart] = []

    if isinstance(part.content, str):
        ui_parts.append(TextUIPart(text=part.content, state='done'))
    else:
        for item in part.content:
            if isinstance(item, str):
                ui_parts.append(TextUIPart(text=item, state='done'))
            elif isinstance(item, BinaryContent):
                ui_parts.append(FileUIPart(url=item.data_uri, media_type=item.media_type))
            elif isinstance(item, ImageUrl | AudioUrl | VideoUrl | DocumentUrl):
                ui_parts.append(FileUIPart(url=item.url, media_type=item.media_type))
            elif isinstance(item, CachePoint):
                # CachePoint is metadata for prompt caching, skip for UI conversion
                pass
            else:
                assert_never(item)

    return ui_parts
