"""Vercel AI adapter for handling requests."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import TypeAdapter
from typing_extensions import assert_never

from ...messages import (
    AudioUrl,
    BaseToolCallPart,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
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
from .request_types import (
    DataUIPart,
    DynamicToolInputAvailablePart,
    DynamicToolOutputAvailablePart,
    DynamicToolOutputErrorPart,
    DynamicToolUIPart,
    FileUIPart,
    ReasoningUIPart,
    RequestData,
    SourceDocumentUIPart,
    SourceUrlUIPart,
    StepStartUIPart,
    TextUIPart,
    ToolInputAvailablePart,
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
BUILTIN_TOOL_CALL_ID_PREFIX = 'pyd_ai_builtin'


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
                        builder.add(TextPart(content=part.text))
                    elif isinstance(part, ReasoningUIPart):
                        builder.add(ThinkingPart(content=part.text))
                    elif isinstance(part, FileUIPart):
                        try:
                            file = BinaryContent.from_data_uri(part.url)
                        except ValueError as e:  # pragma: no cover
                            # We don't yet handle non-data-URI file URLs returned by assistants, as no Pydantic AI models do this.
                            raise ValueError(
                                'Vercel AI integration can currently only handle assistant file parts with data URIs.'
                            ) from e
                        builder.add(FilePart(content=file))
                    elif isinstance(part, ToolUIPart | DynamicToolUIPart):
                        if isinstance(part, DynamicToolUIPart):
                            tool_name = part.tool_name
                            builtin_tool = False
                        else:
                            tool_name = part.type.removeprefix('tool-')
                            builtin_tool = part.provider_executed

                        tool_call_id = part.tool_call_id

                        args = part.input

                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                pass

                        if builtin_tool:
                            call_part = BuiltinToolCallPart(tool_name=tool_name, tool_call_id=tool_call_id, args=args)
                            builder.add(call_part)

                            if isinstance(part, ToolOutputAvailablePart | ToolOutputErrorPart):
                                if part.state == 'output-available':
                                    output = part.output
                                else:
                                    output = {'error_text': part.error_text, 'is_error': True}

                                provider_name = (
                                    (part.call_provider_metadata or {}).get('pydantic_ai', {}).get('provider_name')
                                )
                                call_part.provider_name = provider_name

                                builder.add(
                                    BuiltinToolReturnPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        content=output,
                                        provider_name=provider_name,
                                    )
                                )
                        else:
                            builder.add(ToolCallPart(tool_name=tool_name, tool_call_id=tool_call_id, args=args))

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

    @classmethod
    def dump_messages(  # noqa: C901
        cls,
        messages: Sequence[ModelMessage],
    ) -> list[UIMessage]:
        """Transform Pydantic AI messages into Vercel AI messages.

        Args:
            messages: A sequence of ModelMessage objects to convert
            _id_generator: Optional ID generator function for testing. If not provided, uses uuid.uuid4().

        Returns:
            A list of UIMessage objects in Vercel AI format
        """

        def _message_id_generator() -> str:
            """Generate a message ID."""
            return uuid.uuid4().hex

        tool_returns: dict[str, ToolReturnPart] = {}
        tool_errors: dict[str, RetryPromptPart] = {}

        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        tool_returns[part.tool_call_id] = part
                    elif isinstance(part, RetryPromptPart) and part.tool_call_id:
                        tool_errors[part.tool_call_id] = part

        result: list[UIMessage] = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                system_ui_parts: list[UIMessagePart] = []
                user_ui_parts: list[UIMessagePart] = []

                for part in msg.parts:
                    if isinstance(part, SystemPromptPart):
                        system_ui_parts.append(TextUIPart(text=part.content, state='done'))
                    elif isinstance(part, UserPromptPart):
                        user_ui_parts.extend(_convert_user_prompt_part(part))
                    elif isinstance(part, ToolReturnPart | RetryPromptPart):
                        # Tool returns/errors don't create separate UI parts
                        # They're merged into the tool call in the assistant message
                        pass
                    else:
                        assert_never(part)

                if system_ui_parts:
                    result.append(UIMessage(id=_message_id_generator(), role='system', parts=system_ui_parts))

                if user_ui_parts:
                    result.append(UIMessage(id=_message_id_generator(), role='user', parts=user_ui_parts))

            elif isinstance(  # pragma: no branch
                msg, ModelResponse
            ):
                ui_parts: list[UIMessagePart] = []

                # For builtin tools, returns can be in the same ModelResponse as calls
                # Build a local mapping for this message
                local_builtin_returns: dict[str, BuiltinToolReturnPart] = {}
                for part in msg.parts:
                    if isinstance(part, BuiltinToolReturnPart):
                        local_builtin_returns[part.tool_call_id] = part

                for part in msg.parts:
                    if isinstance(part, BuiltinToolReturnPart):
                        # Skip builtin tool returns - they're handled by the tool call logic
                        continue
                    elif isinstance(part, TextPart):
                        # Combine consecutive text parts by checking the last UI part
                        if ui_parts and isinstance(ui_parts[-1], TextUIPart):
                            last_text = ui_parts[-1]
                            ui_parts[-1] = last_text.model_copy(update={'text': last_text.text + part.content})
                        else:
                            ui_parts.append(TextUIPart(text=part.content, state='done'))
                    elif isinstance(part, ThinkingPart):
                        ui_parts.append(ReasoningUIPart(text=part.content, state='done'))
                    elif isinstance(part, FilePart):
                        ui_parts.append(
                            FileUIPart(
                                url=part.content.data_uri,
                                media_type=part.content.media_type,
                            )
                        )
                    elif isinstance(part, BaseToolCallPart):
                        if isinstance(part, BuiltinToolCallPart):
                            prefixed_id = _make_builtin_tool_call_id(part.provider_name, part.tool_call_id)
                            builtin_return = local_builtin_returns.get(part.tool_call_id)

                            if builtin_return:
                                content = builtin_return.model_response_str()
                                call_provider_metadata = (
                                    {'pydantic_ai': {'provider_name': part.provider_name}}
                                    if part.provider_name
                                    else None
                                )
                                ui_parts.append(
                                    ToolOutputAvailablePart(
                                        type=f'tool-{part.tool_name}',
                                        tool_call_id=prefixed_id,
                                        input=part.args_as_json_str(),
                                        output=content,
                                        state='output-available',
                                        provider_executed=True,
                                        call_provider_metadata=call_provider_metadata,
                                    )
                                )
                            else:  # pragma: no cover
                                ui_parts.append(
                                    ToolInputAvailablePart(
                                        type=f'tool-{part.tool_name}',
                                        tool_call_id=prefixed_id,
                                        input=part.args_as_json_str(),
                                        state='input-available',
                                        provider_executed=True,
                                    )
                                )
                        else:
                            tool_return = tool_returns.get(part.tool_call_id)
                            tool_error = tool_errors.get(part.tool_call_id)

                            if tool_return and isinstance(tool_return, ToolReturnPart):
                                content = tool_return.model_response_str()
                                ui_parts.append(
                                    DynamicToolOutputAvailablePart(
                                        tool_name=part.tool_name,
                                        tool_call_id=part.tool_call_id,
                                        input=part.args_as_json_str(),
                                        output=content,
                                        state='output-available',
                                    )
                                )
                            elif tool_error:
                                error_text = tool_error.model_response()
                                ui_parts.append(
                                    DynamicToolOutputErrorPart(
                                        tool_name=part.tool_name,
                                        tool_call_id=part.tool_call_id,
                                        input=part.args_as_json_str(),
                                        error_text=error_text,
                                        state='output-error',
                                    )
                                )
                            else:
                                ui_parts.append(
                                    DynamicToolInputAvailablePart(
                                        tool_name=part.tool_name,
                                        tool_call_id=part.tool_call_id,
                                        input=part.args_as_json_str(),
                                        state='input-available',
                                    )
                                )
                    else:
                        assert_never(part)

                if ui_parts:  # pragma: no branch
                    result.append(UIMessage(id=_message_id_generator(), role='assistant', parts=ui_parts))
            else:
                assert_never(msg)

        return result


def _make_builtin_tool_call_id(provider_name: str | None, tool_call_id: str) -> str:
    """Create a prefixed tool call ID for builtin tools."""
    return f'{BUILTIN_TOOL_CALL_ID_PREFIX}|{provider_name or ""}|{tool_call_id}'


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
            else:
                assert_never(item)

    return ui_parts
