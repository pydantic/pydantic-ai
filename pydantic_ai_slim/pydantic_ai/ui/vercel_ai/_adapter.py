"""Vercel AI adapter for handling requests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import TypeAdapter
from typing_extensions import assert_never

from ...messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
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
    DynamicToolUIPart,
    FileUIPart,
    ReasoningUIPart,
    RequestData,
    SourceDocumentUIPart,
    SourceUrlUIPart,
    StepStartUIPart,
    TextUIPart,
    ToolOutputAvailablePart,
    ToolOutputErrorPart,
    ToolUIPart,
    UIMessage,
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
