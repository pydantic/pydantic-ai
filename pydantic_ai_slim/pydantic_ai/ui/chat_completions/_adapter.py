import base64
import warnings
from functools import cached_property
from pathlib import Path
from typing import cast, Iterable, Any, Literal, TypeAlias

from openai.types.chat import (
    CompletionCreateParams,
    ChatCompletionMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
)
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from pydantic import TypeAdapter

from ... import ModelMessage, AbstractToolset, ExternalToolset, ToolDefinition
from ...messages import (
    BinaryContent,
    ImageUrl,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from ...output import OutputDataT
from ...tools import AgentDepsT
from ...ui import UIAdapter, UIEventStream, MessagesBuilder
from ._type_guards import (
    _is_function_tool_param, _is_custom_tool_param, _is_text_part, _is_image_part, _is_audio_part,
    _is_file_part, _is_assistant_message, _is_system_message, _is_user_message, _is_tool_message,
    _is_function_message, _is_assistant_text_part, _is_assistant_refusal_part, _is_message_function_tool_call_param,
    _is_message_custom_tool_call_param
)

OpenAIRole: TypeAlias = Literal['developer', 'system', 'assistant', 'user', 'tool', 'function']

# Format mappings for media types
_audio_format_inverse_lookup = {
    'wav': 'audio/wav',
    'mp3': 'audio/mpeg',
}

_document_format_inverse_lookup = {
    '.pdf': 'application/pdf',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.xls': 'application/vnd.ms-excel',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.ppt': 'application/vnd.ms-powerpoint',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.txt': 'text/plain',
    '.csv': 'text/csv',
    '.html': 'text/html',
    '.htm': 'text/html',
}


def _parse_user_content_part(part: ChatCompletionContentPartParam) -> UserContent:
    """Parse a single user content part from OpenAI format.

    Args:
        part: A content part which can be text, image, audio, or file.

    Returns:
        A UserContent object.

    Raises:
        ValueError: If the part type is not recognized or media type is unknown.
    """
    if _is_file_part(part):
        # Encode base64 string as bytes
        b_data = base64.b64decode(part['file']['file_data'].encode())
        filename = part['file']['filename']
        file_ext = Path(filename).suffix
        try:
            media_type = _document_format_inverse_lookup[file_ext]
        except KeyError as e:
            raise ValueError(f'Unknown media type: {file_ext}') from e

        return BinaryContent(data=b_data, media_type=media_type)

    if _is_text_part(part):
        return part['text']

    elif _is_image_part(part):
        url = part['image_url']['url']
        return ImageUrl(url=url)

    elif _is_audio_part(part):
        input_audio = part['input_audio']
        data = input_audio['data']
        format_value = input_audio['format']

        try:
            media_type = _audio_format_inverse_lookup[format_value]
        except KeyError as e:
            raise ValueError(f'Unknown media type: {format_value}') from e
        b_data = base64.b64decode(data.encode())
        return BinaryContent(data=b_data, media_type=media_type)

    raise ValueError(f'Unknown part type: {part}')


class ChatCompletionsAdapter(
    UIAdapter[CompletionCreateParams, ChatCompletionMessageParam, ChatCompletionChunk, AgentDepsT, OutputDataT]):
    """UI adapter for the Chat Completions protocol."""

    @classmethod
    def build_run_input(cls, body: bytes) -> CompletionCreateParams:
        """Build a Chat Completions input object from the request object."""

        type_adapter: TypeAdapter[CompletionCreateParams] = TypeAdapter(CompletionCreateParams)
        out = type_adapter.validate_json(body)
        return out

    def build_event_stream(self) -> UIEventStream[CompletionCreateParams, ChatCompletionChunk, AgentDepsT, OutputDataT]:
        """Build a Chat Completions event stream transformer."""
        from ._event_stream import ChatCompletionsEventStream
        return ChatCompletionsEventStream(self.run_input, accept=self.accept)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the Chat Completions input"""
        run_input = cast(CompletionCreateParamsBase, self.run_input)
        messages = run_input["messages"]

        return self.load_messages(messages)

    @cached_property
    def toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Toolset representing frontend tools from the Chat Completions run input."""
        run_input = cast(CompletionCreateParamsBase, self.run_input)

        tools = run_input["tools"]

        tool_defs = []

        for tool in tools:
            if _is_function_tool_param(tool):
                fn = tool["function"]
                tool_defs.append(ToolDefinition(
                    name=fn["name"],
                    description=fn["description"],
                    parameters_json_schema=fn["parameters"],
                    strict=fn["strict"]
                ))

            elif _is_custom_tool_param(tool):
                warnings.warn(  # pragma: no cover
                    f'Custom tool parameters are not supported in the Chat Completions adapter.',
                    UserWarning,
                )

        if len(tool_defs) > 0:
            return ExternalToolset[AgentDepsT](tool_defs)

        else:
            return None

    @cached_property
    def state(self) -> dict[str, Any] | None:
        return None

    @classmethod
    def load_messages(cls, messages: Iterable[ChatCompletionMessageParam]) -> list[ModelMessage]:
        """Transform OpenAI Chat Completion messages into Pydantic AI messages.
        
        Args:
            messages: An iterable of OpenAI chat completion message parameters.
            
        Returns:
            A list of ModelMessage objects representing the conversation history.
        """
        builder = MessagesBuilder()

        for message in messages:
            if _is_assistant_message(message):
                # Handle assistant messages - add text and tool calls
                if content := message['content']:
                    if isinstance(content, str):
                        builder.add(TextPart(content=content))
                    else:
                        # Handle Iterable[ContentArrayOfContentPart]
                        text_parts = []
                        for part in content:
                            if _is_assistant_text_part(part):
                                text_parts.append(part['text'])
                            elif _is_assistant_refusal_part(part):
                                text_parts.append(part['refusal'])
                        
                        if len(text_parts) > 0:
                            builder.add(TextPart(content=' '.join(text_parts)))

                for tc in message['tool_calls']:
                    if _is_message_function_tool_call_param(tc):
                        builder.add(
                            ToolCallPart(
                                tool_call_id=tc['id'],
                                tool_name=tc['function']['name'],
                                args=tc['function']['arguments'],
                            )
                        )

                    elif _is_message_custom_tool_call_param(tc):
                        builder.add(
                            ToolCallPart(
                                tool_call_id=tc['id'],
                                tool_name=tc['custom']['name'],
                                # Input is arbitrary string
                                args=tc['custom']['input'],
                            )
                        )

            elif _is_system_message(message):
                content = message['content']
                builder.add(SystemPromptPart(content=str(content)))

            elif _is_user_message(message):
                content = message['content']
                if isinstance(content, str):
                    builder.add(UserPromptPart(content=content))
                else:
                    parts: list[UserContent] = []
                    for part in content:
                        parsed_part = _parse_user_content_part(part)
                        if parsed_part is not None:
                            parts.append(parsed_part)

                    if parts:
                        builder.add(UserPromptPart(content=parts))

            elif _is_tool_message(message):
                builder.add(
                    ToolReturnPart(
                        tool_name=message["tool_call_id"],
                        tool_call_id=message['tool_call_id'],
                        content=str(message['content'])
                    )
                )

            elif _is_function_message(message):
                # Function messages are deprecated but still supported
                builder.add(
                    ToolReturnPart(
                        tool_name=message["name"],
                        tool_call_id=message['name'],  # function role uses 'name' field
                        content=str(message['content']),
                    )
                )

        return builder.messages
