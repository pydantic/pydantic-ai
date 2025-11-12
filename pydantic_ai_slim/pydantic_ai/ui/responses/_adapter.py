"""Responses protocol adapter for Pydantic AI agents."""

import base64
import json
from functools import cached_property
from pathlib import Path
from typing import Any, cast, Sequence, Union

from openai.types.responses import ResponseCreateParams, ResponseInputParam, ToolParam
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
from pydantic import TypeAdapter

from ... import ExternalToolset, ModelMessage, ToolDefinition
from ...messages import (
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from ...output import OutputDataT
from ...tools import AgentDepsT
from ...toolsets import AbstractToolset
from ...ui import UIAdapter, UIEventStream, MessagesBuilder

from ._event_stream import ResponsesEventStream
from ._type_guards import (
    _is_input_text_item,
    _is_input_image_item,
    _is_input_file_item,
    _is_message_item,
    _is_function_call_output_item,
    _is_system_role,
    _is_user_role,
    _is_assistant_role,
    _is_text_content_part,
)

# Format mappings for media types
_document_format_inverse_lookup = {
    '.pdf': 'application/pdf',
    '.txt': 'text/plain',
    '.csv': 'text/csv',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.html': 'text/html',
    '.md': 'text/markdown',
    '.xls': 'application/vnd.ms-excel',
}

_audio_format_inverse_lookup = {
    'mp3': 'audio/mpeg',
    'wav': 'audio/wav',
    'flac': 'audio/flac',
    'oga': 'audio/ogg',
    'aiff': 'audio/aiff',
    'aac': 'audio/aac',
}

_image_format_inverse_lookup = {
    'jpeg': 'image/jpeg',
    'png': 'image/png',
    'gif': 'image/gif',
    'webp': 'image/webp',
}


class _ResponsesFrontendToolset(ExternalToolset[AgentDepsT]):
    """Toolset for Responses API frontend tools."""

    def __init__(self, tools: list[dict[str, Any]]):
        """Initialize the toolset with Responses API tools.

        Args:
            tools: List of tool definitions from the Responses API.
        """
        tool_defs: list[ToolDefinition] = []
        for tool in tools:
            if isinstance(tool, dict) and tool.get('type') == 'function':
                func = tool.get('function', {})
                if isinstance(func, dict) and 'name' in func:
                    tool_defs.append(
                        ToolDefinition(
                            name=func['name'],
                            description=func.get('description'),
                            parameters_json_schema=func.get('parameters'),
                        )
                    )
        super().__init__(tool_defs)

    @property
    def label(self) -> str:
        """Return the label for this toolset."""
        return 'the Responses API frontend tools'


def _parse_response_user_content_part(part: dict[str, Any]) -> UserContent | None:
    """Parse a single Responses API user content part.

    Args:
        part: A content part which can be input_text, input_image, or input_file.

    Returns:
        A UserContent object or None if the part cannot be parsed.
    """
    if _is_input_text_item(part):
        return part.get('text', '')

    if _is_input_image_item(part):
        image_url = part.get('image_url')
        if not image_url:
            return None
        # If it's a data URI, convert to BinaryContent; otherwise ImageUrl
        if isinstance(image_url, str) and image_url.startswith('data:'):
            try:
                bc = BinaryContent.from_data_uri(image_url)
                # carry detail if present
                detail = part.get('detail')
                if detail is not None:
                    bc.vendor_metadata = {**(bc.vendor_metadata or {}), 'detail': detail}
                return bc
            except Exception:
                return ImageUrl(url=str(image_url))
        else:
            vendor_metadata = {}
            if (detail := part.get('detail')) is not None:
                vendor_metadata['detail'] = detail
            return ImageUrl(url=str(image_url), vendor_metadata=vendor_metadata or None)

    if _is_input_file_item(part):
        data_uri = part.get('file_data')
        if isinstance(data_uri, str) and data_uri.startswith('data:'):
            try:
                return BinaryContent.from_data_uri(data_uri)
            except Exception:
                return None
        # If file_data missing or not a data URI, ignore for now
        return None

    return None


class ResponsesAdapter(
    UIAdapter[ResponseCreateParamsStreaming, Union[str, ResponseInputParam], Any, AgentDepsT, OutputDataT]
):
    """UI adapter for the OpenAI Responses protocol."""

    _RESPONSES_TA: TypeAdapter[ResponseCreateParamsStreaming] = TypeAdapter(ResponseCreateParamsStreaming)

    @classmethod
    def build_run_input(cls, body: bytes) -> ResponseCreateParamsStreaming:
        """Build a Responses input object from the request body."""
        # Parse JSON first to extract raw tools before validation
        raw_data = json.loads(body)

        # Extract tools if present (before validation to avoid type mismatch)
        raw_tools = raw_data.pop('tools', None)

        # Validate the rest of the params
        params = cls._RESPONSES_TA.validate_python(raw_data)

        # Add back the raw tools without validation
        if raw_tools is not None:
            params['tools'] = raw_tools

        return params

    def build_event_stream(self) -> UIEventStream[ResponseCreateParamsStreaming, Any, AgentDepsT, OutputDataT]:
        """Build a Responses event stream transformer."""
        return ResponsesEventStream(self.run_input, accept=self.accept)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the Responses input."""
        run_input = cast(ResponseCreateParamsStreaming, self.run_input)

        # Extract input and instructions from Responses API format
        input_data = run_input.get('input')
        instructions = run_input.get('instructions')

        # Build a sequence of message items for load_messages
        # We need to handle both input_data and instructions together
        messages_seq: list[Union[str, ResponseInputParam]] = []
        
        # Add instructions first if present (wrap as system message item)
        if instructions:
            # Wrap instructions as a message item with system role
            messages_seq.append([{'role': 'system', 'content': instructions}])
        
        # Add input data
        # The Responses API allows input to be either a string or a list of items
        if input_data is not None:
            messages_seq.append(input_data)

        return self.load_messages(messages_seq)

    @cached_property
    def toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Toolset representing frontend tools from the Responses run input."""
        run_input = cast(ResponseCreateParamsStreaming, self.run_input)
        tools = run_input.get('tools')
        if tools:
            # Tools are already a raw list from build_run_input, no conversion needed
            if isinstance(tools, list) and tools:
                return _ResponsesFrontendToolset[AgentDepsT](tools)
        return None

    @cached_property
    def state(self) -> dict[str, Any] | None:
        """Frontend state from the Responses run input."""
        run_input = cast(ResponseCreateParamsStreaming, self.run_input)
        metadata = run_input.get('metadata')
        if metadata and isinstance(metadata, dict):
            return metadata
        return None

    @classmethod
    def load_messages(cls, messages: Sequence[Union[str, ResponseInputParam]]) -> list[ModelMessage]:
        """Transform OpenAI Responses API input into Pydantic AI messages.
        
        Args:
            messages: Sequence of input data - each can be str or list of items.
            
        Returns:
            A list of ModelMessage objects representing the conversation history.
        """
        builder = MessagesBuilder()

        # Process each message in the sequence
        for input_data in messages:
            # Handle input - can be string or list
            if isinstance(input_data, str):
                # Simple string input becomes a user message
                builder.add(UserPromptPart(content=input_data))
            elif isinstance(input_data, list):
                for item in input_data:
                    if not isinstance(item, dict):
                        continue

                    if _is_message_item(item):
                        content = item.get('content')

                        if _is_system_role(item):
                            # System/developer both map to system prompt
                            if isinstance(content, str):
                                builder.add(SystemPromptPart(content=content))
                            elif isinstance(content, list):
                                # Concatenate text parts into one system string; ignore non-text
                                texts: list[str] = []
                                for c in content:
                                    if _is_text_content_part(c):
                                        txt = c.get('text') or c.get('content') or ''
                                        if isinstance(txt, str):
                                            texts.append(txt)
                                if texts:
                                    builder.add(SystemPromptPart(content=''.join(texts)))

                        elif _is_user_role(item):
                            if isinstance(content, str):
                                builder.add(UserPromptPart(content=content))
                            elif isinstance(content, list):
                                parts: list[UserContent] = []
                                for c in content:
                                    if isinstance(c, dict):
                                        parsed = _parse_response_user_content_part(c)
                                        if parsed is not None:
                                            parts.append(parsed)
                                if parts:
                                    builder.add(UserPromptPart(content=parts))

                        elif _is_assistant_role(item):
                            # Assistant content to ModelResponse
                            if isinstance(content, str):
                                builder.add(TextPart(content=content))
                            elif isinstance(content, list):
                                txt = ''.join(
                                    str(c.get('text', ''))
                                    for c in content
                                    if _is_text_content_part(c)
                                )
                                if txt:
                                    builder.add(TextPart(content=txt))

                    elif _is_function_call_output_item(item):
                        call_id = item.get('call_id', '')
                        output = item.get('output', '')
                        builder.add(ToolReturnPart(tool_name=call_id, tool_call_id=call_id, content=str(output)))

        return builder.messages
