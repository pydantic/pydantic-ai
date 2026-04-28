"""OpenAI Responses protocol adapter for Pydantic AI agents."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import Any

from ... import ExternalToolset, ToolDefinition
from ..._utils import is_str_dict
from ...messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserContent,
    UserPromptPart,
)
from ...output import OutputDataT
from ...tools import AgentDepsT
from ...toolsets import AbstractToolset

try:
    from openai.types.responses import ResponseInputItemParam
    from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
    from openai.types.responses.tool_param import ToolParam

    from .. import MessagesBuilder, UIAdapter, UIEventStream
    from ._event_stream import ResponsesEventStream
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` and `starlette` packages to use the Responses integration, '
        'you can use the `responses` optional group — `pip install "pydantic-ai-slim[responses]"`'
    ) from e

__all__ = ['ResponsesAdapter']


class _ResponsesFrontendToolset(ExternalToolset[AgentDepsT]):
    """Toolset for OpenAI Responses API frontend function tools."""

    def __init__(self, tools: Iterable[ToolParam]):
        tool_defs: list[ToolDefinition] = []
        for raw_tool in tools:
            if not is_str_dict(raw_tool) or raw_tool.get('type') != 'function':
                continue
            name = raw_tool.get('name')
            if not isinstance(name, str):
                continue
            description = raw_tool.get('description')
            parameters = raw_tool.get('parameters')
            strict = raw_tool.get('strict')
            parameters_schema = parameters if is_str_dict(parameters) else {}
            tool_defs.append(
                ToolDefinition(
                    name=name,
                    description=description if isinstance(description, str) else None,
                    parameters_json_schema=parameters_schema,
                    strict=strict if isinstance(strict, bool) else None,
                )
            )
        super().__init__(tool_defs)

    @property
    def label(self) -> str:
        return 'the Responses API frontend tools'  # pragma: no cover


def _parse_user_content_part(part: dict[str, Any]) -> UserContent | None:
    """Parse one Responses input user-content part into a Pydantic AI UserContent value.

    The wire shape is one of `ResponseInputTextParam`, `ResponseInputImageParam`,
    or `ResponseInputFileParam`; we accept the loose `dict[str, Any]` because
    pyright's TypedDict narrowing on the `type` discriminator does not flow
    through helper boundaries.
    """
    part_type = part.get('type')

    if part_type in ('input_text', 'text'):
        text = part.get('text')
        return text if isinstance(text, str) else None

    if part_type == 'input_image':
        image_url = part.get('image_url')
        if not isinstance(image_url, str):
            return None
        if image_url.startswith('data:'):
            return BinaryContent.from_data_uri(image_url)
        return ImageUrl(url=image_url)

    if part_type == 'input_file':
        file_data = part.get('file_data')
        if isinstance(file_data, str) and file_data.startswith('data:'):
            return BinaryContent.from_data_uri(file_data)
        file_id = part.get('file_id')
        if isinstance(file_id, str) and file_id:
            return UploadedFile(file_id=file_id, provider_name='openai')
        return None

    return None


class ResponsesAdapter(UIAdapter[ResponseCreateParamsStreaming, ResponseInputItemParam, Any, AgentDepsT, OutputDataT]):
    """UI adapter for the OpenAI Responses protocol."""

    @classmethod
    def build_run_input(cls, body: bytes) -> ResponseCreateParamsStreaming:
        """Build a Responses input object from the request body.

        The OpenAI SDK's `ResponseCreateParamsStreaming` is a structural TypedDict;
        constructing a `pydantic.TypeAdapter` for it produces an overly strict
        validator that rejects the heterogeneous `input` and `tools` unions in
        practice. We rely on the SDK's own runtime contract instead.
        """
        raw_data = json.loads(body)
        if not is_str_dict(raw_data):
            raise ValueError('Responses API request body must be a JSON object.')
        return raw_data  # pyright: ignore[reportReturnType]

    def build_event_stream(self) -> UIEventStream[ResponseCreateParamsStreaming, Any, AgentDepsT, OutputDataT]:
        return ResponsesEventStream(
            self.run_input,
            accept=self.accept,
            frontend_tool_names=self.frontend_tool_names,
        )

    @cached_property
    def frontend_tool_names(self) -> frozenset[str]:
        """Names of tools the client declared in the request `tools` array.

        Used by the event stream to distinguish frontend tool calls (which surface
        as `function_call` output items the client must round-trip) from backend
        agent-registered tool calls (which run server-side and are suppressed).
        """
        tools = self.run_input.get('tools')
        if not isinstance(tools, list):
            return frozenset()
        names: set[str] = set()
        for raw_tool in tools:
            if not is_str_dict(raw_tool) or raw_tool.get('type') != 'function':
                continue
            name = raw_tool.get('name')
            if isinstance(name, str):
                names.add(name)
        return frozenset(names)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the Responses input."""
        builder = MessagesBuilder()

        instructions = self.run_input.get('instructions')
        if isinstance(instructions, str) and instructions:
            builder.add(SystemPromptPart(content=instructions))

        input_data = self.run_input.get('input')
        if isinstance(input_data, str):
            builder.add(UserPromptPart(content=input_data))
        elif isinstance(input_data, list):
            self._load_input_items(input_data, builder)

        return builder.messages

    @classmethod
    def _load_input_items(cls, items: Sequence[ResponseInputItemParam], builder: MessagesBuilder) -> None:
        tool_call_names: dict[str, str] = {}

        for raw_item in items:
            if not is_str_dict(raw_item):
                continue

            item_type = raw_item.get('type')
            role = raw_item.get('role')

            # Implicit message item: has role + content but no explicit type
            if item_type == 'message' or (item_type is None and role is not None and 'content' in raw_item):
                cls._load_message_item(raw_item, builder)
                continue

            if item_type == 'function_call':
                call_id_raw = raw_item.get('call_id') or raw_item.get('id') or ''
                name_raw = raw_item.get('name', '')
                if isinstance(call_id_raw, str) and isinstance(name_raw, str) and name_raw:
                    tool_call_names[call_id_raw] = name_raw
                    builder.add(
                        ToolCallPart(
                            tool_name=name_raw,
                            tool_call_id=call_id_raw,
                            args=raw_item.get('arguments'),
                        )
                    )
                continue

            if item_type == 'function_call_output':
                call_id_raw = raw_item.get('call_id', '')
                if not isinstance(call_id_raw, str):
                    continue
                tool_name = tool_call_names.get(call_id_raw)
                if tool_name is None:
                    raise ValueError(f'function_call_output references unknown call_id {call_id_raw!r}.')
                builder.add(
                    ToolReturnPart(
                        tool_name=tool_name,
                        tool_call_id=call_id_raw,
                        content=str(raw_item.get('output', '')),
                    )
                )
                continue

    @classmethod
    def _load_message_item(cls, item: dict[str, Any], builder: MessagesBuilder) -> None:
        role = item.get('role')
        content = item.get('content')

        if role in ('system', 'developer'):
            text = cls._extract_text_content(content)
            if text:
                builder.add(SystemPromptPart(content=text))
            return

        if role == 'user':
            if isinstance(content, str):
                builder.add(UserPromptPart(content=content))
                return
            if isinstance(content, list):
                parts: list[UserContent] = []
                for raw_c in content:  # pyright: ignore[reportUnknownVariableType]
                    if not is_str_dict(raw_c):
                        continue
                    parsed = _parse_user_content_part(raw_c)
                    if parsed is not None:
                        parts.append(parsed)
                if parts:
                    builder.add(UserPromptPart(content=parts))
            return

        if role == 'assistant':
            text = cls._extract_text_content(content)
            if text:
                builder.add(TextPart(content=text))

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: list[str] = []
            for raw_c in content:  # pyright: ignore[reportUnknownVariableType]
                if not is_str_dict(raw_c):
                    continue
                if raw_c.get('type') in ('input_text', 'text', 'output_text'):
                    text = raw_c.get('text')
                    if isinstance(text, str):
                        texts.append(text)
            return ''.join(texts)
        return ''

    @cached_property
    def toolset(self) -> AbstractToolset[AgentDepsT] | None:
        tools = self.run_input.get('tools')
        if isinstance(tools, list) and tools:
            return _ResponsesFrontendToolset[AgentDepsT](tools)
        return None

    @cached_property
    def state(self) -> dict[str, Any] | None:
        metadata = self.run_input.get('metadata')
        if isinstance(metadata, dict) and metadata:
            return metadata
        return None

    @classmethod
    def load_messages(cls, messages: Sequence[ResponseInputItemParam]) -> list[ModelMessage]:
        """Transform a flat sequence of Responses input items into Pydantic AI messages."""
        builder = MessagesBuilder()
        cls._load_input_items(messages, builder)
        return builder.messages
