"""OpenAI Responses protocol adapter for Pydantic AI agents."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

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
from ...output import NativeOutput, OutputDataT, OutputSpec, StructuredDict
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


ResponsesMode = Literal['openai_compat', 'openresponses', 'auto']
"""Mode for [`ResponsesAdapter`][pydantic_ai.ui.responses.ResponsesAdapter] emission.

- `'openai_compat'`: strict OpenAI Responses subset. Backend (agent-registered) tool
  calls are hidden from the wire; round-trip across turns is lossy without server-side
  state. Vanilla `openai-python` SDK clients work cleanly.
- `'openresponses'`: [OpenResponses](https://www.openresponses.org/specification) extended
  emission. Backend tool calls surface as `pydantic_ai:custom_tool_call` paired with
  `pydantic_ai:custom_tool_call_output` items. Lossless round-trip; requires an
  OpenResponses-aware client (Vercel AI Gateway, HuggingFace, OpenResponses-conformant
  SDKs).
- `'auto'` (default): structurally sniff the request input — any item type carrying a
  `<slug>:` extension prefix means an OpenResponses-aware client; otherwise emit
  `openai_compat`. Lets a single endpoint serve both kinds of clients.
"""


@dataclass
class ResponsesAdapter(UIAdapter[ResponseCreateParamsStreaming, ResponseInputItemParam, Any, AgentDepsT, OutputDataT]):
    """UI adapter for the OpenAI Responses protocol."""

    mode: ResponsesMode = 'auto'
    """How the adapter emits backend tool calls and parses extension input items.

    See [`ResponsesMode`][pydantic_ai.ui.responses.ResponsesMode] for the three modes.
    """

    manage_system_prompt: Literal['server', 'client'] = 'client'
    """Override of the base [`UIAdapter`][pydantic_ai.ui.UIAdapter] default of `'server'`.

    Responses is a generic API where requests typically carry an `instructions` field
    that the caller controls; trusting the client by default matches the spec's
    semantics. Flip to `'server'` for adversarial-client deployments.
    """

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
            mode=self.resolved_mode,
        )

    @cached_property
    def resolved_mode(self) -> Literal['openai_compat', 'openresponses']:
        """Concrete emission mode after resolving `'auto'`.

        - `'auto'` → structurally sniff `run_input['input']` for any item type carrying
          a `<slug>:` extension prefix (per the OpenResponses spec, vendor extensions
          MUST be prefixed with the implementer's slug). Any such item means the client
          speaks OpenResponses; otherwise the client is treated as a vanilla OpenAI
          Responses SDK.
        - `'openai_compat'` / `'openresponses'` → returned as-is.
        """
        if self.mode != 'auto':
            return self.mode
        items = self.run_input.get('input')
        if isinstance(items, list):
            for raw_item in items:
                if not is_str_dict(raw_item):
                    continue
                item_type = raw_item.get('type')
                if isinstance(item_type, str) and ':' in item_type:
                    return 'openresponses'
        return 'openai_compat'

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
        """Pydantic AI messages from the Responses input.

        Note: `run_input['instructions']` is intentionally NOT mapped onto a
        `SystemPromptPart` here — it surfaces via [`request_instructions`][pydantic_ai.ui.UIAdapter.request_instructions]
        and is passed to the agent as a per-run `instructions=` kwarg, matching the
        OpenAI Responses spec semantic ("appended to the model's instructions for this
        request"). Routing through `SystemPromptPart` would be filtered by
        `manage_system_prompt='server'` in deployments that flip back to server-managed
        system prompts.
        """
        builder = MessagesBuilder()

        input_data = self.run_input.get('input')
        if isinstance(input_data, str):
            builder.add(UserPromptPart(content=input_data))
        elif isinstance(input_data, list):
            self._load_input_items(input_data, builder)

        return builder.messages

    @cached_property
    def request_instructions(self) -> str | None:
        instructions = self.run_input.get('instructions')
        return instructions if isinstance(instructions, str) and instructions else None

    @cached_property
    def request_output_type(self) -> OutputSpec[Any] | None:
        text_cfg = self.run_input.get('text')
        if not is_str_dict(text_cfg):
            return None
        format_cfg = text_cfg.get('format')
        if not is_str_dict(format_cfg):
            return None

        format_type = format_cfg.get('type')
        if format_type == 'json_schema':
            schema = format_cfg.get('schema')
            if not is_str_dict(schema):
                return None
            name_raw = format_cfg.get('name')
            description_raw = format_cfg.get('description')
            strict_raw = format_cfg.get('strict')
            structured = StructuredDict(
                schema,
                name=name_raw if isinstance(name_raw, str) else None,
                description=description_raw if isinstance(description_raw, str) else None,
            )
            return NativeOutput(
                structured,
                name=name_raw if isinstance(name_raw, str) else None,
                description=description_raw if isinstance(description_raw, str) else None,
                strict=strict_raw if isinstance(strict_raw, bool) else None,
            )
        if format_type == 'json_object':
            return dict[str, Any]
        return None

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

            # Frontend tool round-trip: `function_call` (in) → `function_call_output` (out).
            # OpenResponses backend tool round-trip: `pydantic_ai:custom_tool_call` paired with
            # `pydantic_ai:custom_tool_call_output`. Both shapes carry the same
            # `(call_id, name, arguments)` / `(call_id, output)` payload — the difference is the
            # `type` discriminator. Treat them uniformly here so message-history reconstruction
            # is symmetric across modes.
            if item_type in ('function_call', 'pydantic_ai:custom_tool_call'):
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

            if item_type in ('function_call_output', 'pydantic_ai:custom_tool_call_output'):
                call_id_raw = raw_item.get('call_id', '')
                if not isinstance(call_id_raw, str):
                    continue
                tool_name = tool_call_names.get(call_id_raw)
                if tool_name is None:
                    raise ValueError(f'{item_type} references unknown call_id {call_id_raw!r}.')
                builder.add(
                    ToolReturnPart(
                        tool_name=tool_name,
                        tool_call_id=call_id_raw,
                        content=str(raw_item.get('output', '')),
                    )
                )
                continue

            # OpenResponses extension item that an outer agent emits to feed an inner agent
            # narrative / observation context. v1 maps all three roles onto a SystemPromptPart
            # with the provenance encoded in the prefix; reversible later via a dedicated inbound
            # part if a use case justifies it.
            if item_type == 'pydantic_ai:agent_context':
                from_agent = raw_item.get('from_agent', '')
                role = raw_item.get('role', 'context')
                content = raw_item.get('content', '')
                if isinstance(from_agent, str) and isinstance(role, str) and isinstance(content, str) and content:
                    builder.add(SystemPromptPart(content=f'[from {from_agent}, role={role}] {content}'))
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

    @cached_property
    def previous_response_id(self) -> str | None:
        """The `previous_response_id` field from the request, if set.

        Exposed separately from [`conversation_id`][pydantic_ai.ui.responses.ResponsesAdapter.conversation_id]
        so user code can distinguish "client wants to continue this specific prior response"
        from "client wants to label this run with a thread ID". When both are present the
        adapter prefers `conversation` for `conversation_id` but keeps `previous_response_id`
        accessible here for `history_loader` lookups.
        """
        value = self.run_input.get('previous_response_id')
        return value if isinstance(value, str) and value else None

    @cached_property
    def conversation_id(self) -> str | None:
        """Conversation ID derived from the request.

        Priority chain (per OpenAI Responses semantics):

        1. `conversation` — explicit conversation thread label (the [Conversations API](https://platform.openai.com/docs/api-reference/conversations) field).
           More specific than `previous_response_id`; wins when both are set.
        2. `previous_response_id` — point-to-point continuation reference. Used as the conversation
           key when no explicit `conversation` is provided.
        3. `None` — request is a fresh standalone run.
        """
        conversation = self.run_input.get('conversation')
        if isinstance(conversation, str) and conversation:
            return conversation
        return self.previous_response_id

    @classmethod
    def load_messages(cls, messages: Sequence[ResponseInputItemParam]) -> list[ModelMessage]:
        """Transform a flat sequence of Responses input items into Pydantic AI messages."""
        builder = MessagesBuilder()
        cls._load_input_items(messages, builder)
        return builder.messages
