"""OpenAI Responses API request types accepted by the [`Agent.to_responses()`][pydantic_ai.agent.Agent.to_responses] endpoint.

These mirror the subset of the [Responses API request body](https://platform.openai.com/docs/api-reference/responses/create)
that the endpoint honors when serving an agent as an OpenAI-compatible endpoint. Unknown fields
(`temperature`, `metadata`, `tools`, `store`, …) are ignored rather than rejected, so standard
OpenAI clients can talk to the endpoint unchanged.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter

__all__ = ['ResponsesRequest', 'ResponsesInputItem', 'responses_request_ta']


class _InputModel(BaseModel):
    model_config = ConfigDict(extra='ignore')


class InputTextContent(_InputModel):
    """A text content part inside a message's `content` list."""

    type: Literal['input_text', 'output_text']
    text: str


MessageContent = str | list[InputTextContent]


class InputMessage(_InputModel):
    """A role-tagged message item, matching the Responses `EasyInputMessage` shape.

    The `type` field is optional in the Responses API for message items, so it defaults to
    `'message'` to let bare `{'role': ..., 'content': ...}` items validate.
    """

    type: Literal['message'] = 'message'
    role: Literal['user', 'assistant', 'system', 'developer']
    content: MessageContent


class FunctionCall(_InputModel):
    """A prior assistant function tool call, replayed by the client for multi-turn history."""

    type: Literal['function_call']
    call_id: str
    name: str
    arguments: str


class FunctionCallOutput(_InputModel):
    """The output of a prior function tool call, replayed by the client for multi-turn history."""

    type: Literal['function_call_output']
    call_id: str
    output: str


# Smart union: variants are distinguished by their required fields (`role` vs `name` vs `output`) and
# the `type` literal, so items without an explicit `type` still resolve to a message.
ResponsesInputItem = InputMessage | FunctionCall | FunctionCallOutput
"""An item in the Responses API `input` list that the endpoint understands."""


class ResponsesRequest(BaseModel):
    """The Responses API request body, restricted to the fields the endpoint honors."""

    model_config = ConfigDict(extra='ignore')

    input: str | list[ResponsesInputItem]
    """The prompt: a bare string, or a list of conversation items to replay as history."""

    model: str | None = None
    """The model ID requested by the client. Informational only; the served agent's model is used."""

    instructions: str | None = None
    """A system/developer message passed as additional instructions for this run."""

    stream: bool = False
    """Whether to stream the response as SSE. The Responses API default is non-streaming."""

    previous_response_id: str | None = None
    """Accepted but not yet honored: server-side conversation state is not supported, so history
    must be replayed via `input`."""


responses_request_ta: TypeAdapter[ResponsesRequest] = TypeAdapter(ResponsesRequest)
"""[`TypeAdapter`][pydantic.TypeAdapter] for validating a request body into a [`ResponsesRequest`][pydantic_ai._responses.types.ResponsesRequest]."""
