"""Serve a Pydantic AI agent as an [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) endpoint.

This lets any OpenAI-compatible client (the `openai` SDK pointed at a custom `base_url`, OpenWebUI,
LLM gateways, …) talk to a Pydantic AI agent as if it were an OpenAI model. The agent runs its own
tool loop server-side and is projected as a single model: only the assistant's text is returned.

The entry point is [`Agent.to_responses()`][pydantic_ai.agent.Agent.to_responses]; for per-request
control use [`handle_responses_request`][pydantic_ai._responses.handle_responses_request].
"""

from ._app import create_responses_app, handle_responses_request

__all__ = [
    'create_responses_app',
    'handle_responses_request',
]
