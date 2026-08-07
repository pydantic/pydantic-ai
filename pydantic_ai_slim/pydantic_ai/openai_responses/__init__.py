"""Serve a Pydantic AI agent as an [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) endpoint.

This lets any OpenAI-compatible client (the `openai` SDK pointed at a custom `base_url`, OpenWebUI,
LLM gateways, …) talk to a Pydantic AI agent as if it were an OpenAI model. The agent runs its own
tool loop server-side and is projected as a single model: assistant text is returned.

The entry point is [`Agent.to_openai_responses()`][pydantic_ai.agent.AbstractAgent.to_openai_responses]; for per-request
control use [`handle_openai_responses_request`][pydantic_ai.openai_responses.handle_openai_responses_request].
"""

from ._app import handle_openai_responses_request

__all__ = [
    'handle_openai_responses_request',
]
