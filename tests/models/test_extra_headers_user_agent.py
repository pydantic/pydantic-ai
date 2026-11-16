"""A user-agent in `extra_headers` goes out in place of ours, however the caller spells it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import Model, get_user_agent
from pydantic_ai.settings import ModelSettings

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.groq import GroqProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai, anthropic or groq not installed')


@dataclass
class UserAgentLines:
    """The user-agent lines of each request, read off its raw headers so that a second line shows.

    Every request is answered with a 400, which ends the run: the claim is about the request alone.
    """

    sent: list[list[str]] = field(default_factory=list[list[str]])

    def record(self, raw_headers: list[tuple[bytes, bytes]]) -> None:
        self.sent.append([value.decode() for name, value in raw_headers if name.lower() == b'user-agent'])

    def httpx2_client(self) -> httpx2.AsyncClient:
        def answer(request: httpx2.Request) -> httpx2.Response:
            self.record(request.headers.raw)
            return httpx2.Response(400)

        return httpx2.AsyncClient(transport=httpx2.MockTransport(answer))

    def httpx_client(self) -> httpx.AsyncClient:
        """For `GroqProvider`, which takes only a legacy `httpx` client."""

        def answer(request: httpx.Request) -> httpx.Response:
            self.record(request.headers.raw)
            return httpx.Response(400)

        return httpx.AsyncClient(transport=httpx.MockTransport(answer))


@dataclass(frozen=True)
class Case:
    id: str
    model: Callable[[UserAgentLines], Model]


CASES = [
    Case(
        id='openai-chat',
        model=lambda lines: OpenAIChatModel(
            'gpt-4o', provider=OpenAIProvider(api_key='test', http_client=lines.httpx2_client())
        ),
    ),
    Case(
        id='openai-responses',
        model=lambda lines: OpenAIResponsesModel(
            'gpt-4o', provider=OpenAIProvider(api_key='test', http_client=lines.httpx2_client())
        ),
    ),
    Case(
        id='anthropic',
        model=lambda lines: AnthropicModel(
            'claude-sonnet-4-5', provider=AnthropicProvider(api_key='test', http_client=lines.httpx2_client())
        ),
    ),
    Case(
        id='groq',
        model=lambda lines: GroqModel(
            'llama-3.3-70b-versatile', provider=GroqProvider(api_key='test', http_client=lines.httpx_client())
        ),
    ),
]


CALLER_USER_AGENT = 'support-bot/1.0'


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CASES])
@pytest.mark.parametrize(
    ('settings', 'user_agent'),
    [
        pytest.param(ModelSettings(), get_user_agent(), id='no extra_headers'),
        pytest.param(
            ModelSettings(extra_headers={'User-Agent': CALLER_USER_AGENT}), CALLER_USER_AGENT, id='User-Agent'
        ),
        pytest.param(
            ModelSettings(extra_headers={'user-agent': CALLER_USER_AGENT}), CALLER_USER_AGENT, id='user-agent'
        ),
        pytest.param(
            ModelSettings(extra_headers={'USER-AGENT': CALLER_USER_AGENT}), CALLER_USER_AGENT, id='USER-AGENT'
        ),
    ],
)
async def test_extra_headers_user_agent(
    allow_model_requests: None,
    case: Case,
    settings: ModelSettings,
    user_agent: str,
):
    """The request carries one user-agent line: the caller's in any spelling, or ours when they set none.

    Not recorded: the claim is about the header lines the SDK sends, which no response bears on.
    """
    lines = UserAgentLines()

    with pytest.raises(ModelHTTPError):
        await Agent(case.model(lines)).run('hello', model_settings=settings)

    assert lines.sent == [[user_agent]]
