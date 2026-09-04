"""Tests for `GitHubCopilotModel`.

Copilot's Chat Completions envelope leaves out required OpenAI fields, and which ones depends on
the model: GPT ids omit `object` and `created`, Anthropic ids omit `object` and each choice's
`index`. The openai SDK builds responses without validating, so those arrive as `None` and blow up
in the parent's validation step — which is why a plain `OpenAIChatModel` pointed at a Copilot base
URL cannot complete a single request. Repairing that envelope is the point of this model class, and
`test_github_copilot_envelope_breaks_the_stock_openai_model` is the recording that proves it.

The other half is thinking: Copilot rejects `reasoning_effort` outright for Anthropic ids, for the
disabling value as well as the enabling ones, so the model raises rather than degrade silently and
drops a `thinking=False` that would otherwise go out as `reasoning_effort='none'`.
"""

from __future__ import annotations as _annotations

import os
import re

import pytest

from pydantic_ai import (
    Agent,
    ModelHTTPError,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UnexpectedModelBehavior,
    UserError,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsStr, RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.github_copilot import GitHubCopilotModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.github_copilot import GitHubCopilotProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_github_copilot_model_simple(allow_model_requests: None, github_copilot_api_key: str):
    """A GPT id round-trips even though its envelope carries neither `created` nor `object`.

    The recorded body's top-level keys are `choices`, `copilot_usage`, `id`, `model`, `service_tier`
    and `usage` — the two OpenAI-required fields are simply absent. `created` is filled by
    `OpenAIChatModel._process_response` with the receive time, which is why the timestamp below is
    `IsDatetime()` rather than a value from the recording. Its choices do carry `index`, which the
    Anthropic ids drop; the two together cover both sides of that repair.
    """
    model = GitHubCopilotModel('gpt-5.4', provider=GitHubCopilotProvider(api_key=github_copilot_api_key))
    agent = Agent(model, instructions='Be concise.')

    result = await agent.run('What is the capital of France?')

    assert result.output == snapshot('Paris.')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions='Be concise.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Paris.')],
                usage=RequestUsage(
                    details={'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0},
                    input_tokens=20,
                    output_tokens=5,
                ),
                model_name='gpt-5.4',
                timestamp=IsDatetime(),
                provider_name='github-copilot',
                provider_url='https://api.githubcopilot.com',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_github_copilot_envelope_breaks_the_stock_openai_model(
    allow_model_requests: None, github_copilot_api_key: str
):
    """The same request through an unmodified `OpenAIChatModel` fails, which is why this PR exists.

    Before `GitHubCopilotModel`, a Copilot subscriber's only option was `openai-chat:<id>` with a
    Copilot base URL. This is that configuration: it reaches the API, gets a 200 back, and then dies
    validating the envelope. Recorded live against Copilot rather than hand-built, so the body is the
    real one and this stops being a claim about what Copilot returns.
    """
    model = OpenAIChatModel('gpt-5.4', provider=GitHubCopilotProvider(api_key=github_copilot_api_key))
    agent = Agent(model, instructions='Be concise.')

    with pytest.raises(UnexpectedModelBehavior, match=r'Invalid response from .* chat completions endpoint'):
        await agent.run('What is the capital of France?')


async def test_github_copilot_claude_model(allow_model_requests: None, github_copilot_api_key: str):
    """Claude ids are served on the same surface but drop a different set of fields.

    They carry `created` and omit `object` — and, unlike the GPT ids, omit each choice's `index` too.
    The asymmetry is why the repair fills whichever field is missing rather than assuming one shape.
    """
    model = GitHubCopilotModel('claude-haiku-4.5', provider=GitHubCopilotProvider(api_key=github_copilot_api_key))
    agent = Agent(model, instructions='Be concise.')

    result = await agent.run('What is the capital of France?')

    assert result.output == snapshot('Paris is the capital of France.')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions='Be concise.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Paris is the capital of France.')],
                usage=RequestUsage(input_tokens=18, output_tokens=10),
                model_name='claude-haiku-4.5',
                timestamp=IsDatetime(),
                provider_name='github-copilot',
                provider_url='https://api.githubcopilot.com',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_github_copilot_model_stream(allow_model_requests: None, github_copilot_api_key: str):
    """Streamed chunks carry `created` and the standard `delta` shape, so streaming needs no repair.

    The negative case for the envelope handling above: `_process_streamed_response` is deliberately
    not overridden, and this is what says so.
    """
    model = GitHubCopilotModel('gpt-5.4', provider=GitHubCopilotProvider(api_key=github_copilot_api_key))
    agent = Agent(model, instructions='Be concise.')

    async with agent.run_stream('What is the capital of France?') as result:
        output = await result.get_output()

    assert output == snapshot('Paris.')


async def test_github_copilot_sends_model_id_verbatim(
    allow_model_requests: None, github_copilot_api_key: str, request_capture: RequestCapture
):
    """Copilot ids go out exactly as the user wrote them, dots and all.

    Consumers that namespace ids as `copilot/<id>` or rewrite dotted Claude ids to hyphenated ones do
    so because no provider owned the mapping; now that one does, it deliberately owns none of it. The
    `User-Agent` is asserted here too: Copilot accepts `pydantic-ai/<version>`, so nothing overrides
    it, and posing as a Copilot chat client would be impersonation for no benefit.
    """
    model = GitHubCopilotModel(
        'claude-haiku-4.5',
        provider=GitHubCopilotProvider(api_key=github_copilot_api_key, http_client=request_capture.client),
    )

    await Agent(model, instructions='Be concise.').run('What is the capital of France?')

    assert request_capture.body('/chat/completions')['model'] == 'claude-haiku-4.5'
    headers = request_capture.headers[0]
    assert headers['user-agent'].startswith('pydantic-ai/')
    assert headers['editor-version'] == 'vscode/1.95.0'


async def test_github_copilot_thinking_sends_reasoning_effort(
    allow_model_requests: None, github_copilot_api_key: str, request_capture: RequestCapture
):
    """The `github_copilot_supports_reasoning_effort` flag-on side: GPT ids take the parameter."""
    model = GitHubCopilotModel(
        'gpt-5.4',
        provider=GitHubCopilotProvider(api_key=github_copilot_api_key, http_client=request_capture.client),
    )

    await Agent(model, instructions='Be concise.').run(
        'What is the capital of France?', model_settings=ModelSettings(thinking='high')
    )

    assert request_capture.body('/chat/completions')['reasoning_effort'] == 'high'


@pytest.mark.parametrize('thinking', [True, 'high'])
@pytest.mark.parametrize('setting_source', ['run', 'model'])
def test_github_copilot_thinking_raises_for_claude(
    allow_model_requests: None, github_copilot_api_key: str, thinking: bool | str, setting_source: str
):
    """The flag-off side: Copilot rejects `reasoning_effort` for Anthropic ids, so we say so.

    Not a VCR test — the error is raised before any request, which is the whole point: silently
    dropping `thinking` here would return an answer with no `ThinkingPart` and no explanation.

    Both places a user can put the setting are covered, because the check reads the *resolved*
    parameters: a model-level `settings=` is invisible to a check that only inspects the argument
    passed to `run`.
    """
    settings = ModelSettings(thinking=thinking)  # pyright: ignore[reportArgumentType]
    model = GitHubCopilotModel(
        'claude-haiku-4.5',
        provider=GitHubCopilotProvider(api_key=github_copilot_api_key),
        settings=settings if setting_source == 'model' else None,
    )
    agent = Agent(model)

    with pytest.raises(
        UserError,
        match=re.escape(
            "`thinking` is not supported with `GitHubCopilotModel` and model 'claude-haiku-4.5': "
            "GitHub Copilot's chat completions API rejects `reasoning_effort` for Anthropic models."
        ),
    ):
        agent.run_sync('What is the capital of France?', model_settings=settings if setting_source == 'run' else None)


async def test_github_copilot_thinking_false_is_dropped_for_claude(
    allow_model_requests: None, github_copilot_api_key: str, request_capture: RequestCapture
):
    """`thinking=False` asks for nothing, so it is satisfied by sending nothing.

    Copilot rejects `reasoning_effort='none'` on Anthropic ids exactly as it rejects `'high'`, and
    `Model.prepare_request` resolves `False` onto the parameters because the Claude profile does
    support thinking. Forwarding it would break code that merely turns reasoning off.
    """
    model = GitHubCopilotModel(
        'claude-haiku-4.5',
        provider=GitHubCopilotProvider(api_key=github_copilot_api_key, http_client=request_capture.client),
    )

    result = await Agent(model, instructions='Be concise.').run(
        'What is the capital of France?', model_settings=ModelSettings(thinking=False)
    )

    assert 'reasoning_effort' not in request_capture.body('/chat/completions')
    assert result.output == snapshot('The capital of France is Paris.')


@pytest.mark.xfail(
    strict=True,
    reason='Blocked on https://github.com/pydantic/genai-prices/issues/681 — genai-prices has no '
    '`github-copilot` provider, so no Copilot model resolves a context window or a price. An XPASS '
    'means the entry shipped: drop this marker and pin the window in the profile snapshot.',
)
def test_github_copilot_context_window_is_known(github_copilot_api_key: str):
    """Not a VCR test: the window is filled from the bundled genai-prices snapshot, not the network."""
    model = GitHubCopilotModel('gpt-5.4', provider=GitHubCopilotProvider(api_key=github_copilot_api_key))
    assert model.profile.get('context_window') is not None


@pytest.mark.xfail(
    strict=True,
    raises=UserError,
    reason="Blocked on the Copilot `/v1/messages` transport. Copilot's Claude models think, but its "
    'Chat Completions endpoint cannot ask them to, so this raises `UserError` today. An XPASS means '
    'the Messages transport landed and the `github_copilot_supports_reasoning_effort` gate can go.',
)
async def test_github_copilot_claude_thinking(allow_model_requests: None, github_copilot_api_key: str):
    """No cassette: the `UserError` is raised before any request goes out."""
    model = GitHubCopilotModel('claude-haiku-4.5', provider=GitHubCopilotProvider(api_key=github_copilot_api_key))
    agent = Agent(model, instructions='Be concise.')

    result = await agent.run('What is 2 + 2?', model_settings=ModelSettings(thinking=True))

    assert any(isinstance(part, ThinkingPart) for part in result.new_messages()[-1].parts)


@pytest.mark.xfail(
    strict=True,
    raises=ModelHTTPError,
    reason='Blocked on a Copilot `/responses` transport. Copilot lists these ids but serves them only '
    'on the Responses API, so Chat Completions answers `unsupported_api_for_model` — the recorded '
    'body here. An XPASS means the Responses transport landed.',
)
async def test_github_copilot_responses_only_model(allow_model_requests: None, github_copilot_api_key: str):
    model = GitHubCopilotModel('gpt-5.6-luna', provider=GitHubCopilotProvider(api_key=github_copilot_api_key))
    agent = Agent(model, instructions='Be concise.')

    result = await agent.run('What is the capital of France?')

    assert result.output


@pytest.mark.xfail(
    strict=True,
    raises=ModelHTTPError,
    reason="Blocked on GitHub. A fine-grained PAT with Copilot Requests is listed by GitHub's Copilot "
    'SDK docs but returns `401 unauthorized: AuthenticateToken authentication failed` — the recorded '
    'response here. An XPASS after a re-record means GitHub widened it and the docs caveat can go.',
)
async def test_github_copilot_fine_grained_pat_authenticates(allow_model_requests: None):
    """Recorded with a real `github_pat_` credential; VCR scrubs the `authorization` header."""
    api_key = os.getenv('GITHUB_COPILOT_FINE_GRAINED_PAT', 'mock-api-key')
    model = GitHubCopilotModel('claude-haiku-4.5', provider=GitHubCopilotProvider(api_key=api_key))
    agent = Agent(model, instructions='Be concise.')

    result = await agent.run('What is the capital of France?')

    assert result.output
