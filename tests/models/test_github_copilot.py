"""Tests for `GitHubCopilotModel`.

Copilot's Chat Completions envelope leaves out required OpenAI fields, and which ones depends on
the model: GPT ids omit `object` and `created`, Anthropic ids omit `object` and each choice's
`index`. The openai SDK builds responses without validating, so those arrive as `None` and blow up
in the parent's validation step — which is why a plain `OpenAIChatModel` pointed at a Copilot base
URL cannot complete a single request. Repairing that envelope is the point of this model class, and
`test_github_copilot_envelope_breaks_the_stock_openai_model` is the recording that proves it.

The other half is thinking. Copilot returns an Anthropic id's reasoning in `reasoning_text`, a field
name `OpenAIChatModel` does not know on its own, so the provider profile points
`openai_chat_thinking_field` at it and the ordinary Chat Completions machinery does the rest — mapping
it to a `ThinkingPart` and sending it back on later turns. Nothing in this model gates `thinking`:
Copilot itself answers `400 invalid_reasoning_effort` for an id whose catalog entry has no
`reasoning_effort`, and for the `'none'` that `thinking=False` maps to, which its Claude ids do not
offer because they reason adaptively.
"""

from __future__ import annotations as _annotations

import json
import os
from dataclasses import dataclass, field

import httpx2
import pytest

from pydantic_ai import (
    Agent,
    ModelHTTPError,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UnexpectedModelBehavior,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsStr, RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.github_copilot import GitHubCopilotModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
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


async def test_github_copilot_tool_call_round_trip(
    allow_model_requests: None, github_copilot_api_key: str, request_capture: RequestCapture
):
    """Copilot accepts `strict` tool definitions, which is why the profile leaves the default alone.

    Every sibling gateway answers this question explicitly — `SnowflakeProvider` and `OllamaProvider`
    both set `openai_supports_strict_tool_definition=False` with a cited reason — so the Copilot
    overlay leaving it at `True` is a claim about the API, and this is the recording that backs it.
    A Claude id runs it because those omit each choice's `index`: the envelope repair has to hold on
    a `tool_calls` choice too, not only on the plain text one above.
    """
    model = GitHubCopilotModel(
        'claude-haiku-4.5',
        provider=GitHubCopilotProvider(api_key=github_copilot_api_key, http_client=request_capture.client),
    )
    agent = Agent(model, instructions='Be concise.')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the weather in a city."""
        return 'sunny'

    result = await agent.run('What is the weather in Paris?')

    assert request_capture.body('/chat/completions')['tools'] == snapshot(
        [
            {
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'description': 'Get the weather in a city.',
                    'parameters': {
                        'additionalProperties': False,
                        'properties': {'city': {'type': 'string'}},
                        'required': ['city'],
                        'type': 'object',
                    },
                    'strict': True,
                },
            }
        ]
    )
    assert result.output == snapshot('The weather in Paris is currently **sunny**.')


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


async def test_github_copilot_claude_stream_tool_call(allow_model_requests: None, github_copilot_api_key: str):
    """The streamed twin of the repair, on the family that omits the most.

    `test_github_copilot_model_stream` records a GPT id, and the non-streamed path proves the two
    families drop different fields — so "streaming needs no repair" cannot be generalized from the
    GPT recording alone. This is the Claude recording that closes that gap, and it streams a tool
    call because `_map_tool_call_delta` keys parts on each delta's `index`: were Copilot to omit it
    the way it omits a choice's `index` off-stream, nothing would raise and parallel tool calls
    would merge their argument fragments instead.
    """
    model = GitHubCopilotModel('claude-haiku-4.5', provider=GitHubCopilotProvider(api_key=github_copilot_api_key))
    agent = Agent(model, instructions='Be concise.')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the weather in a city."""
        return 'sunny'

    async with agent.run_stream('What is the weather in Paris?') as result:
        output = await result.get_output()

    assert output == snapshot('The weather in Paris is currently **sunny**.')


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
    """`thinking` reaches the wire as `reasoning_effort`, unchanged by this model class.

    `gpt-5.4` is the id whose catalog entry lists `reasoning_effort` and which returns no reasoning
    field at all, so it pins the forwarding on its own, separately from the Claude ids below.
    """
    model = GitHubCopilotModel(
        'gpt-5.4',
        provider=GitHubCopilotProvider(api_key=github_copilot_api_key, http_client=request_capture.client),
    )

    await Agent(model, instructions='Be concise.').run(
        'What is the capital of France?', model_settings=ModelSettings(thinking='high')
    )

    assert request_capture.body('/chat/completions')['reasoning_effort'] == 'high'


async def test_github_copilot_flagged_claude_drops_sampling_settings_on_the_wire(
    allow_model_requests: None, github_copilot_api_key: str, request_capture: RequestCapture
):
    """A flagged Claude id drops `temperature` and `top_p` before the request, as `AnthropicModel` does.

    `test_github_copilot_provider_sampling_restriction_follows_the_family` pins the profile tuple;
    this pins the joint — the family flag reaching the shared drop on a real request — and that
    Copilot answers `200` to the stripped body. `claude-sonnet-5` carries the flag, which the
    unflagged `claude-haiku-4.5` used by the tests above does not.
    """
    model = GitHubCopilotModel(
        'claude-sonnet-5',
        provider=GitHubCopilotProvider(api_key=github_copilot_api_key, http_client=request_capture.client),
    )

    result = await Agent(model, instructions='Be concise.').run(
        'What is the capital of France?', model_settings=ModelSettings(temperature=0.2, top_p=0.9)
    )

    body = request_capture.body('/chat/completions')
    assert 'temperature' not in body
    assert 'top_p' not in body
    assert result.output == snapshot('Paris')


async def test_github_copilot_sends_max_completion_tokens(
    allow_model_requests: None, github_copilot_api_key: str, request_capture: RequestCapture
):
    """The `max_tokens` setting goes out as `max_completion_tokens`, and Copilot answers `200`.

    `test_github_copilot_provider_gpt_profile` pins the flag on this id at profile level; this pins
    the joint — the flag reaching the field substitution on a real request, and the substituted body
    being the one Copilot accepts. `test_github_copilot_rejects_max_tokens` records the other side.
    """
    model = GitHubCopilotModel(
        'gpt-5.4',
        provider=GitHubCopilotProvider(api_key=github_copilot_api_key, http_client=request_capture.client),
    )

    result = await Agent(model, instructions='Be concise.').run(
        'What is the capital of France?', model_settings=ModelSettings(max_tokens=123)
    )

    body = request_capture.body('/chat/completions')
    assert body['max_completion_tokens'] == 123
    assert 'max_tokens' not in body
    assert result.output == snapshot('Paris.')


async def test_github_copilot_rejects_max_tokens(
    allow_model_requests: None, github_copilot_api_key: str, request_capture: RequestCapture
):
    """Copilot answers a bare `400 Bad Request` to the stock `max_tokens` field.

    This is why the overlay pins `openai_chat_supports_max_completion_tokens`, and it is the
    flag-off side of that branch: a partial `profile=` merges on top of the provider's, so the
    request goes out the way an unpinned Copilot model would send it. The 400 carries a plain-text
    body rather than Copilot's usual `{'message', 'code'}` JSON, so there is no field name in it to
    match on — the status and the absence of a response are the whole signal a user gets.
    """
    model = GitHubCopilotModel(
        'gpt-5.4',
        provider=GitHubCopilotProvider(api_key=github_copilot_api_key, http_client=request_capture.client),
        profile=OpenAIModelProfile(openai_chat_supports_max_completion_tokens=False),
    )
    assert model.profile.get('openai_chat_supports_max_completion_tokens') is False

    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model, instructions='Be concise.').run(
            'What is the capital of France?', model_settings=ModelSettings(max_tokens=123)
        )

    body = request_capture.body('/chat/completions')
    assert body['max_tokens'] == 123
    assert 'max_completion_tokens' not in body
    assert exc_info.value.status_code == 400
    assert exc_info.value.body == snapshot('Bad Request')


@pytest.mark.xfail(
    strict=True,
    reason='Blocked on a genai-prices release carrying the `github-copilot` provider added in '
    'https://github.com/pydantic/genai-prices/pull/683; the bundled snapshot has none, so no Copilot '
    'model resolves a context window or a price. An XPASS means the release landed: drop this marker '
    'and pin the window in the profile snapshot.',
)
def test_github_copilot_context_window_is_known(github_copilot_api_key: str):
    """Not a VCR test: the window is filled from the bundled genai-prices snapshot, not the network."""
    model = GitHubCopilotModel('gpt-5.4', provider=GitHubCopilotProvider(api_key=github_copilot_api_key))
    assert model.profile.get('context_window') is not None


async def test_github_copilot_claude_thinking(
    allow_model_requests: None, github_copilot_api_key: str, request_capture: RequestCapture
):
    """Copilot's Claude ids reason on Chat Completions, and the reasoning reaches the user.

    It arrives in `reasoning_text`, which is neither of the two field names `OpenAIChatModel` falls
    back to, so the `ThinkingPart` below exists only because the provider profile names that field.
    Copilot returns a `reasoning_opaque` signature alongside it that Pydantic AI deliberately does
    not carry; `test_github_copilot_claude_thinking_is_sent_back` is what says Copilot accepts a
    later turn without it.

    The prompt is chosen, not incidental. These ids reason *adaptively*: the effort is a ceiling, not
    an instruction, and the model answers an easy question without reasoning at any effort. Probed
    live on 2026-09-07, this prompt at `thinking=True` returned reasoning on 4 of 4 attempts while
    `'Is 221 prime?'` returned none on 4 of 4 — so a re-record needs a question worth thinking about,
    not a higher effort.
    """
    model = GitHubCopilotModel(
        'claude-sonnet-5',
        provider=GitHubCopilotProvider(api_key=github_copilot_api_key, http_client=request_capture.client),
    )
    agent = Agent(model, instructions='Be concise.')

    result = await agent.run(
        'Factor 3599 into two primes. Show only the answer.', model_settings=ModelSettings(thinking=True)
    )

    assert request_capture.body('/chat/completions')['reasoning_effort'] == 'medium'
    assert result.all_messages()[-1] == snapshot(
        ModelResponse(
            parts=[
                ThinkingPart(
                    content="""\
3599 factors as 59 times 61.

""",
                    id='reasoning_text',
                    provider_name='github-copilot',
                ),
                TextPart(content='3599 = 59 × 61'),
            ],
            usage=RequestUsage(input_tokens=31, output_tokens=24),
            model_name='claude-sonnet-5',
            timestamp=IsDatetime(),
            provider_name='github-copilot',
            provider_url='https://api.githubcopilot.com',
            provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
            provider_response_id=IsStr(),
            finish_reason='stop',
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


async def test_github_copilot_claude_thinking_stream(allow_model_requests: None, github_copilot_api_key: str):
    """The streamed twin: `reasoning_text` arrives as deltas and accumulates into a `ThinkingPart`.

    `_map_thinking_delta` reads the same profile field as the non-streamed path, so a profile that
    covered only one of the two would leave streaming users with the reasoning silently dropped.
    """
    model = GitHubCopilotModel('claude-sonnet-5', provider=GitHubCopilotProvider(api_key=github_copilot_api_key))
    agent = Agent(model, instructions='Be concise.')

    async with agent.run_stream(
        'Factor 3599 into two primes. Show only the answer.', model_settings=ModelSettings(thinking=True)
    ) as result:
        output = await result.get_output()

    assert output == snapshot('**3599 = 59 × 61**')
    assert result.all_messages()[-1] == snapshot(
        ModelResponse(
            parts=[
                ThinkingPart(
                    content="""\
3599 factors as a difference of squares: 60²-1² = 59×61.

""",
                    id='reasoning_text',
                    provider_name='github-copilot',
                ),
                TextPart(content='**3599 = 59 × 61**'),
            ],
            usage=RequestUsage(output_tokens=41, input_tokens=31),
            model_name='claude-sonnet-5',
            timestamp=IsDatetime(),
            provider_name='github-copilot',
            provider_url='https://api.githubcopilot.com',
            provider_details={'timestamp': IsDatetime(), 'finish_reason': 'stop'},
            provider_response_id=IsStr(),
            finish_reason='stop',
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


async def test_github_copilot_claude_thinking_is_sent_back(
    allow_model_requests: None, github_copilot_api_key: str, request_capture: RequestCapture
):
    """A second turn echoes the reasoning back in `reasoning_text`, and Copilot accepts it.

    `openai_chat_send_back_thinking_parts` is left at its `'auto'` default, which sends a
    `ThinkingPart` back in the field it came from when its `id` matches the profile's field name.
    Copilot does not require the echo — probed live on 2026-09-07 it answered `200` to tool round
    trips that omitted it, and to ones carrying only the `reasoning_opaque` signature we drop — so
    the profile does not force `'field'` mode.
    """
    model = GitHubCopilotModel(
        'claude-sonnet-5',
        provider=GitHubCopilotProvider(api_key=github_copilot_api_key, http_client=request_capture.client),
    )
    agent = Agent(model, instructions='Be concise.')
    settings = ModelSettings(thinking=True)

    first = await agent.run('Factor 3599 into two primes. Show only the answer.', model_settings=settings)
    await agent.run('Now factor 5183 the same way.', message_history=first.all_messages(), model_settings=settings)

    assert request_capture.bodies('/chat/completions')[1]['messages'] == snapshot(
        [
            {'role': 'system', 'content': 'Be concise.'},
            {'role': 'user', 'content': 'Factor 3599 into two primes. Show only the answer.'},
            {
                'role': 'assistant',
                'reasoning_text': """\
3599 factors as 59 times 61.

""",
                'content': '3599 = 59 × 61',
            },
            {'role': 'user', 'content': 'Now factor 5183 the same way.'},
        ]
    )


async def test_github_copilot_gemini_thinking(
    allow_model_requests: None, github_copilot_api_key: str, request_capture: RequestCapture
):
    """Copilot's Gemini ids return reasoning in the same `reasoning_text` field the Claude ids use.

    The second family on that field, and the reason the profile keys it on two prefixes rather than
    on `claude-`. Easy to miss: `GET /models` only lists the Gemini ids when the request carries the
    `copilot-integration-id` header the provider sends, so a bare listing suggests they aren't served.

    The reasoning text itself is matched loosely: Gemini's is several paragraphs and would churn the
    snapshot on every re-record, while what this test is about is the part existing at all with the
    `id` the profile names. `usage.details` carries the reasoning tokens Copilot billed for it.
    """
    model = GitHubCopilotModel(
        'gemini-3.8-flash',
        provider=GitHubCopilotProvider(api_key=github_copilot_api_key, http_client=request_capture.client),
    )
    agent = Agent(model, instructions='Be concise.')

    result = await agent.run(
        'Factor 3599 into two primes. Show only the answer.', model_settings=ModelSettings(thinking=True)
    )

    assert request_capture.body('/chat/completions')['reasoning_effort'] == 'medium'
    assert result.all_messages()[-1] == snapshot(
        ModelResponse(
            parts=[
                ThinkingPart(
                    content=IsStr(),
                    id='reasoning_text',
                    provider_name='github-copilot',
                ),
                TextPart(content='59 × 61'),
            ],
            usage=RequestUsage(details={'reasoning_tokens': 122}, input_tokens=18, output_tokens=6),
            model_name='gemini-3.8-flash',
            timestamp=IsDatetime(),
            provider_name='github-copilot',
            provider_url='https://api.githubcopilot.com',
            provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
            provider_response_id=IsStr(),
            finish_reason='stop',
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


async def test_github_copilot_gemini_thinking_stream(allow_model_requests: None, github_copilot_api_key: str):
    """The Gemini streamed twin: `reasoning_text` reaches a `ThinkingPart` on the streamed path too.

    `test_github_copilot_claude_thinking_stream` covers the same path on the other family that
    reasons in this field; both are pinned because the profile keys `openai_chat_thinking_field` on
    two prefixes, and a profile covering only one would drop the reasoning for the other's users.

    The prompt asks for one step more than the Claude twin's because Gemini reasons adaptively: at
    the `medium` effort `thinking=True` maps to, the bare factoring prompt sometimes streams back
    `content` alone, with no `reasoning_text` for the part to come from.

    The reasoning text itself is matched loosely, as the non-streamed Gemini test matches it: its
    wording and length change on every re-record, and what this pins is the part arriving at all
    with the `id` the profile names.
    """
    model = GitHubCopilotModel('gemini-3.8-flash', provider=GitHubCopilotProvider(api_key=github_copilot_api_key))
    agent = Agent(model, instructions='Be concise.')

    async with agent.run_stream(
        'Factor 3599 into two primes, then show only their sum.', model_settings=ModelSettings(thinking=True)
    ) as result:
        output = await result.get_output()

    assert output == snapshot('120')
    assert result.all_messages()[-1] == snapshot(
        ModelResponse(
            parts=[
                ThinkingPart(
                    content=IsStr(),
                    id='reasoning_text',
                    provider_name='github-copilot',
                ),
                TextPart(content='120'),
            ],
            usage=RequestUsage(details={'reasoning_tokens': 199}, input_tokens=19, output_tokens=3),
            model_name='gemini-3.8-flash',
            timestamp=IsDatetime(),
            provider_name='github-copilot',
            provider_url='https://api.githubcopilot.com',
            provider_details={'timestamp': IsDatetime(), 'finish_reason': 'stop'},
            provider_response_id=IsStr(),
            finish_reason='stop',
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


async def test_github_copilot_claude_thinking_false_is_rejected(
    allow_model_requests: None, github_copilot_api_key: str
):
    """`thinking=False` maps to `reasoning_effort='none'`, which Copilot's Claude ids do not offer.

    They reason adaptively and expose no off switch — the catalog lists
    `[low medium high xhigh max]` and no `none` — so Copilot answers `400`. Surfacing that beats
    dropping the setting: a user who asked to turn reasoning off would otherwise be billed for
    reasoning they believed they had disabled.
    """
    model = GitHubCopilotModel('claude-sonnet-5', provider=GitHubCopilotProvider(api_key=github_copilot_api_key))
    agent = Agent(model, instructions='Be concise.')

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Is 221 prime?', model_settings=ModelSettings(thinking=False))

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == snapshot(
        {
            'message': 'reasoning_effort "none" is not supported by model claude-sonnet-5; supported values: [low medium high xhigh max]',
            'code': 'invalid_reasoning_effort',
        }
    )


async def test_github_copilot_claude_without_reasoning_effort_support_is_rejected(
    allow_model_requests: None, github_copilot_api_key: str
):
    """Not every Claude id Copilot serves takes `reasoning_effort`, and Copilot says which.

    `claude-haiku-4.5`'s catalog entry carries no `reasoning_effort` key at all, so the parameter is
    rejected outright. That is a per-id fact Copilot owns and reports. A client-side gate keyed on
    the `claude-` prefix would have to guess it and would guess wrong, which is what
    `test_github_copilot_claude_thinking` — the same parameter accepted on `claude-sonnet-5` — shows.
    """
    model = GitHubCopilotModel('claude-haiku-4.5', provider=GitHubCopilotProvider(api_key=github_copilot_api_key))
    agent = Agent(model, instructions='Be concise.')

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Is 221 prime?', model_settings=ModelSettings(thinking=True))

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == snapshot(
        {
            'message': 'reasoning_effort "medium" was provided, but model claude-haiku-4.5 does not support reasoning effort',
            'code': 'invalid_reasoning_effort',
        }
    )


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


_PROXY_RESPONSE_ID = 'msg_proxy_stream_01'
_PROXY_TOOL_CALL_ID = 'toolu_proxy_stream_01'


def _proxy_chunk(delta: dict[str, object], finish_reason: str | None = None) -> dict[str, object]:
    """One Copilot streamed chunk, modelled on the shapes in this file's live cassettes.

    Copilot omits `object` on every chunk, which is why it is absent here too.
    """
    choice: dict[str, object] = {'index': 0, 'delta': delta}
    if finish_reason is not None:
        choice['finish_reason'] = finish_reason
    return {'choices': [choice], 'created': 1788538002, 'id': _PROXY_RESPONSE_ID, 'model': 'claude-haiku-4.5'}


def _proxy_sse(*chunks: dict[str, object]) -> bytes:
    return ''.join(f'data: {json.dumps(chunk)}\n\n' for chunk in chunks).encode() + b'data: [DONE]\n\n'


# The tool call arrives as an opening delta carrying `id`/`type`/`function.name` and then bare
# `function.arguments` fragments, each keyed by the same `index` — the shape
# `test_github_copilot_claude_stream_tool_call` recorded against Copilot.
_PROXY_TOOL_CALL_STREAM = _proxy_sse(
    _proxy_chunk(
        {
            'content': None,
            'tool_calls': [
                {'function': {'name': 'get_weather'}, 'id': _PROXY_TOOL_CALL_ID, 'index': 0, 'type': 'function'}
            ],
        }
    ),
    _proxy_chunk({'content': None, 'tool_calls': [{'function': {'arguments': '{"city": '}, 'index': 0}]}),
    _proxy_chunk({'content': None, 'tool_calls': [{'function': {'arguments': '"Paris"}'}, 'index': 0}]}),
    _proxy_chunk({'content': None}, finish_reason='tool_calls'),
)

_PROXY_TEXT_STREAM = _proxy_sse(
    _proxy_chunk({'content': 'The weather in Paris is sunny.'}),
    _proxy_chunk({'content': None}, finish_reason='stop'),
)


@dataclass
class _CopilotProxy:
    """A stand-in for a Copilot-compatible proxy: records what reached it, replays two streams."""

    requests: list[httpx2.Request] = field(default_factory=list[httpx2.Request])
    bodies: list[dict[str, object]] = field(default_factory=list[dict[str, object]])

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content))
        stream = _PROXY_TOOL_CALL_STREAM if len(self.requests) == 1 else _PROXY_TEXT_STREAM
        return httpx2.Response(200, content=stream, headers={'content-type': 'text/event-stream'})


@pytest.fixture
def copilot_proxy() -> _CopilotProxy:
    return _CopilotProxy()


@pytest.mark.vcr(ignore_hosts=['copilot-proxy.example'])
async def test_github_copilot_streams_a_tool_call_round_trip_through_a_proxy(
    allow_model_requests: None, copilot_proxy: _CopilotProxy
):
    """A custom base URL, a placeholder bearer, and a fully streamed tool-call round trip.

    This is the shape a downstream engine such as gh-aw drives: `GitHubCopilotProvider` pointed at a
    proxy that swaps the token out, so the credential Pydantic AI holds is a placeholder that must
    never reach a network. What it adds over its two neighbours — `test_github_copilot_provider_base_url_argument`
    for the URL and `test_github_copilot_claude_stream_tool_call` for the recorded streamed shapes —
    is the three together across one transport: where the request lands, which bearer rides with it,
    and a tool call reassembled from split argument fragments and answered on a second streamed
    request. The proxy here is a `MockTransport` stand-in; nothing is asserted about a real one.
    """
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(copilot_proxy.handle)) as http_client:
        provider = GitHubCopilotProvider(
            base_url='https://copilot-proxy.example/api',
            api_key='placeholder-token',
            http_client=http_client,
        )
        agent = Agent(GitHubCopilotModel('claude-haiku-4.5', provider=provider), instructions='Be concise.')
        cities: list[str] = []

        @agent.tool_plain
        def get_weather(city: str) -> str:
            """Get the weather in a city."""
            cities.append(city)
            return 'sunny'

        async with agent.run_stream('What is the weather in Paris?') as result:
            output = await result.get_output()

    assert output == snapshot('The weather in Paris is sunny.')
    assert cities == ['Paris']
    assert [str(request.url) for request in copilot_proxy.requests] == snapshot(
        ['https://copilot-proxy.example/api/chat/completions', 'https://copilot-proxy.example/api/chat/completions']
    )
    assert [request.headers['authorization'] for request in copilot_proxy.requests] == snapshot(
        ['Bearer placeholder-token', 'Bearer placeholder-token']
    )
    assert [body['model'] for body in copilot_proxy.bodies] == snapshot(['claude-haiku-4.5', 'claude-haiku-4.5'])
    assert copilot_proxy.bodies[1]['messages'] == snapshot(
        [
            {'role': 'system', 'content': 'Be concise.'},
            {'role': 'user', 'content': 'What is the weather in Paris?'},
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': 'toolu_proxy_stream_01',
                        'type': 'function',
                        'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'},
                    }
                ],
            },
            {'role': 'tool', 'tool_call_id': 'toolu_proxy_stream_01', 'content': 'sunny'},
        ]
    )
