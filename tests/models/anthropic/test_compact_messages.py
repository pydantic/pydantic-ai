"""Tests for `AnthropicModel.compact_messages`, Anthropic's on-demand compaction.

The recorded tests compact a short synthetic conversation of project decisions on `claude-haiku-5-5`, pin the
compaction request as the model built it (via `request_capture`, which also sees the beta header the cassettes
strip), and send the returned block back on later requests to the same model and to `claude-sonnet-4-6`.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from pydantic_ai import (
    Agent,
    ModelRequest,
    ModelResponse,
    RequestUsage,
    TextPart,
    ToolCallPart,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai._context_size import context_tokens
from pydantic_ai.messages import CompactionPart, ModelMessage
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.tools import ToolDefinition

from ..._inline_snapshot import snapshot
from ...conftest import IsStr, RequestCapture, try_import
from ..conftest import message_shape

with try_import() as imports_successful:
    from anthropic import (
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

if TYPE_CHECKING:
    from pydantic_ai.providers.anthropic import AsyncAnthropicClient

    from ..conftest import AnthropicModelFactory

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.vcr,
]

_DECISIONS = [
    ('DB-1', 'Use PostgreSQL 17 as the primary database', 'the team already runs it in two other services'),
    ('FE-1', 'Use React 19 for the web client', 'server components cut the bundle size for the dashboard'),
    ('DB-2', 'Put pgbouncer in front of PostgreSQL', 'the worker fleet opens too many short-lived connections'),
    ('OPS-1', 'Deploy on Fly.io in two regions', 'latency for European customers was over 300 ms'),
    ('DB-3', 'Take nightly base backups to S3 with 30-day retention', 'compliance asked for a month of history'),
    ('FE-2', 'Use Tailwind CSS for styling', 'the design system tokens map onto its theme config'),
    ('OPS-2', 'Ship logs and traces to Logfire', 'one place to debug both the API and the workers'),
    ('DB-4', 'Run migrations with Alembic in a pre-deploy step', 'a failed migration must block the release'),
]


def _conversation() -> list[ModelMessage]:
    """A synthetic conversation recording project decisions, ending with a request that is still open."""
    messages: list[ModelMessage] = []
    for decision_id, decision, reason in _DECISIONS:
        messages.append(
            ModelRequest.user_text_prompt(
                f'Record project decision {decision_id}: {decision}. The reason: {reason}. '
                'Acknowledge it in one sentence.',
                instructions='You keep track of the decisions for a web application project.',
            )
        )
        messages.append(
            ModelResponse(parts=[TextPart(f'Recorded {decision_id}: {decision}.')], provider_name='anthropic')
        )
    messages.append(
        ModelRequest.user_text_prompt(
            'Next, help me pick a job queue for the workers.',
            instructions='You keep track of the decisions for a web application project.',
        )
    )
    return messages


def _context(
    model: AnthropicModel,
    messages: list[ModelMessage],
    model_settings: AnthropicModelSettings | None = None,
    model_request_parameters: ModelRequestParameters | None = None,
) -> ModelRequestContext:
    return ModelRequestContext(
        model=model,
        messages=messages,
        model_settings=model_settings,
        model_request_parameters=model_request_parameters or ModelRequestParameters(),
    )


def _betas(request_capture: RequestCapture, index: int) -> list[str]:
    return sorted(request_capture.headers[index].get('anthropic-beta', '').split(','))


async def test_compact_messages(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
):
    """The response holds the signed summary, and its usage is the compaction iteration's.

    Without `max_tokens`, the model's maximum output is sent, so the SDK has the request streamed behind the scenes
    and the block arrives in a `content_block_start` event. Anthropic reports the usage of a compaction request in
    `usage.iterations` only, with the top-level counts at zero.
    """
    model = anthropic_model('claude-haiku-5-5', capture=True)

    response = await model.compact_messages(_context(model, _conversation()))

    assert response.parts == snapshot(
        [
            CompactionPart(
                content=IsStr(regex=r'(?s).*PostgreSQL.*Tailwind.*'),
                provider_name='anthropic',
                provider_details={'signature': IsStr()},
            )
        ]
    )
    assert response.finish_reason == 'stop'
    assert response.provider_details == snapshot({'finish_reason': 'compaction'})
    assert response.usage == snapshot(
        RequestUsage(
            details={
                'input_tokens': 0,
                'output_tokens': 0,
                'cache_creation_input_tokens': 0,
                'cache_read_input_tokens': 0,
                'message_iterations': 0,
                'compaction_iterations': 1,
                'compaction_input_tokens': 790,
                'compaction_output_tokens': 859,
            },
            input_tokens=790,
            output_tokens=859,
        )
    )

    # The usage is the summarized conversation, but the context the response leaves behind is the summary alone,
    # so a size trigger that reads it doesn't fire again right after compacting.
    assert context_tokens(response) == response.usage.output_tokens

    body = request_capture.body('/v1/messages')
    assert body['compaction'] == snapshot({'type': 'summarize'})
    assert 'context_management' not in body
    assert _betas(request_capture, 0) == snapshot(['compact-2026-09-04'])


async def test_compact_messages_with_focus(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
):
    """A focus is sent inside a full summarization prompt, since Anthropic's own prompt is replaced entirely.

    The effort is one of the settings a compaction request keeps, as it would be sent on any other request.
    """
    model = anthropic_model('claude-haiku-5-5', capture=True)

    response = await model.compact_messages(
        _context(model, _conversation(), AnthropicModelSettings(max_tokens=4096, anthropic_effort='low')),
        instructions='Only the database decisions; leave out everything else.',
    )

    compaction = response.parts[0]
    assert isinstance(compaction, CompactionPart)
    assert compaction.content is not None
    assert 'PostgreSQL' in compaction.content
    assert 'Tailwind' not in compaction.content
    # Without streaming, the usage comes from the compaction iteration too.
    assert response.usage.input_tokens == response.usage.details['compaction_input_tokens'] > 0
    assert response.usage.output_tokens == response.usage.details['compaction_output_tokens'] > 0

    body = request_capture.body('/v1/messages')
    assert body['output_config'] == snapshot({'effort': 'low'})
    assert body['compaction'] == snapshot(
        {
            'type': 'summarize',
            'instructions': """\
Write a summary of the conversation so far. It replaces the conversation: whoever continues it sees only your summary, so anything you leave out is lost.

Keep:
- the user's goals, requests and constraints, and how they changed;
- the decisions made and why, with exact names, identifiers, values, file paths and commands;
- the tool results that later work depends on;
- what is done, what is in progress and what comes next, including any request still waiting for an answer.

Write it as notes for whoever continues the conversation. Don't answer the latest request, and don't add anything the conversation doesn't contain.

The summary should focus on the following. Where it conflicts with the guidance above, follow it:
<focus>
Only the database decisions; leave out everything else.
</focus>\
""",
        }
    )


@pytest.mark.moves_cache_prefix(reason='replaces the conversation with its compaction block on purpose')
@pytest.mark.parametrize('replay_model_name', ['claude-haiku-5-5', 'claude-sonnet-4-6'])
async def test_compact_messages_replay(
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
    replay_model_name: str,
):
    """A later request sends the block back in place of the conversation, with the beta it needs.

    It's billed for the summary rather than the conversation, on the model that compacted and on another
    compaction-capable one. The block isn't combined with the threshold compaction edit, which Anthropic rejects
    alongside a signed block.
    """
    model = anthropic_model('claude-haiku-5-5', capture=True)
    conversation = _conversation()
    compacted = await model.compact_messages(_context(model, conversation, AnthropicModelSettings(max_tokens=4096)))

    agent = Agent(
        anthropic_model(replay_model_name, capture=True), model_settings=AnthropicModelSettings(max_tokens=512)
    )
    result = await agent.run(
        'Which database did we pick, and what sits in front of it? Answer in one sentence.',
        message_history=[*conversation, compacted],
    )

    assert 'pgbouncer' in result.output.lower()
    body = request_capture.body('/v1/messages', index=1)
    assert message_shape(body) == snapshot([('assistant', ['compaction']), ('user', ['text'])])
    assert 'context_management' not in body
    assert 'compact-2026-09-04' in _betas(request_capture, 1)
    assert result.usage.input_tokens < compacted.usage.input_tokens


@pytest.mark.moves_cache_prefix(reason='replaces the conversation with its compaction block on purpose')
@pytest.mark.parametrize(
    ('transport', 'model_name'),
    [('bedrock', 'us.anthropic.claude-haiku-5-5'), ('mantle', 'anthropic.claude-opus-5-5')],
)
async def test_compact_messages_on_aws(
    allow_model_requests: None,
    request_capture: RequestCapture,
    request: pytest.FixtureRequest,
    transport: str,
    model_name: str,
):
    """Both Amazon Bedrock APIs compact, and accept the block back, although Anthropic's docs don't list Bedrock.

    The legacy InvokeModel API carries the beta in the body's `anthropic_beta`, the Bedrock Messages API
    (`AsyncAnthropicBedrockMantle`) in the `anthropic-beta` header, on the compaction request and on the request that
    sends the block back. The Messages API serves neither Claude Haiku 5.5 nor Sonnet 4.6, so it's recorded with
    Claude Opus 5.5.
    """
    client = request.getfixturevalue(f'anthropic_{"bedrock" if transport == "bedrock" else "bedrock_mantle"}_client')
    model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client))
    conversation = _conversation()

    compacted = await model.compact_messages(_context(model, conversation, AnthropicModelSettings(max_tokens=4096)))
    result = await Agent(model, model_settings=AnthropicModelSettings(max_tokens=512)).run(
        'Which database did we pick, and what sits in front of it? Answer in one sentence.',
        message_history=[*conversation, compacted],
    )

    assert compacted.parts == snapshot(
        [
            CompactionPart(
                content=IsStr(regex=r'(?s).*PostgreSQL.*'),
                provider_name='anthropic',
                provider_details={'signature': IsStr()},
            )
        ]
    )
    assert 'pgbouncer' in result.output.lower()
    compaction_body, replay_body = request_capture.bodies()
    assert compaction_body['compaction'] == snapshot({'type': 'summarize'})
    assert message_shape(replay_body) == snapshot([('assistant', ['compaction']), ('user', ['text'])])
    for index, body in enumerate((compaction_body, replay_body)):
        if transport == 'bedrock':
            assert body['anthropic_beta'] == snapshot(['compact-2026-09-04'])
        else:
            assert _betas(request_capture, index) == snapshot(['compact-2026-09-04'])


@pytest.mark.parametrize(
    'client',
    [
        pytest.param(lambda: AsyncAnthropic(api_key='x'), id='anthropic'),
        pytest.param(lambda: AsyncAnthropicBedrock(api_key='x', aws_region='us-east-1'), id='bedrock'),
        pytest.param(lambda: AsyncAnthropicBedrockMantle(api_key='x', aws_region='us-east-1'), id='mantle'),
        pytest.param(lambda: AsyncAnthropicVertex(project_id='p', region='global', access_token='x'), id='vertex'),
        pytest.param(lambda: AsyncAnthropicFoundry(api_key='x', resource='r'), id='foundry'),
    ],
)
def test_compact_messages_supported_on_every_client(client: Callable[[], AsyncAnthropicClient]):
    """No client narrows the flag, as none has been seen to reject on-demand compaction.

    The direct API and both Bedrock clients are recorded above. Vertex AI and Microsoft Foundry are documented as
    supported, but not recorded: our credentials reach no Claude model there.
    """
    model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(anthropic_client=client()))

    assert model.profile.get('anthropic_supports_on_demand_compaction') is True


async def test_compact_messages_settles_rejected_parameters(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
):
    """What Anthropic rejects on a compaction request is left out, and a forcing `tool_choice` is relaxed.

    The trailing tool call has no result yet, which Anthropic rejects too, so it's dropped along with
    `stop_sequences`, structured output's `output_config.format`, the `remaining` task budget and
    `anthropic_context_management`. The rest of the request stays as it would otherwise be sent.
    """
    model = anthropic_model('claude-haiku-5-5', capture=True)
    messages = [
        *_conversation(),
        ModelResponse(
            parts=[
                TextPart('Let me look up the queue options.'),
                ToolCallPart('list_job_queues', {}, tool_call_id='toolu_01'),
            ],
            provider_name='anthropic',
        ),
    ]
    settings = AnthropicModelSettings(
        max_tokens=4096,
        stop_sequences=['END'],
        tool_choice='required',
        anthropic_task_budget={'type': 'tokens', 'total': 100_000, 'remaining': 50_000},
        anthropic_context_management={'edits': [{'type': 'clear_tool_uses_20250919'}]},
    )
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='list_job_queues', description='List the job queue options.')],
        output_mode='native',
        output_object=OutputObjectDefinition(
            json_schema={'type': 'object', 'properties': {'queue': {'type': 'string'}}, 'required': ['queue']}
        ),
    )

    response = await model.compact_messages(_context(model, messages, settings, parameters))

    assert isinstance(response.parts[0], CompactionPart)
    body = request_capture.body('/v1/messages')
    assert {key: body.get(key) for key in ('tool_choice', 'output_config', 'stop_sequences', 'context_management')} == (
        snapshot(
            {
                'tool_choice': {'type': 'auto'},
                'output_config': {'task_budget': {'type': 'tokens', 'total': 100000}},
                'stop_sequences': None,
                'context_management': None,
            }
        )
    )
    assert message_shape(body)[-2:] == snapshot([('user', ['text']), ('assistant', ['text'])])


async def test_compact_messages_without_summary(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
):
    """A 200 without a block is an error, not an empty summary that would drop the history it can't replace."""
    model = anthropic_model('claude-haiku-5-5', capture=True)

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        await model.compact_messages(_context(model, _conversation(), AnthropicModelSettings(max_tokens=5)))

    assert str(exc_info.value) == snapshot(
        "Anthropic returned no compaction summary (stop reason 'max_tokens'). Raise `max_tokens` to leave room for the summary."
    )


async def test_compact_messages_unsupported_model(allow_model_requests: None, anthropic_model: AnthropicModelFactory):
    """A model without on-demand compaction raises before sending, rather than with the API's 400.

    No request is sent, so there is nothing to record.
    """
    model = anthropic_model('claude-haiku-4-5')

    with pytest.raises(UserError, match="'claude-haiku-4-5' does not support on-demand compaction"):
        await model.compact_messages(_context(model, _conversation()))
