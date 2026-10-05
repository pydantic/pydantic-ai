"""`/logfire optimize`: gated on observability's Logfire sign-in, reads runs over `/v1/query`, proposes advisory changes."""

import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
import pytest
from pydantic import JsonValue
from rich.console import Console

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_clai2.builtin_plugins import logfire_optimize
from pydantic_clai2.builtin_plugins.logfire import NO_READ_TOKEN, NOT_SIGNED_IN, LogfirePlugin
from pydantic_clai2.builtin_plugins.logfire_optimize import instructions_text, transcript
from pydantic_clai2.config.api_keys import save_key
from pydantic_clai2.plugins import LoadedPlugin, PluginHost, SessionEnd, Transcript, load_plugin
from tests.clai2.test_logfire import Recorder, recorder as recorder

US = 'https://logfire-us.pydantic.dev'
SIGNED_IN: dict[str, JsonValue] = {
    'token': {'name': 'LOGFIRE_TOKEN_TEAM'},
    'base_url': US,
    'read_access': {'token': {'name': 'LOGFIRE_READ_TOKEN_TEAM'}, 'write_token': {'name': 'LOGFIRE_TOKEN_TEAM'}},
    'send_to_logfire': False,
}
MESSAGES = json.dumps(
    [
        {'role': 'system', 'parts': [{'type': 'text', 'content': 'SYSTEM PROMPT, SENT SEPARATELY'}]},
        {'role': 'user', 'parts': [{'type': 'text', 'content': 'Fix the  failing\ntest'}]},
        {
            'role': 'assistant',
            'parts': [
                {'type': 'thinking', 'content': 'hmm'},
                {'type': 'tool_call', 'id': 'c1', 'name': 'run_tests', 'arguments': {'path': 'tests'}},
            ],
        },
        {
            'role': 'user',
            'parts': [{'type': 'tool_call_response', 'id': 'c1', 'name': 'run_tests', 'result': '1 failed'}],
        },
        {'role': 'assistant', 'parts': ['not a part', {'type': 'image'}]},
        {'role': 'assistant', 'parts': 'not a list'},
        'not a message',
    ]
)
RUNS: list[dict[str, JsonValue]] = [
    {
        'trace_id': 'a' * 32,
        'start_timestamp': '2026-10-05T12:00:00.123Z',
        'failed': True,
        'error': 'ValueError: boom',
        'instructions': json.dumps([{'type': 'text', 'content': 'Be helpful.'}]),
        'messages': MESSAGES,
        'final_result': None,
    },
    {
        'trace_id': 'b' * 32,
        'start_timestamp': '2026-10-05T13:00:00.000Z',
        'failed': False,
        'error': None,
        'instructions': json.dumps([{'type': 'text', 'content': 'Be terse.'}, {'type': 'text'}, 'odd']),
        'messages': None,
        'final_result': 'Done',
    },
    {'trace_id': 'c' * 32, 'start_timestamp': '2026-10-04T09:30:00.000Z', 'failed': True},
]
PROPOSAL: dict[str, JsonValue] = {
    'summary': 'Runs stop at the first failing test.',
    'evidence_quality': 'moderate',
    'issues': [
        {'title': 'Gives up early', 'detail': 'Retry once before answering.', 'evidence': ['E1', 'E1', 'E99']},
        {'title': 'Uncited', 'detail': 'Nothing shows it.', 'evidence': []},
    ],
    'working_well': ['Short answers'],
    'recommendations': [],
    'proposed_system_prompt': 'Be terse.\nRetry a failing test once.',
}


@dataclass
class FakeQuery:
    """Logfire's `/v1/query`, answering scripted responses in order and recording the requests."""

    answers: list[httpx.Response | Callable[[httpx.Request], httpx.Response]]
    requests: list[httpx.Request] = field(default_factory=list[httpx.Request])

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.pop(0)
        return answer if isinstance(answer, httpx.Response) else answer(request)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))


def rows(*rows: dict[str, JsonValue], project: str = 'pydantic/agents') -> httpx.Response:
    return httpx.Response(200, json={'columns': [], 'rows': list(rows)}, headers={'X-Logfire-Context': project})


def refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError('refused', request=request)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> FakeQuery:
    fake = FakeQuery(answers=[])
    monkeypatch.setattr(logfire_optimize, 'HTTP', fake.client)
    return fake


def proposing(seen: list[list[ModelMessage]], proposal: dict[str, JsonValue] = PROPOSAL) -> FunctionModel:
    def review(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, proposal)])

    return FunctionModel(review)


async def load(
    settings: dict[str, JsonValue] = SIGNED_IN, model: Model | None = None
) -> tuple[LoadedPlugin[None], io.StringIO]:
    save_key(name='LOGFIRE_READ_TOKEN_TEAM', value='pylf_v1_us_read')
    output = io.StringIO()
    host = PluginHost[None](
        name='observability',
        console=Console(file=output, width=200),
        settings=settings,
        conversation=Transcript(model=model),
    )
    return load_plugin(LogfirePlugin, host), output


async def run(command: str, *, settings: dict[str, JsonValue] = SIGNED_IN, model: Model | None = None) -> str:
    plugin, _ = await load(settings, model)
    try:
        return await plugin.commands.execute_async(command)
    finally:
        await plugin.dispatch(SessionEnd(reason='exit'))


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({}, NOT_SIGNED_IN),
        ({'token': {'name': 'LOGFIRE_TOKEN_TEAM'}}, NOT_SIGNED_IN),  # LOGFIRE_TOKEN-style: no setup URL.
        ({'token': {'name': 'LOGFIRE_TOKEN_TEAM'}, 'base_url': US}, NO_READ_TOKEN),  # Set up before this command.
        ({**SIGNED_IN, 'token': {'name': 'LOGFIRE_TOKEN_OTHER'}}, NO_READ_TOKEN),  # Read token is another project's.
        (
            {**SIGNED_IN, 'read_access': {'token': {'name': 'GONE'}, 'write_token': {'name': 'LOGFIRE_TOKEN_TEAM'}}},
            f'GONE is not in /keys. {NO_READ_TOKEN}',
        ),
    ],
)
async def test_refuses_without_the_observability_sign_in(
    settings: dict[str, JsonValue], message: str, server: FakeQuery, recorder: Recorder
) -> None:
    with pytest.raises(ValueError) as error:
        await run('/logfire optimize', settings={'send_to_logfire': False, **settings})
    assert str(error.value) == message
    assert server.requests == []  # Refused before anything reached Logfire.


@pytest.mark.parametrize(
    'command',
    [
        '/logfire',
        '/logfire status',
        '/logfire optimize preview',
        '/logfire optimize propose',
        '/logfire optimize agents extra',
        '/logfire optimize apply x',
    ],
)
async def test_usage(command: str, server: FakeQuery, recorder: Recorder) -> None:
    with pytest.raises(ValueError, match='Usage: /logfire optimize'):
        await run(command)


async def test_lists_agents_then_completes_their_names(server: FakeQuery, recorder: Recorder) -> None:
    server.answers = [
        rows(
            {'agent': 'support', 'runs': 42, 'failures': 3, 'last_run': '2026-10-05T13:00:00Z'},
            {'agent': 'clai2', 'runs': 1, 'failures': 0, 'last_run': '2026-10-04T09:30:00Z'},
        )
    ]
    plugin, _ = await load()
    try:
        assert await plugin.commands.execute_async('/logfire optimize') == (
            'Agents with runs in pydantic/agents over the last 7 days:\n'
            '  support  42 runs, 3 failed, last 2026-10-05 13:00\n'
            '  clai2    1 runs, 0 failed, last 2026-10-04 09:30\n'
            'Next: /logfire optimize preview AGENT, then /logfire optimize propose AGENT [focus]'
        )
        [command] = plugin.commands
        assert list(command.complete([])) == ['optimize']
        assert list(command.complete(['optimize', ''])) == ['agents', 'preview', 'propose']
        assert list(command.complete(['other', ''])) == []
        assert list(command.complete(['optimize', 'propose', ''])) == ['support', 'clai2']
        assert list(command.complete(['optimize', 'agents', ''])) == []
    finally:
        await plugin.dispatch(SessionEnd(reason='exit'))
    [request] = server.requests
    assert str(request.url).startswith(f'{US}/v1/query?')
    assert request.headers['Authorization'] == 'pylf_v1_us_read'
    params = request.url.params
    assert (params['json_rows'], params['limit']) == ('true', '20')
    assert params['min_timestamp']
    assert 'FROM records' in params['sql'] and "LIKE 'invoke_agent %'" in params['sql']


async def test_no_agents_says_how_runs_appear(server: FakeQuery, recorder: Recorder) -> None:
    server.answers = [httpx.Response(200, json={'rows': []})]
    assert await run('/logfire optimize agents') == (
        'No agent runs in your Logfire project over the last 7 days. Runs appear once an instrumented '
        'Pydantic AI agent sends traces there.'
    )


async def test_preview_shows_the_prompt_and_each_run(server: FakeQuery, recorder: Recorder) -> None:
    server.answers = [rows(*RUNS)]
    assert await run("/logfire optimize preview O'Brien") == (
        "Evidence for O'Brien in pydantic/agents, last 7 days: 3 runs, 2 failed (failures first).\n"
        'Current system prompt: 9 characters, from the newest run that recorded it.\n'
        f'  E1  failed  2026-10-05 12:00  trace {"a" * 32}\n'
        '      ValueError: boom\n'
        f'  E2  ok      2026-10-05 13:00  trace {"b" * 32}\n'
        '      output: Done\n'
        f'  E3  failed  2026-10-04 09:30  trace {"c" * 32}\n'
        '      error status\n'
        "Next: /logfire optimize propose O'Brien [focus]"
    )
    assert "= 'O''Brien'" in server.requests[0].url.params['sql']  # Quoted, not injected.


async def test_preview_without_runs_or_a_recorded_prompt(server: FakeQuery, recorder: Recorder) -> None:
    server.answers = [rows(), rows(RUNS[2])]
    assert (
        await run('/logfire optimize preview support') == 'No runs of support in pydantic/agents over the last 7 days.'
    )
    preview = await run('/logfire optimize preview support')
    assert 'Current system prompt: not recorded, so a proposal can only recommend changes.' in preview


async def test_propose_reviews_the_evidence_with_the_session_model(server: FakeQuery, recorder: Recorder) -> None:
    server.answers = [rows(*RUNS)]
    seen: list[list[ModelMessage]] = []
    plugin, output = await load(model=proposing(seen))
    try:
        result = await plugin.commands.execute_async("/logfire optimize propose support don't skip retries")
    finally:
        await plugin.dispatch(SessionEnd(reason='exit'))
    assert result == (
        'Proposal for support from 3 runs in pydantic/agents (2 failed, last 7 days). Evidence quality: moderate.\n'
        '\n'
        'Runs stop at the first failing test.\n'
        '\n'
        'Issues\n'
        '1. Gives up early\n'
        '   Retry once before answering.\n'
        f'   Evidence: E1 (trace {"a" * 32})\n'
        '2. Uncited\n'
        '   Nothing shows it.\n'
        '\n'
        'Working well\n'
        '- Short answers\n'
        '\n'
        'Proposed system prompt change:\n'
        '--- current prompt\n'
        '+++ proposed prompt\n'
        '@@ -1 +1,2 @@\n'
        ' Be terse.\n'
        '+Retry a failing test once.\n'
        '\n'
        'Advisory only: nothing was changed. Edit the prompt in your agent code.'
    )
    assert output.getvalue().endswith('Reviewing 3 runs of support...\n')
    [[request]] = seen
    assert isinstance(request, ModelRequest)
    [prompt] = [part.content for part in request.parts if isinstance(part, UserPromptPart)]
    assert isinstance(prompt, str)
    assert prompt.startswith('Agent: support\n\nCurrent system prompt:\n<prompt>\nBe terse.\n</prompt>')
    assert (
        '<evidence id="E1" started="2026-10-05T12:00:00.123Z" outcome="failed: ValueError: boom">\n'
        'user: Fix the failing test\n'
        'assistant called run_tests({"path": "tests"})\n'
        'tool run_tests returned: 1 failed\n'
        'assistant: [image]\n'
        '</evidence>'
    ) in prompt
    assert 'outcome="failed: error status"' in prompt
    assert 'SENT SEPARATELY' not in prompt
    assert prompt.endswith("The reviewer asks you to focus on: don't skip retries")
    assert request.instructions == logfire_optimize.INSTRUCTIONS.strip()


@pytest.mark.parametrize(
    ('runs', 'proposed'),
    [
        ([RUNS[2]], 'Invented prompt'),  # The prompt was never recorded, so nothing replaces it.
        (RUNS, ' Be terse. '),  # Unchanged.
        (RUNS, None),
    ],
)
async def test_propose_without_a_prompt_change(
    runs: list[dict[str, JsonValue]], proposed: str | None, server: FakeQuery, recorder: Recorder
) -> None:
    server.answers = [rows(*runs)]
    proposal: dict[str, JsonValue] = {
        **PROPOSAL,
        'issues': [],
        'working_well': [],
        'recommendations': ['Add a retry tool.'],
    }
    seen: list[list[ModelMessage]] = []
    result = await run(
        '/logfire optimize propose support', model=proposing(seen, {**proposal, 'proposed_system_prompt': proposed})
    )
    assert 'Issues' not in result and 'Working well' not in result
    assert '\nRecommendations\n- Add a retry tool.\n\nNo system prompt change proposed.\n' in result
    [[request]] = seen
    assert isinstance(request, ModelRequest)
    part = request.parts[-1]
    assert isinstance(part, UserPromptPart)
    prompt = str(part.content)
    assert ('(unknown: no run recorded its system instructions)' in prompt) == (runs == [RUNS[2]])
    assert 'focus on' not in prompt


async def test_propose_needs_a_model(server: FakeQuery, recorder: Recorder) -> None:
    server.answers = [rows(*RUNS)]
    with pytest.raises(ValueError, match='Choose a model first'):
        await run('/logfire optimize propose support')


@pytest.mark.parametrize(
    ('answer', 'message'),
    [
        (refuse, 'Could not reach Logfire (ConnectError); try again.'),
        (httpx.Response(401), 'Logfire refused the read token (HTTP 401); it may have been revoked.'),
        (httpx.Response(403), 'Logfire refused the read token (HTTP 403)'),
        (
            httpx.Response(400, json={'detail': 'bad \x1b[2J sql'}),
            'Logfire refused the query (HTTP 400): bad \\x1b[2J sql.',
        ),
        (httpx.Response(400, json={'detail': [{'msg': 'x'}]}), 'Logfire refused the query (HTTP 400).'),
        (httpx.Response(500, text='oops'), 'Logfire refused the query (HTTP 500).'),
        (httpx.Response(200, json={'columns': []}), 'Logfire answered the query with something unexpected.'),
        (
            httpx.Response(200, json={'rows': [{'agent': 1}]}),
            'Logfire answered the query with rows CLAI did not expect.',
        ),
    ],
)
async def test_query_failures_say_what_went_wrong(
    answer: httpx.Response | Callable[[httpx.Request], httpx.Response],
    message: str,
    server: FakeQuery,
    recorder: Recorder,
) -> None:
    server.answers = [answer]
    with pytest.raises(ValueError) as error:
        await run('/logfire optimize')
    assert str(error.value).startswith(message)


async def test_logfire_text_is_made_inert(server: FakeQuery, recorder: Recorder) -> None:
    server.answers = [rows({'agent': 'evil\x1b]52;c;x\x07', 'runs': 1, 'failures': 0, 'last_run': '2026-10-05T13:00'})]
    listing = await run('/logfire optimize')
    assert '\x1b' not in listing and 'evil\\x1b]52;c;x\\x07' in listing


def test_instructions_text_reads_every_recorded_shape() -> None:
    assert instructions_text(None) is None
    assert instructions_text('"Be brief."') == 'Be brief.'
    assert instructions_text('""') is None
    assert instructions_text('{"content": "x"}') is None
    assert instructions_text('[]') is None
    assert instructions_text('not JSON, but the prompt') == 'not JSON, but the prompt'


def test_long_transcripts_keep_the_start_and_the_end() -> None:
    turns = [{'role': 'user', 'parts': [{'type': 'text', 'content': f'{index} ' + 'x' * 500}]} for index in range(30)]
    text = transcript(json.dumps(turns), 'final answer')
    assert len(text) < 4100
    assert text.startswith('user: 0 xxx')
    assert '\n[...]\n' in text
    assert text.endswith('output: final answer')
    assert transcript('{"not": "a list"}', None).startswith('(no messages recorded')


async def test_the_real_client_waits_for_slow_queries() -> None:
    async with logfire_optimize.HTTP() as http:
        assert (http.timeout.connect, http.timeout.read) == (30, 120)
