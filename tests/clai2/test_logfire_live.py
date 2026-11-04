"""The `observability` plugin's side panel: Logfire's live view of the conversation, without reaching Logfire."""

import io
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from opentelemetry.trace import SpanContext, format_span_id, format_trace_id
from pydantic import JsonValue
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2.builtin_plugins import logfire as logfire_plugin
from pydantic_clai2.builtin_plugins.logfire import NOT_SENDING
from pydantic_clai2.builtin_plugins.logfire_live import live_view_url, project_url
from pydantic_clai2.builtin_plugins.logfire_setup import Setup
from pydantic_clai2.config import Settings
from pydantic_clai2.plugins import LoadedPlugin, PanelView, PluginHost, SessionStart
from tests.clai2.test_logfire import Recorder, close, load_logfire, operation, recorder as recorder

US_TOKEN = 'pylf_v1_us_' + 'a' * 40
EU_TOKEN = 'pylf_v1_eu_' + 'b' * 40


def answering(*responses: httpx.Response) -> tuple[Setup, list[httpx.Request]]:
    """A `Setup` whose Logfire gives these answers in turn, and the requests it received."""
    received: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return responses[min(len(received), len(responses)) - 1]

    return Setup(http=lambda: httpx.AsyncClient(transport=httpx.MockTransport(answer))), received


def info(project: str) -> httpx.Response:
    return httpx.Response(200, json={'project_name': 'clai', 'project_url': project})


def logfire_with(
    settings: dict[str, JsonValue] | None = None, *, conversation: list[str] | None = None
) -> LoadedPlugin[None]:
    """The plugin in a shell holding `conversation[0]`, which a test may switch, else `session-1`."""
    held = conversation or ['session-1']
    host = PluginHost[None](
        name='observability', console=Console(file=io.StringIO()), settings=settings or {}, session_id=lambda: held[0]
    )
    return load_logfire(host)


async def view(plugin: LoadedPlugin[None]) -> PanelView:
    [panel] = plugin.side_panels
    assert (panel.id, panel.title, panel.icon) == ('live', 'Logfire live view', 'traces')
    return await panel.view()


def focus(url: str | None) -> dict[str, list[str]]:
    assert url is not None
    return parse_qs(urlsplit(url).query)


@pytest.mark.parametrize(
    'token, settings, api, site',
    [
        pytest.param(US_TOKEN, {}, 'https://logfire-us.pydantic.dev', 'https://logfire-us.pydantic.dev', id='us'),
        pytest.param(EU_TOKEN, {}, 'https://logfire-eu.pydantic.dev', 'https://logfire-eu.pydantic.dev', id='eu'),
        pytest.param(
            US_TOKEN,
            {'base_url': 'logfire.example.com/mcp'},
            'https://logfire.example.com',
            'https://ui.example.com',
            id='custom',
        ),
    ],
)
async def test_the_live_view_follows_the_conversation(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, token: str, settings: dict[str, JsonValue], api: str, site: str
) -> None:
    """Logfire names the project once; the view shows the conversation's trace, opened at its latest run."""
    monkeypatch.setenv('LOGFIRE_TOKEN', token)
    setup, received = answering(info(f'{site}/team/clai'))
    monkeypatch.setattr(logfire_plugin, 'SETUP', setup)
    conversation = ['session-1']
    plugin = logfire_with(settings, conversation=conversation)
    project = f'{site}/team/clai'
    # Before the session starts there is no trace yet: the project's live view.
    assert await view(plugin) == PanelView(url=project)
    agent = Agent(TestModel(), deps_type=type(None))
    await plugin.dispatch(SessionStart(agent=agent, settings=Settings()))
    try:
        started = (await view(plugin)).url
        assert started is not None and started.startswith(f'{project}?q=trace_id%3D%27')
        [query] = focus(started)['q']
        trace = query.removeprefix("trace_id='").removesuffix("'")
        assert len(trace) == 32 and set(focus(started)) == {'q'}
        runs: list[str] = []
        for prompt in ('first', 'second'):
            await agent.run(prompt, capabilities=plugin.capabilities, conversation_id='session-1')
            opened = focus((await view(plugin)).url)
            assert (opened['q'], opened['traceId']) == ([query], [trace])
            runs.append(opened['spanId'][0])
        # Another conversation's run, such as a background fork, leaves the view on this one's latest.
        await agent.run('elsewhere', capabilities=plugin.capabilities, conversation_id='other')
        assert focus((await view(plugin)).url)['spanId'] == [runs[-1]]
        # A new conversation has no trace until something is recorded in it; reading the view opens none.
        conversation[0] = 'session-2'
        assert await view(plugin) == PanelView(url=project)
        assert await view(plugin) == PanelView(url=project)
        await agent.run('fresh', capabilities=plugin.capabilities, conversation_id='session-2')
        fresh = focus((await view(plugin)).url)
        assert fresh['q'] != [query] and fresh['traceId'] != [trace]
    finally:
        await close(plugin)
    spans = {
        format_span_id(context.span_id): (span, format_trace_id(context.trace_id))
        for span in recorder.spans()
        if (context := span.get_span_context()) is not None
    }
    assert runs[0] != runs[1]
    assert [(operation(spans[run][0]), spans[run][1]) for run in runs] == [('invoke_agent', trace)] * 2
    assert [(str(request.url), request.headers['authorization']) for request in received] == [(f'{api}/v1/info', token)]


@pytest.mark.parametrize(
    'environment, settings',
    [
        pytest.param({}, {}, id='no-token'),
        pytest.param({'LOGFIRE_TOKEN': US_TOKEN}, {'send_to_logfire': False}, id='sending-off'),
    ],
)
async def test_without_a_project_the_panel_says_how_to_set_one_up(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], settings: dict[str, JsonValue]
) -> None:
    monkeypatch.delenv('LOGFIRE_TOKEN', raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    setup, received = answering(info('https://logfire-us.pydantic.dev/team/clai'))
    monkeypatch.setattr(logfire_plugin, 'SETUP', setup)
    plugin = logfire_with(settings)
    assert await view(plugin) == PanelView(note=NOT_SENDING)
    assert received == []


async def test_logfire_is_asked_again_until_it_names_the_project(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('LOGFIRE_TOKEN', US_TOKEN)
    setup, received = answering(httpx.Response(503), info('https://logfire-us.pydantic.dev/team/clai'))
    monkeypatch.setattr(logfire_plugin, 'SETUP', setup)
    plugin = logfire_with()
    assert await view(plugin) == PanelView(
        note='https://logfire-us.pydantic.dev did not say which project the write token belongs to.'
    )
    assert await view(plugin) == PanelView(url='https://logfire-us.pydantic.dev/team/clai')
    assert await view(plugin) == PanelView(url='https://logfire-us.pydantic.dev/team/clai')
    assert len(received) == 2


@pytest.mark.parametrize(
    'response',
    [
        pytest.param(httpx.Response(401, json={'detail': 'Invalid token'}), id='refused'),
        pytest.param(httpx.Response(200, json={'project_name': 'clai'}), id='unexpected'),
        pytest.param(info('javascript:alert(1)'), id='not-a-web-page'),
    ],
)
async def test_project_url_without_an_answer(response: httpx.Response) -> None:
    setup, _ = answering(response)
    assert await project_url(setup, base_url='https://logfire-us.pydantic.dev/', token=US_TOKEN) is None


async def test_project_url_when_logfire_is_unreachable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('unreachable', request=request)

    setup = Setup(http=lambda: httpx.AsyncClient(transport=httpx.MockTransport(refuse)))
    assert await project_url(setup, base_url='https://logfire-us.pydantic.dev', token=US_TOKEN) is None


def test_live_view_url() -> None:
    trace = SpanContext(trace_id=0x0AF7651916CD43DD8448EB211C80319C, span_id=1, is_remote=False)
    run = SpanContext(trace_id=trace.trace_id, span_id=0xB7AD6B7169203331, is_remote=False)
    project = 'https://logfire-eu.pydantic.dev/team/clai/'
    assert live_view_url(project, trace=trace, run=None) == (
        'https://logfire-eu.pydantic.dev/team/clai?q=trace_id%3D%270af7651916cd43dd8448eb211c80319c%27'
    )
    assert live_view_url(project, trace=trace, run=run) == (
        'https://logfire-eu.pydantic.dev/team/clai?q=trace_id%3D%270af7651916cd43dd8448eb211c80319c%27'
        '&traceId=0af7651916cd43dd8448eb211c80319c&spanId=b7ad6b7169203331'
    )
