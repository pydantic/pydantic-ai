"""The orchestrator's session tools: find the saved sessions of every project, and archive or unarchive them."""

import functools
import json
import os
import re
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary
from pydantic_clai2.builtin_plugins.web.protocol import ServerMessage, SessionsChanged, ToolDone, TurnEnded
from pydantic_clai2.builtin_plugins.web.session_tools import SessionTools
from pydantic_clai2.builtin_plugins.web.workspace_tools import reported
from tests.clai2.test_session_browser import repository
from tests.clai2.test_web import TOKEN, MakeWeb, Web, make_web
from tests.clai2.test_web_orchestrator import WORKSPACES, opened, orchestrator_until

__all__ = ('make_web',)


def backdate(web_: Web, conversation_id: str, *, days: float) -> None:
    """Make a saved session last active `days` ago, as if no one had touched it since."""
    moment = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    with closing(sqlite3.connect(web_.shell.sessions.store.database)) as conn, conn:
        conn.execute(
            "UPDATE conversations SET updated_at = ?, metadata = json_set(metadata, '$.updated_at', ?) WHERE id = ?",
            (moment, moment, conversation_id),
        )


def save(web_: Web, summary: ConversationSummary, *, days: float = 0) -> str:
    """Save a session through the store `/resume` uses, last active `days` ago."""
    store = web_.shell.sessions.store
    saved = web_.portal.call(lambda: store.save(summary=summary, messages=[]))
    backdate(web_, saved.id, days=days)
    return saved.id


def sidebar(web_: Web, *, archived: bool = False) -> dict[str, bool]:
    """The titles the sidebar lists, each with whether it is archived."""
    projects = web_.get(f'/api/sessions?include_archived={str(archived).lower()}').json()
    return {
        entry['title']: entry['archived']
        for project in projects
        for worktree in project['worktrees']
        for entry in worktree['sessions']
    }


def returned(messages: list[ModelMessage]) -> str:
    return next(
        str(part.content)
        for message in reversed(messages)
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )


async def cleaning(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    """Asked to clean up, finds the sessions older than three days and archives every one it found."""
    step = sum(isinstance(message, ModelResponse) for message in messages)
    if step == 0:
        yield {0: DeltaToolCall(name='sessions_find', json_args=json.dumps({'older_than_days': 3}))}
    elif step == 1:
        found = [found for found in re.findall(r'^\| (\S+) \|', returned(messages), flags=re.M) if found != 'id']
        yield {0: DeltaToolCall(name='sessions_archive', json_args=json.dumps({'ids': found}))}
    else:
        yield 'Archived the old sessions.'


def test_the_orchestrator_archives_sessions_older_than_three_days(make_web: MakeWeb, tmp_path: Path) -> None:
    """Old sessions of every project leave the sidebar, listed again under archived; open and running ones stay."""
    web_ = make_web(Agent(FunctionModel(stream_function=cleaning), deps_type=type(None)), extra=[WORKSPACES])
    alpha, feature, beta = tmp_path / 'alpha', tmp_path / 'alpha-feature', tmp_path / 'beta'
    repository(alpha, linked=feature, branch='feature')
    beta.mkdir()
    old = {
        'Alpha old': save(web_, ConversationSummary(workspace=str(alpha.resolve()), title='Alpha old'), days=5),
        'Feature old': save(web_, ConversationSummary(workspace=str(feature.resolve()), title='Feature old'), days=4),
        'Beta old': save(web_, ConversationSummary(workspace=str(beta.resolve()), title='Beta old'), days=10),
    }
    save(web_, ConversationSummary(workspace=str(alpha.resolve()), title='Alpha new'), days=1)
    busy = ConversationSummary(workspace=str(beta.resolve()), title='Busy', outcome='running', owner_pid=os.getpid())
    busy_id = save(web_, busy, days=7)
    web_.portal.call(web_.shell.session.commit_messages, list[ModelMessage]())
    current = web_.shell.session.conversation_id
    backdate(web_, current, days=6)

    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        opened(socket)
        web_.post('/api/orchestrator/input', {'text': 'Blast any worktree on any project older than 3 days'})
        seen, ended = orchestrator_until(socket, TurnEnded)
    assert ended.outcome == 'completed'
    found, archived = [event.text for event in seen if isinstance(event, ToolDone)]
    assert found.splitlines()[:3] == [
        '5 of 5 matching sessions, newest first.',
        '| id | title | project | worktree | last active | archived |',
        '|---|---|---|---|---|---|',
    ]
    rows = {line.split(' | ')[1]: line for line in found.splitlines()[3:]}
    assert list(rows) == ['Feature old', 'Alpha old', 'New session (current session)', 'Busy', 'Beta old']
    assert rows['Feature old'].split(' | ')[2:4] == ['alpha', 'feature']
    assert rows['Alpha old'].split(' | ')[2:4] == ['alpha', 'main']
    assert '(5 days ago) | no |' in rows['Alpha old']
    assert archived.splitlines() == [
        'Archived 3 sessions:',
        f'- {old["Feature old"]} (Feature old)',
        f'- {old["Alpha old"]} (Alpha old)',
        f'- {old["Beta old"]} (Beta old)',
        'Refused 2:',
        f'- {current}: The current session cannot be archived. Switch to another one first.',
        f'- {busy_id}: Session is busy in process {os.getpid()}. A running session cannot be archived.',
    ]
    assert sidebar(web_) == {'Alpha new': False, 'New session': False, 'Busy': False}
    assert sidebar(web_, archived=True) == {
        'Alpha new': False,
        'New session': False,
        'Busy': False,
        'Alpha old': True,
        'Feature old': True,
        'Beta old': True,
    }


def test_finding_and_unarchiving(make_web: MakeWeb, tmp_path: Path) -> None:
    """Search by text or project, cut to `limit`; archive and unarchive by ID, refusing unknown ones."""
    web_ = make_web(Agent(TestModel()))
    published: list[ServerMessage] = []
    tools = SessionTools(web_.plugin.host, publish=published.append)

    def call(tool: Callable[..., Awaitable[str]], *args: object, **kwargs: object) -> str:
        return web_.portal.call(functools.partial(tool, *args, **kwargs))

    assert call(tools.sessions_find) == 'No saved sessions match.'
    alpha, beta = tmp_path / 'alpha', tmp_path / 'beta'
    alpha.mkdir()
    beta.mkdir()
    parser = save(web_, ConversationSummary(workspace=str(alpha), title='Parser | fix'), days=1)
    save(web_, ConversationSummary(workspace=str(beta), title='Docs'))

    both = call(tools.sessions_find, limit=1).splitlines()
    assert both[0] == '1 of 2 matching sessions, newest first.'
    assert ' | Docs | beta | ' in both[3] and '(today) | no |' in both[3]
    [row] = call(tools.sessions_find, project='ALPHA').splitlines()[3:]
    assert row.startswith(f'| {parser} | Parser / fix | alpha | {alpha} | ') and '(1 day ago)' in row
    assert call(tools.sessions_find, query='nothing like it') == 'No saved sessions match.'
    # A folder removed since the last search is placed again, as the sidebar places it.
    beta.rmdir()
    [gone] = call(tools.sessions_find, query='Docs').splitlines()[3:]
    assert ' | Docs | Removed folders | beta | ' in gone

    assert call(tools.sessions_archive, ['missing', parser, parser]).splitlines() == [
        'Archived 1 session:',
        f'- {parser} (Parser / fix)',
        'Refused 1:',
        '- missing: No saved session: missing',
    ]
    assert published == [SessionsChanged()]
    assert [line.split(' | ')[1] for line in call(tools.sessions_find).splitlines()[3:]] == ['Docs']
    archived = call(tools.sessions_find, query='Parser', include_archived=True).splitlines()[3]
    assert archived.endswith('| yes |')
    assert call(tools.sessions_unarchive, [parser]).splitlines() == [
        'Unarchived 1 session:',
        f'- {parser} (Parser / fix)',
    ]
    assert call(tools.sessions_unarchive, ['missing']).splitlines() == [
        'Unarchived 0 sessions.',
        'Refused 1:',
        '- missing: No saved session: missing',
    ]
    assert len(published) == 2, 'nothing changed, so the sidebar is not told'

    web_.plugin.host.sessions = None
    assert call(reported(tools.sessions_find)) == 'Error: This CLAI does not save sessions.'
