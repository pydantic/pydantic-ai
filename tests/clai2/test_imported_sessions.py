"""Claude Code and Codex sessions: reading their transcripts, listing them in `/resume`, and importing them."""

import json
import os
from collections.abc import Sequence
from dataclasses import replace
from io import StringIO
from pathlib import Path
from uuid import UUID, uuid4, uuid5

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic import TypeAdapter
from rich.console import Console
from termflow.tui.completion import CompleteEvent, Document
from termflow.tui.keys import Key

from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    CompactionPart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, SqliteConversationStore
from pydantic_ai_harness.subagents import DelegationTask, DelegationTasks, is_task_report
from pydantic_clai2 import chat
from pydantic_clai2.app import create_stock_agent
from pydantic_clai2.cli import cli, headless
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime import claude_code_sessions, codex_sessions
from pydantic_clai2.runtime.imported_history import (
    Header,
    HistoryBuilder,
    imported_kind,
    is_imported_context,
    is_typed_prompt,
    records,
    tail_records,
    timestamp,
)
from pydantic_clai2.runtime.imported_sessions import (
    ImportCatalog,
    find_import,
    import_source,
    imported_from,
    merge,
    save_import,
)
from pydantic_clai2.runtime.session import Session
from pydantic_clai2.runtime.session_runtime import create_shell
from pydantic_clai2.runtime.sessions import Sessions
from pydantic_clai2.ui.menus.session_browser import SessionBrowser

AT = '2026-10-01T12:00:00Z'
CLAUDE_ID = '1f6c7d2e-0b4a-4c1e-9d3f-5a2b8c9e7f10'
CODEX_ID = '01a0f2e1-2984-7223-97ad-c798766558e7'
ENCRYPTED = "[Encrypted by Codex for OpenAI's models, and left out.]"


def write_jsonl(path: Path, lines: Sequence[object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ''.join((line if isinstance(line, str) else json.dumps(line)) + '\n' for line in lines), encoding='utf-8'
    )
    return path


def claude_entry(uuid: str, parent: str | None, kind: str, content: object, **extra: object) -> dict[str, object]:
    message: dict[str, object] = {'role': kind, 'content': content}
    if 'model' in extra:
        message['model'] = extra.pop('model')
    if 'usage' in extra:
        message['usage'] = extra.pop('usage')
    if 'id' in extra:
        message['id'] = extra.pop('id')
    return {'type': kind, 'uuid': uuid, 'parentUuid': parent, 'timestamp': AT, 'message': message, **extra}


def claude_session(cwd: Path, *, session_id: str = CLAUDE_ID) -> Path:
    """A session with a summary, a tool round, a rewound branch, a synthetic notice, and a sub-agent sidechain."""
    model = 'claude-sonnet-4-5'
    return write_jsonl(
        Path.home() / '.claude' / 'projects' / '-work' / f'{session_id}.jsonl',
        [
            {'type': 'summary', 'summary': 'Fix the parser', 'leafUuid': 'elsewhere'},
            {**claude_entry('u1', None, 'user', 'Fix the parser please'), 'cwd': str(cwd)},
            claude_entry('meta', 'u1', 'user', 'Caveat: local command output', isMeta=True),
            claude_entry(
                'a1',
                'meta',
                'assistant',
                [
                    {'type': 'thinking', 'thinking': 'Read it first', 'signature': 'sig'},
                    {'type': 'text', 'text': 'Looking'},
                    {'type': 'tool_use', 'id': 't1', 'name': 'Read', 'input': {'file_path': 'a.py'}},
                ],
                model=model,
                usage={
                    'input_tokens': 10,
                    'cache_read_input_tokens': 100,
                    'cache_creation_input_tokens': 5,
                    'output_tokens': 7,
                },
            ),
            claude_entry(
                'a1b',
                'a1',
                'assistant',
                [
                    {'type': 'redacted_thinking', 'data': 'x'},
                    {'type': 'tool_use', 'id': 't2', 'name': 'Bash', 'input': {'command': 'ls'}},
                ],
                model=model,
            ),
            claude_entry(
                'u2',
                'a1b',
                'user',
                [
                    {'type': 'tool_result', 'tool_use_id': 't1', 'content': 'file text'},
                    {'type': 'tool_result', 'tool_use_id': 't2', 'content': [{'type': 'text', 'text': 'two'}]},
                    {'type': 'tool_result', 'tool_use_id': 'compacted-away', 'content': 'orphan'},
                ],
            ),
            claude_entry('rewound', 'u2', 'assistant', [{'type': 'text', 'text': 'Abandoned answer'}], model=model),
            claude_entry(
                'u3',
                'u2',
                'user',
                [
                    {'type': 'text', 'text': 'Now with this image'},
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'cG5n'}},
                    {'type': 'image', 'source': {'type': 'url', 'url': 'https://example.com/a.png'}},
                    {'type': 'document'},
                ],
            ),
            claude_entry('synthetic', 'u3', 'assistant', [{'type': 'text', 'text': 'API Error'}], model='<synthetic>'),
            claude_entry('a3', 'synthetic', 'assistant', [{'type': 'text', 'text': 'Done'}], model=model),
            {'type': 'system', 'uuid': 'tail', 'parentUuid': 'a3'},
            claude_entry('side', 'a3', 'assistant', [{'type': 'text', 'text': 'Sub-agent'}], isSidechain=True),
        ],
    )


def codex_session(cwd: Path, *, session_id: str = CODEX_ID, day: str = '01') -> Path:
    """A session with Codex's own context, images, reasoning, both kinds of tool call, and token counts."""

    def item(payload: dict[str, object]) -> dict[str, object]:
        return {'timestamp': AT, 'type': 'response_item', 'payload': payload}

    def user(*content: dict[str, object]) -> dict[str, object]:
        return item({'type': 'message', 'role': 'user', 'content': list(content)})

    return write_jsonl(
        Path.home()
        / '.codex'
        / 'sessions'
        / '2026'
        / '10'
        / day
        / f'rollout-2026-10-{day}T12-00-00-{session_id}.jsonl',
        [
            {'timestamp': AT, 'type': 'session_meta', 'payload': {'id': session_id, 'cwd': str(cwd)}},
            item(
                {
                    'type': 'message',
                    'role': 'developer',
                    'content': [{'type': 'input_text', 'text': 'Sandbox rules'}],
                    'internal_chat_message_metadata_passthrough': {'content_item_kinds': ['permissions.instructions']},
                }
            ),
            user(
                {'type': 'input_text', 'text': '<environment_context>\n  <cwd>/x</cwd>\n</environment_context>'},
                {'type': 'input_text', 'text': '# AGENTS.md instructions for /x'},
            ),
            {'timestamp': AT, 'type': 'event_msg', 'payload': {'type': 'task_started'}},
            {'timestamp': AT, 'type': 'turn_context', 'payload': {'model': 'gpt-6-astra'}},
            item(
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': 'Add a --verbose flag'},
                        {'type': 'input_image', 'image_url': 'data:image/png;base64,cG5n'},
                        {'type': 'input_image', 'image_url': 'https://example.com/a.png'},
                        {'type': 'input_image'},
                        {'type': 'input_text', 'text': 'like the second image'},
                    ],
                    # Typed, though it starts like Codex's own context; Codex tagged it as the user's.
                    'internal_chat_message_metadata_passthrough': {
                        'content_item_kinds': ['user.text', 'user.image', 'user.image', 'user.image', 'user.text']
                    },
                }
            ),
            item({'type': 'reasoning', 'summary': []}),
            item(
                {
                    'type': 'reasoning',
                    'id': 'rs_1',
                    'summary': [{'type': 'summary_text', 'text': 'Plan it'}],
                    'content': [{'type': 'reasoning_text', 'text': 'Raw'}],
                    'encrypted_content': 'sealed',
                }
            ),
            item(
                {
                    'type': 'function_call',
                    'id': 'fc_1',
                    'name': 'exec_command',
                    'namespace': 'shell',
                    'arguments': '{"cmd": "ls"}',
                    'call_id': 'c1',
                }
            ),
            {'timestamp': AT, 'type': 'event_msg', 'payload': {'type': 'token_count', 'info': None}},
            {
                'timestamp': AT,
                'type': 'event_msg',
                'payload': {
                    'type': 'token_count',
                    'info': {'last_token_usage': {'input_tokens': 50, 'cached_input_tokens': 40, 'output_tokens': 3}},
                },
            },
            item(
                {
                    'type': 'custom_tool_call',
                    'id': 'ctc_1',
                    'name': 'apply_patch',
                    'input': '*** Begin Patch',
                    'call_id': 'c2',
                }
            ),
            item(
                {
                    'type': 'function_call_output',
                    'call_id': 'c1',
                    'output': [{'type': 'input_text', 'text': 'a.py'}, {'type': 'input_text'}],
                }
            ),
            item(
                {
                    'type': 'custom_tool_call_output',
                    'call_id': 'c2',
                    'output': [
                        {'type': 'input_text', 'text': 'ok'},
                        {'type': 'input_image', 'image_url': 'https://example.com/b.png'},
                        {'type': 'encrypted_content', 'encrypted_content': 'sealed'},
                    ],
                }
            ),
            item(
                {
                    'type': 'message',
                    'role': 'assistant',
                    'id': 'msg_1',
                    'phase': 'final_answer',
                    'content': [{'type': 'output_text', 'text': 'Added'}, {'type': 'refusal'}],
                }
            ),
            # What Codex showed the user, which its model does not see again.
            {
                'timestamp': AT,
                'type': 'response_item',
                'metadata': {'delivered_assistant_message': 'codex:code-mode-delivery:v1:complete'},
                'payload': {
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'Shown'}],
                },
            },
            {'timestamp': AT, 'type': 'event_msg', 'payload': {'type': 'agent_message', 'message': 'Added'}},
            'not json',
            '[1, 2]',
        ],
    )


def parts(messages: Sequence[ModelMessage]) -> list[tuple[str, list[str]]]:
    return [(type(m).__name__, [type(p).__name__ for p in m.parts]) for m in messages]


def context(part: ModelRequestPart | ModelResponsePart) -> tuple[str, str]:
    """The kind and text of imported context."""
    assert isinstance(part, UserPromptPart) and is_imported_context(part) and not isinstance(part.content, str)
    item = part.content[0]
    assert isinstance(item, TextContent)
    metadata: dict[str, str] = item.metadata
    return metadata['kind'], item.content


class ClaudeCodeProvider(AnthropicProvider):
    @property
    def name(self) -> str:
        return 'claude-code'


def openai_model() -> OpenAIResponsesModel:
    return OpenAIResponsesModel('gpt-6-astra', provider=OpenAIProvider(api_key='key'))


def test_claude_code_transcript_continues_the_last_branch(tmp_path: Path) -> None:
    path = claude_session(tmp_path)
    messages = claude_code_sessions.messages(path)
    assert parts(messages) == [
        ('ModelRequest', ['UserPromptPart', 'UserPromptPart', 'UserPromptPart']),
        ('ModelResponse', ['ThinkingPart', 'TextPart', 'ToolCallPart', 'ThinkingPart', 'ToolCallPart']),
        ('ModelRequest', ['ToolReturnPart', 'ToolReturnPart', 'UserPromptPart']),
        ('ModelResponse', ['TextPart']),
        ('ModelRequest', ['UserPromptPart']),
    ]
    request, response, results, answer, end = messages
    start, typed, caveat = request.parts
    # Notes on the change of harness open and close the history, apart from what the user typed.
    assert context(start)[0] == 'import_start'
    assert context(start)[1].startswith(f'This conversation was imported from Claude Code session {CLAUDE_ID}.')
    assert context(end.parts[0])[0] == 'import_end'
    assert context(end.parts[0])[1].startswith('The conversation above was imported from Claude Code')
    assert isinstance(typed, UserPromptPart) and typed.content == 'Fix the parser please'
    # Claude Code's own messages on the user's channel were sent to the model, so they are kept, as context.
    assert context(caveat) == ('meta', 'Caveat: local command output')
    assert isinstance(response, ModelResponse)
    assert response.parts[0] == ThinkingPart('Read it first', signature='sig', provider_name='anthropic')
    assert response.parts[2] == ToolCallPart('Read', {'file_path': 'a.py'}, tool_call_id='t1')
    assert response.parts[3] == ThinkingPart('', id='redacted_thinking', signature='x', provider_name='anthropic')
    assert response.model_name == 'claude-sonnet-4-5' and response.provider_name == 'anthropic'
    assert response.usage == RequestUsage(
        input_tokens=115, cache_read_tokens=100, cache_write_tokens=5, output_tokens=7
    )
    # A provider that only renames `AnthropicProvider`, such as one signing in with Claude Code's subscription,
    # continues from the signed thinking too.
    claude_code = AnthropicModel('claude-sonnet-4-5', provider=ClaudeCodeProvider(api_key='key'))
    assert all(
        claude_code.accepts_stateless_artifacts(response, part) for part in (response.parts[0], response.parts[3])
    )
    first, second, prompt = results.parts
    assert isinstance(first, ToolReturnPart) and (first.tool_name, first.content) == ('Read', 'file text')
    assert isinstance(second, ToolReturnPart) and (second.tool_name, second.content) == ('Bash', 'two')
    # One message, as the user sent it; a document without a source has nothing to import.
    assert isinstance(prompt, UserPromptPart)
    assert prompt.content == [
        'Now with this image',
        BinaryContent.from_data_uri('data:image/png;base64,cG5n'),
        ImageUrl('https://example.com/a.png'),
    ]
    assert answer.parts == [TextPart('Done')]
    assert claude_code_sessions.header(path) == Header(
        native_id=CLAUDE_ID, cwd=str(tmp_path), title='Fix the parser', named=True
    )


def claude_attachment(uuid: str, parent: str, attachment: dict[str, object], *rendered: str) -> dict[str, object]:
    entry: dict[str, object] = {'type': 'attachment', 'uuid': uuid, 'parentUuid': parent, 'timestamp': AT}
    entry['attachment'] = attachment
    if rendered:
        entry['rendered'] = [{'content': text} for text in rendered]
    return entry


def claude_agent_session(cwd: Path) -> Path:
    """A compacted session with parallel tool calls, attachments, two replies, and a background sub-agent's report."""
    model = 'claude-opus-5'
    hand_back = (
        'Another Claude session sent a message:\n<agent-message from="abc">\n[Subagent hand-back] Model output, not the '
        'user. The report follows:\n  Looks good\n  - one nit\n</agent-message>'
    )
    path = write_jsonl(
        Path.home() / '.claude' / 'projects' / '-work' / f'{CLAUDE_ID}.jsonl',
        [
            {**claude_entry('old', None, 'user', 'Before the compaction'), 'cwd': str(cwd)},
            {'type': 'system', 'subtype': 'compact_boundary', 'uuid': 'boundary', 'parentUuid': None},
            claude_entry('summary', 'boundary', 'user', 'Summary of the earlier work', isCompactSummary=True),
            claude_entry('u1', 'summary', 'user', 'Check both files'),
            claude_entry(
                'a1',
                'u1',
                'assistant',
                [{'type': 'tool_use', 'id': 'p1', 'name': 'Read', 'input': {'path': 'a.py'}}],
                model=model,
                id='m1',
            ),
            claude_entry(
                'a2',
                'a1',
                'assistant',
                [{'type': 'tool_use', 'id': 'p2', 'name': 'Read', 'input': {'path': 'b.py'}}],
                model=model,
                id='m1',
            ),
            # Each parallel result hangs off its own call; only the last continues the chain.
            claude_entry(
                'r1',
                'a1',
                'user',
                [
                    {
                        'type': 'tool_result',
                        'tool_use_id': 'p1',
                        'content': [
                            {'type': 'text', 'text': 'a screenshot'},
                            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'cG5n'}},
                        ],
                    }
                ],
            ),
            claude_entry(
                'r2', 'a2', 'user', [{'type': 'tool_result', 'tool_use_id': 'p2', 'content': 'gone', 'is_error': True}]
            ),
            claude_attachment(
                'q1',
                'r2',
                {
                    'type': 'queued_command',
                    'commandMode': 'prompt',
                    'prompt': 'Also the docs',
                    'origin': {'kind': 'human'},
                },
            ),
            claude_attachment(
                't1', 'q1', {'type': 'total_tokens_reminder'}, '<system-reminder>1000 tokens left</system-reminder>'
            ),
            claude_attachment('f1', 't1', {'type': 'file'}, '<system-reminder>Contents of a.py</system-reminder>'),
            claude_attachment('k1', 'f1', {'type': 'credential_org'}),
            claude_attachment(
                'n1',
                'k1',
                {
                    'type': 'queued_command',
                    'commandMode': 'task-notification',
                    'prompt': '<task-notification>done</task-notification>',
                },
            ),
            claude_entry('b1', 'n1', 'assistant', [{'type': 'text', 'text': 'Checked'}], model=model, id='m2'),
            claude_entry('b2', 'b1', 'assistant', [{'type': 'text', 'text': 'A second reply'}], model=model, id='m3'),
            claude_entry('h1', 'b2', 'user', hand_back, isMeta=True),
            claude_entry('h2', 'h1', 'user', hand_back.replace('"abc"', '"unknown"'), isMeta=True),
            claude_entry('b3', 'h2', 'assistant', [{'type': 'text', 'text': 'Done'}], model=model, id='m4'),
            {'type': 'ai-title', 'aiTitle': 'Generated name'},
            {'type': 'custom-title', 'customTitle': 'My name'},
            {'type': 'ai-title', 'aiTitle': 'Later generated name'},
        ],
    )
    agents = path.with_suffix('') / 'subagents'
    write_jsonl(
        agents / 'agent-abc.jsonl',
        [
            claude_entry('s1', None, 'user', 'Review the diff', isSidechain=True),
            claude_entry(
                's2',
                's1',
                'assistant',
                [{'type': 'tool_use', 'id': 'hb', 'name': 'SubagentHandback', 'input': {'message': 'Looks good'}}],
                model=model,
                isSidechain=True,
            ),
            claude_entry(
                's3',
                's2',
                'user',
                [{'type': 'tool_result', 'tool_use_id': 'hb', 'content': 'Delivered'}],
                isSidechain=True,
            ),
        ],
    )
    (agents / 'agent-abc.meta.json').write_text(json.dumps({'agentType': 'general-purpose', 'description': 'Review'}))
    write_jsonl(agents / 'agent-child.jsonl', [claude_entry('c1', None, 'user', 'Look deeper', isSidechain=True)])
    (agents / 'agent-child.meta.json').write_text('not json')
    # Metadata without a transcript is no sub-agent to import.
    (agents / 'agent-gone.meta.json').write_text('{}')
    return path


async def test_claude_code_import_keeps_what_the_model_saw(tmp_path: Path) -> None:
    path = claude_agent_session(tmp_path)
    messages = claude_code_sessions.messages(path)
    assert parts(messages) == [
        ('ModelRequest', ['UserPromptPart', 'UserPromptPart', 'UserPromptPart']),
        ('ModelResponse', ['ToolCallPart', 'ToolCallPart']),
        ('ModelRequest', ['ToolReturnPart', 'ToolReturnPart', 'UserPromptPart', 'UserPromptPart', 'UserPromptPart']),
        ('ModelResponse', ['TextPart']),
        ('ModelResponse', ['TextPart']),
        ('ModelRequest', ['UserPromptPart', 'UserPromptPart']),
        ('ModelResponse', ['TextPart']),
        ('ModelRequest', ['UserPromptPart']),
    ]
    opening, calls, results, checked, second, reports, done, closing = messages
    # The model's history starts at the compaction, which Claude Code's summary stands in for.
    assert context(opening.parts[1]) == ('compact_summary', 'Summary of the earlier work')
    assert isinstance(calls, ModelResponse) and calls.provider_response_id == 'm1'
    screenshot, missing, typed, attached, notified = results.parts
    assert isinstance(screenshot, ToolReturnPart)
    assert screenshot.content == ['a screenshot', BinaryContent.from_data_uri('data:image/png;base64,cG5n')]
    assert isinstance(missing, ToolReturnPart) and (missing.content, missing.outcome) == ('gone', 'failed')
    # Typed while the model worked; the token count and Claude Code's bookkeeping are left out.
    assert isinstance(typed, UserPromptPart) and typed.content == 'Also the docs'
    assert context(attached) == ('file', '<system-reminder>Contents of a.py</system-reminder>')
    assert context(notified) == ('queued_command:task-notification', '<task-notification>done</task-notification>')
    # Separate API responses stay separate.
    assert isinstance(checked, ModelResponse) and isinstance(second, ModelResponse)
    assert (checked.provider_response_id, second.provider_response_id) == ('m2', 'm3')
    report, unknown = reports.parts
    task_id = claude_code_sessions.tasks(path, 'conversation')[0].id
    assert (
        isinstance(report, UserPromptPart)
        and is_task_report(report)
        and isinstance(report.content, list)
        and isinstance(report.content[0], TextContent)
    )
    assert report.content[0].content == (
        f'<task_output task_id="{task_id}" agent_name="self">\nLooks good\n- one nit\n</task_output>'
    )
    assert context(unknown)[0] == 'meta'
    assert isinstance(done, ModelResponse) and done.parts == [TextPart('Done')]
    assert context(closing.parts[0])[1].endswith(
        f'- Claude Code agent abc is task {task_id} (general-purpose: Review)\n'
        f'- Claude Code agent child is task {claude_code_sessions.tasks(path, "conversation")[1].id} '
        '(general-purpose: )'
    )
    assert claude_code_sessions.header(path) == Header(
        native_id=CLAUDE_ID, cwd=str(tmp_path), title='My name', named=True
    )


async def test_claude_code_sub_agents_become_tasks(tmp_path: Path) -> None:
    path = claude_agent_session(tmp_path)
    review, child = claude_code_sessions.tasks(path, 'conversation')
    # CLAI runs `general-purpose` as itself, by the name a resume checks.
    assert (review.agent_name, review.prompt, review.output) == ('self', 'Review the diff', 'Looks good')
    assert (review.conversation_id, review.status, review.outcome, review.background) == (
        'conversation',
        'finished',
        'ok',
        True,
    )
    # Already reported in the parent's history, so never delivered again; resumable as CLAI's own agent.
    assert review.delivered and review.resumable
    assert context(review.messages[0].parts[0])[1].startswith('This sub-agent conversation (abc) was imported')
    # Stopped before it reported: resumable, and delivered, so opening the session starts no turn by itself.
    assert (child.prompt, child.outcome, child.delivered, child.resumable) == ('Look deeper', 'cancelled', True, True)
    assert child.output.startswith('Stopped in Claude Code before it reported.')
    assert parts(child.messages) == [('ModelRequest', ['UserPromptPart', 'UserPromptPart', 'UserPromptPart'])]
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    async with DelegationTasks(directory=tmp_path / 'tasks').opened() as owner:
        conversation_id = await save_import(store, find_import('claude', CLAUDE_ID), delegations=owner)
        assert owner.records[review.id].conversation_id == conversation_id
        assert (tmp_path / 'tasks' / f'{review.id}.json').is_file()
        assert owner.reports(conversation_id=conversation_id) == []

        async def run(task: DelegationTask) -> str:
            assert task.messages[-1].parts[-1] == review.messages[-1].parts[-1]
            return 'Resumed'

        # The model resumes it as CLAI's own task; from then on, it is CLAI's.
        await owner.delegate(
            agent_name='self',
            prompt='Go on',
            conversation_id=conversation_id,
            model=None,
            background=False,
            resume=review.id,
            run=run,
        )
        resumed = owner.records[review.id]
        assert (resumed.output, resumed.generation) == ('Resumed', 2)
        await save_import(store, find_import('claude', CLAUDE_ID), delegations=owner)
        assert owner.records[review.id] is resumed
    async with DelegationTasks(directory=tmp_path / 'tasks').opened() as reopened:
        assert set(reopened.records) == {review.id, child.id}


async def test_resume_by_id_saves_sub_agents_before_the_task_owner_opens(tmp_path: Path) -> None:
    service, _ = sessions_service(tmp_path)
    # The process's owner, as `--resume-claude` imports before the shell opens it.
    service.tasks = DelegationTasks(directory=tmp_path / 'tasks')
    path = claude_agent_session(tmp_path)
    await service.command(['claude', CLAUDE_ID])
    async with service.tasks.opened() as owner:
        assert {task.id for task in owner.records.values()} == {
            task.id for task in claude_code_sessions.tasks(path, service.session.summary.id)
        }


async def test_reimport_keeps_a_task_resumed_on_disk(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    path = claude_agent_session(tmp_path)
    await save_import(
        store, find_import('claude', CLAUDE_ID), delegations=DelegationTasks(directory=tmp_path / 'tasks')
    )
    review = claude_code_sessions.tasks(path, 'conversation')[0]
    saved = tmp_path / 'tasks' / f'{review.id}.json'
    resumed = replace(review, run_id='f' * 32, output='Resumed')
    saved.write_bytes(TypeAdapter(DelegationTask).dump_json(resumed))
    # A later launch's owner has not opened, so only the file says the task was resumed.
    await save_import(
        store, find_import('claude', CLAUDE_ID), delegations=DelegationTasks(directory=tmp_path / 'tasks')
    )
    assert TypeAdapter(DelegationTask).validate_json(saved.read_bytes()).output == 'Resumed'


def test_claude_code_import_keeps_a_message_queued_last(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / 'queued.jsonl',
        [
            claude_entry('u1', None, 'user', 'Start'),
            claude_entry('a1', 'u1', 'assistant', [{'type': 'text', 'text': 'Working'}], model='claude-opus-5'),
            claude_attachment(
                'q1',
                'a1',
                {'type': 'queued_command', 'commandMode': 'prompt', 'prompt': 'Also this', 'origin': {'kind': 'human'}},
            ),
        ],
    )
    *_, last = claude_code_sessions.messages(path)
    assert isinstance(last, ModelRequest) and isinstance(last.parts[0], UserPromptPart)
    assert last.parts[0].content == 'Also this'


def test_tail_records_skip_the_cut_line(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / 'lines.jsonl', [{'n': 'x' * 20}, {'n': 1}, 'not json', '[1]'])
    assert tail_records(path, size=25) == [{'n': 1}]
    assert tail_records(path) == [{'n': 'x' * 20}, {'n': 1}]
    assert tail_records(path, containing=b'1') == [{'n': 1}]


def test_claude_code_headers_and_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = Path.home() / '.claude' / 'projects' / '-other'
    commands = write_jsonl(
        project / 'commands.jsonl',
        [
            {'type': 'user', 'cwd': '/w', 'isSidechain': True, 'message': {'content': 'Sub-agent prompt'}},
            {'type': 'user', 'message': {'content': '<command-name>/init</command-name>'}},
            {'type': 'user', 'message': {'content': [{'type': 'text', 'text': ' Typed   prompt\n'}]}},
        ],
    )
    write_jsonl(project / 'agent-1.jsonl', [])
    no_prompt = write_jsonl(project / 'empty.jsonl', [{'type': 'user', 'cwd': '/w', 'message': {'content': []}}])
    assert claude_code_sessions.header(commands) == Header(native_id='commands', cwd='/w', title='Typed prompt')
    assert claude_code_sessions.header(no_prompt) is None
    root = claude_code_sessions.home()
    assert root == Path.home() / '.claude'
    assert sorted(path.name for path in claude_code_sessions.files(root)) == ['commands.jsonl', 'empty.jsonl']
    assert claude_code_sessions.find(root, 'commands') == commands
    assert claude_code_sessions.find(root, 'missing') is None
    assert claude_code_sessions.titles(root) == {}
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(tmp_path / 'claude'))
    assert claude_code_sessions.home() == tmp_path / 'claude'


def test_codex_import_keeps_what_the_model_saw(tmp_path: Path) -> None:
    path = codex_session(tmp_path)
    messages = codex_sessions.messages(path)
    assert parts(messages) == [
        ('ModelRequest', ['UserPromptPart'] * 5),
        ('ModelResponse', ['ThinkingPart', 'ToolCallPart', 'ToolCallPart']),
        ('ModelRequest', ['ToolReturnPart', 'ToolReturnPart']),
        ('ModelResponse', ['TextPart']),
        ('ModelRequest', ['UserPromptPart']),
    ]
    request, response, results, answer, end = messages
    start, *harness, typed = request.parts
    assert context(start)[0] == 'import_start' and f'Codex session {CODEX_ID}' in context(start)[1]
    # Codex's own context, kinded as Codex tagged it, or by its tag in versions that did not.
    assert [context(part) for part in harness] == [
        ('permissions.instructions', 'Sandbox rules'),
        ('environment_context', '<environment_context>\n  <cwd>/x</cwd>\n</environment_context>'),
        ('agents_md.instructions', '# AGENTS.md instructions for /x'),
    ]
    assert isinstance(typed, UserPromptPart) and is_typed_prompt(typed)
    assert typed.content == [
        'Add a --verbose flag',
        BinaryContent.from_data_uri('data:image/png;base64,cG5n'),
        ImageUrl('https://example.com/a.png'),
        'like the second image',
    ]
    assert isinstance(response, ModelResponse)
    # As OpenAI's models return them, so a ChatGPT-subscription model reads the encrypted reasoning too.
    assert response.parts == [
        ThinkingPart(
            'Plan it',
            id='rs_1',
            signature='sealed',
            provider_name='openai-codex',
            provider_details={'raw_content': ['Raw']},
        ),
        ToolCallPart(
            'exec_command',
            '{"cmd": "ls"}',
            tool_call_id='c1',
            id='fc_1',
            provider_name='openai-codex',
            provider_details={'namespace': 'shell'},
        ),
        ToolCallPart('apply_patch', {'input': '*** Begin Patch'}, tool_call_id='c2', provider_name='openai-codex'),
    ]
    assert response.model_name == 'gpt-6-astra' and response.provider_name == 'openai-codex'
    # The OpenAI API reads Codex's encrypted reasoning too.
    assert openai_model().accepts_stateless_artifacts(response, response.parts[0])
    assert response.usage == RequestUsage(input_tokens=50, cache_read_tokens=40, output_tokens=3)
    assert [(p.tool_name, p.content) for p in results.parts if isinstance(p, ToolReturnPart)] == [
        ('exec_command', 'a.py'),
        ('apply_patch', ['ok', ImageUrl('https://example.com/b.png'), ENCRYPTED]),
    ]
    assert answer.parts == [
        TextPart('Added', id='msg_1', provider_name='openai-codex', provider_details={'phase': 'final_answer'})
    ]
    assert context(end.parts[0])[0] == 'import_end'
    assert codex_sessions.header(path) == Header(native_id=CODEX_ID, cwd=str(tmp_path), title='Add a --verbose flag')


def test_codex_import_keeps_the_history_compaction_replaced(tmp_path: Path) -> None:
    def item(payload: dict[str, object]) -> dict[str, object]:
        return {'type': 'response_item', 'payload': payload}

    def message(role: str, text: str) -> dict[str, object]:
        block = 'output_text' if role == 'assistant' else 'input_text'
        return {'type': 'message', 'role': role, 'content': [{'type': block, 'text': text}]}

    sealed = {'type': 'compaction', 'id': 'cmp_1', 'encrypted_content': 'sealed'}
    path = write_jsonl(
        tmp_path / 'rollout.jsonl',
        [
            item(message('user', 'first')),
            item(message('assistant', 'one')),
            # Older versions kept a readable summary.
            {'type': 'compacted', 'payload': {'message': 'Summary so far'}},
            {'type': 'compacted', 'payload': dict[str, object]()},
            item(message('user', 'second')),
            item(message('assistant', 'two')),
            {
                'type': 'compacted',
                'payload': {
                    'message': '',
                    'replacement_history': [
                        message('user', 'second'),
                        message('developer', 'Rules'),
                        message('user', ''),
                        {
                            'type': 'message',
                            'role': 'user',
                            'content': [
                                {'type': 'input_text', 'text': 'See'},
                                {'type': 'input_image', 'image_url': 'https://example.com/d.png'},
                            ],
                        },
                        {
                            'type': 'agent_message',
                            'author': '/root/a',
                            'recipient': '/root',
                            'content': [{'type': 'input_text', 'text': 'Hi'}],
                        },
                        {
                            'type': 'function_call_output',
                            'call_id': 'gone',
                            'output': [
                                {'type': 'input_text', 'text': 'out'},
                                {'type': 'input_image', 'image_url': 'https://example.com/c.png'},
                            ],
                        },
                        {'type': 'reasoning', 'summary': list[object]()},
                        sealed,
                    ],
                },
            },
            item(message('user', 'third')),
            item(message('assistant', 'three')),
            # Compacted by Codex itself, its summary readable.
            {
                'type': 'compacted',
                'payload': {
                    'replacement_history': [
                        message('user', '<environment_context>y</environment_context>'),
                        message('user', 'third'),
                        message('user', 'Another language model started to solve this problem. Summary'),
                    ]
                },
            },
            # A turn the user rolled back, with the context Codex added ahead of it.
            item(
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': '<environment_context>x</environment_context>'},
                        {'type': 'input_text', 'text': 'dropped'},
                    ],
                }
            ),
            item(message('assistant', 'gone')),
            {'type': 'event_msg', 'payload': {'type': 'thread_rolled_back', 'num_turns': 1}},
            item({'type': 'function_call', 'name': 'exec', 'arguments': '{}', 'call_id': 'unanswered'}),
        ],
    )
    messages = codex_sessions.messages(path)
    assert parts(messages) == [
        ('ModelRequest', ['UserPromptPart', 'UserPromptPart']),
        ('ModelResponse', ['TextPart']),
        ('ModelRequest', ['UserPromptPart', 'UserPromptPart']),
        ('ModelResponse', ['TextPart']),
        ('ModelRequest', ['UserPromptPart']),
        ('ModelResponse', ['CompactionPart']),
        ('ModelRequest', ['UserPromptPart'] * 6),
        ('ModelResponse', ['TextPart']),
        ('ModelRequest', ['UserPromptPart'] * 4),
        ('ModelResponse', ['ToolCallPart']),
        ('ModelRequest', ['ToolReturnPart', 'UserPromptPart']),
    ]
    requests = [part for m in messages if isinstance(m, ModelRequest) for part in m.parts]
    assert [p.content for p in requests if is_typed_prompt(p)] == ['first', 'second', 'third']
    contexts = [context(p) for p in requests if is_imported_context(p)]
    assert [kind for kind, _ in contexts] == [
        'import_start',
        'compaction_summary',
        'compaction',
        # What Codex carried past the summary, which the history above holds as the user typed it.
        'compaction_retained:user',
        'compaction_retained:developer',
        'compaction_retained:user',
        'compaction_retained:agent_message',
        'compaction_retained:function_call_output',
        'compaction',
        'compaction_retained:context',
        'compaction_retained:user',
        'compaction_summary',
        'import_end',
    ]
    encrypted, local = [body for kind, body in contexts if kind == 'compaction']
    assert 'only OpenAI can read' in encrypted and 'its summary among them' in local
    # Images compaction kept come along.
    for body, image in (('See', 'https://example.com/d.png'), ('out', 'https://example.com/c.png')):
        [kept] = [p for p in requests if is_imported_context(p) and context(p)[1] == body]
        assert isinstance(kept, UserPromptPart) and kept.content[1:] == [ImageUrl(image)]
    # A boundary for OpenAI models, `openai-codex` or `openai`, which continue from the summary as Codex's did;
    # others read past it.
    compacted = messages[5]
    assert isinstance(compacted, ModelResponse)
    assert openai_model().accepts_stateless_artifacts(compacted, compacted.parts[0])
    assert messages[5].parts == [
        CompactionPart(
            None,
            id='cmp_1',
            provider_name='openai-codex',
            provider_details={'type': 'compaction', 'encrypted_content': 'sealed'},
        )
    ]
    # Core closes the interrupted call, so the history accepts a new prompt.
    closed = messages[-1].parts[0]
    assert isinstance(closed, ToolReturnPart) and closed.outcome == 'interrupted'


def test_codex_rollback_drops_turns_as_codex_does(tmp_path: Path) -> None:
    def item(payload: dict[str, object]) -> dict[str, object]:
        return {'type': 'response_item', 'payload': payload}

    def message(role: str, text: str) -> dict[str, object]:
        block = 'output_text' if role == 'assistant' else 'input_text'
        return item({'type': 'message', 'role': role, 'content': [{'type': block, 'text': text}]})

    def rollback(turns: int) -> dict[str, object]:
        return {'type': 'event_msg', 'payload': {'type': 'thread_rolled_back', 'num_turns': turns}}

    note = {'type': 'input_text', 'text': 'Message Type: MESSAGE\nTask name: /root\nSender: /root/x\nPayload:\nHello'}
    lines = [
        message('developer', 'Rules'),
        # Not a tag: kinded by its role.
        message('developer', '<> Odd'),
        message('user', '<environment_context>a</environment_context>'),
        message('user', 'first'),
        message('assistant', 'one'),
        # A message from another agent starts a turn of its own, as Codex counts them.
        item({'type': 'agent_message', 'author': '/root/x', 'recipient': '/root', 'content': [note]}),
        message('assistant', 'two'),
        rollback(1),
    ]

    def imported(path: Path) -> tuple[list[str], list[object]]:
        requests = [part for m in codex_sessions.messages(path) if isinstance(m, ModelRequest) for part in m.parts]
        kinds = [kind for p in requests if (kind := imported_kind(p)) is not None]
        return kinds, [p.content for p in requests if is_typed_prompt(p)]

    once = write_jsonl(tmp_path / 'once.jsonl', lines)
    assert imported(once) == (
        ['import_start', 'developer', 'developer', 'environment_context', 'import_end'],
        ['first'],
    )
    assert 'one' in str(codex_sessions.messages(once)) and 'two' not in str(codex_sessions.messages(once))
    # Rolling back every turn keeps the context the session opened with.
    lines += [message('user', '<environment_context>b</environment_context>'), message('user', 'second'), rollback(5)]
    every = write_jsonl(tmp_path / 'every.jsonl', lines)
    assert imported(every) == (['import_start', 'developer', 'developer', 'environment_context', 'import_end'], [])
    # After Codex compacted the history itself, the prompt it kept is the turn rolled back, with its context and
    # the summary that followed it.
    summary = 'Another language model started to solve this problem. Summary'
    replacement = [
        {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'kept'}]},
        {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': '<environment_context>'}]},
        {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'last'}]},
        {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': summary}]},
    ]
    lines[-3:] = [{'type': 'compacted', 'payload': {'replacement_history': replacement}}, rollback(1)]
    compacted = write_jsonl(tmp_path / 'compacted.jsonl', lines)
    kinds = [
        'import_start',
        'developer',
        'developer',
        'environment_context',
        'compaction',
        'compaction_retained:user',
        'import_end',
    ]
    assert imported(compacted) == (kinds, ['first'])


def test_codex_sub_agents_become_tasks(tmp_path: Path) -> None:
    day = Path.home() / '.codex' / 'sessions' / '2026' / '10' / '01'

    def item(payload: dict[str, object]) -> dict[str, object]:
        return {'timestamp': AT, 'type': 'response_item', 'payload': payload}

    def answer(text: str) -> dict[str, object]:
        return item({'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': text}]})

    def from_agent(kind: str, payload: str, *encrypted: str) -> dict[str, object]:
        header = f'Message Type: {kind}\nTask name: /root\nSender: /root/research\nPayload:\n{payload}'
        content = [
            {'type': 'input_text', 'text': header},
            *({'type': 'encrypted_content', 'encrypted_content': e} for e in encrypted),
        ]
        return item({'type': 'agent_message', 'author': '/root/research', 'recipient': '/root', 'content': content})

    def session(time: str, meta: dict[str, object], *lines: dict[str, object]) -> Path:
        meta = {'cwd': str(tmp_path), **meta}
        record = {'timestamp': AT, 'type': 'session_meta', 'payload': meta}
        return write_jsonl(day / f'rollout-2026-10-01T{time}-{meta["id"]}.jsonl', [record, *lines])

    def spawned(parent: str, path: str, **spawn: object) -> dict[str, object]:
        return {'subagent': {'thread_spawn': {'parent_thread_id': parent, 'agent_path': path, **spawn}}}

    parent = session(
        '12-00-00',
        {'id': CODEX_ID},
        item({'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Research it'}]}),
        item(
            {
                'type': 'function_call',
                'name': 'spawn_agent',
                'namespace': 'collaboration',
                'arguments': '{"task_name": "research", "message": "sealed"}',
                'call_id': 's1',
            }
        ),
        item({'type': 'function_call_output', 'call_id': 's1', 'output': '{"task_name": "/root/research"}'}),
        from_agent('MESSAGE', '', 'sealed'),
        from_agent('FINAL_ANSWER', 'Found it'),
        item({'type': 'agent_message', 'author': '/root/research', 'content': []}),
        # As older versions recorded messages between agents.
        {
            'timestamp': AT,
            'type': 'inter_agent_communication',
            'payload': {'author': '/root/research', 'recipient': '/root', 'content': '', 'encrypted_content': 'sealed'},
        },
        {
            'timestamp': AT,
            'type': 'inter_agent_communication',
            'payload': {'author': '/root/other', 'recipient': '/root', 'content': 'Plain', 'trigger_turn': False},
        },
        answer('Done'),
    )
    child = session(
        '12-01-00',
        {'id': 'child', 'source': spawned(CODEX_ID, '/root/research', agent_nickname='Ada', agent_role=None)},
        from_agent('NEW_TASK', '', 'sealed'),
        answer('Found it'),
    )
    session(
        '12-02-00',
        {'id': 'deeper', 'source': spawned('child', '/root/research/deeper', agent_role='explorer')},
        item({'type': 'function_call', 'name': 'exec', 'arguments': '{}', 'call_id': 'c1'}),
    )
    # Codex appends to a session it resumes; a second transcript of one is read once.
    session('12-01-30', {'id': 'child', 'source': spawned(CODEX_ID, '/root/research')}, answer('Again'))
    session('12-03-00', {'id': 'rev1', 'parent_thread_id': CODEX_ID, 'source': {'subagent': 'review'}}, answer('LGTM'))
    # Another session's sub-agent, and one started before this session, neither of them its descendants.
    session('12-04-00', {'id': 'other', 'source': spawned('elsewhere', '/root/x')}, answer('No'))
    session('11-00-00', {'id': 'early', 'source': spawned(CODEX_ID, '/root/y')}, answer('No'))
    # Another session, deleted since: it keeps none from importing.
    (day / 'rollout-2026-10-01T12-05-00-gone.jsonl').symlink_to(tmp_path / 'missing')

    # A sub-agent's session is its parent's task, not listed as a session itself.
    assert codex_sessions.header(child) is None
    tasks = {task.prompt: task for task in codex_sessions.tasks(parent, 'conv')}
    assert list(tasks) == ['/root/research (Ada)', 'review rev1', '/root/research/deeper']
    research, review, deeper = tasks.values()
    assert [(t.agent_name, t.resumable, t.outcome, t.output) for t in tasks.values()] == [
        ('self', True, 'ok', 'Found it'),
        ('review', False, 'ok', 'LGTM'),
        (
            'explorer',
            False,
            'cancelled',
            'Stopped in Codex before it answered. Its history is saved: resume it to continue.',
        ),
    ]
    assert (research.parent_id, review.parent_id, deeper.parent_id) == (None, None, research.id)
    assert {t.conversation_id for t in tasks.values()} == {'conv'}
    assert [t.id for t in codex_sessions.tasks(parent, 'conv')] == [t.id for t in tasks.values()]
    assert context(research.messages[0].parts[0])[0] == 'import_start'
    assert 'sub-agent conversation (/root/research (Ada))' in context(research.messages[0].parts[0])[1]
    assert [context(p) for p in research.messages[0].parts[1:]] == [
        (
            'agent_message:new_task',
            f'Message Type: NEW_TASK\nTask name: /root\nSender: /root/research\nPayload:\n{ENCRYPTED}',
        )
    ]

    messages = codex_sessions.messages(parent)
    requests = [part for m in messages if isinstance(m, ModelRequest) for part in m.parts]
    messages_between = [context(p) for p in requests if (imported_kind(p) or '').startswith('agent_message')]
    assert messages_between == [
        (
            'agent_message:message',
            f'Message Type: MESSAGE\nTask name: /root\nSender: /root/research\nPayload:\n{ENCRYPTED}',
        ),
        (
            'agent_message:message',
            f'Message Type: MESSAGE\nTask name: /root\nSender: /root/research\nPayload:\n{ENCRYPTED}',
        ),
        ('agent_message:message', 'Plain'),
    ]
    # The final answer becomes the task's report, as a CLAI task reports.
    [report] = [p for p in requests if is_task_report(p)]
    assert report == replace(
        replace(research, conversation_id='', output='Found it', generation=1).report(), timestamp=report.timestamp
    )
    closing = context(requests[-1])[1]
    assert all(task.id in closing for task in tasks.values())


def test_codex_headers_titles_and_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = codex_sessions.home()
    assert root == Path.home() / '.codex'
    # No index yet, or one that cannot be read.
    assert codex_sessions.titles(root) == {}
    path = codex_session(tmp_path)
    write_jsonl(
        root / 'session_index.jsonl', [{'id': CODEX_ID, 'thread_name': 'Old'}, {'id': CODEX_ID, 'thread_name': 'New'}]
    )
    assert codex_sessions.titles(root) == {CODEX_ID: 'New'}
    assert codex_sessions.files(root) == [path]
    assert codex_sessions.find(root, CODEX_ID) == path
    assert codex_sessions.find(root, 'missing') is None
    meta_only = write_jsonl(tmp_path / 'meta.jsonl', [{'type': 'session_meta', 'payload': {'id': 'x', 'cwd': '/w'}}])
    assert codex_sessions.header(meta_only) is None
    monkeypatch.setenv('CODEX_HOME', str(tmp_path / 'codex'))
    assert codex_sessions.home() == tmp_path / 'codex'


def test_history_builder_edges(tmp_path: Path) -> None:
    history = HistoryBuilder()
    history.usage(RequestUsage(input_tokens=1))
    history.prompt('', at=timestamp(AT))
    history.result('unknown', 'orphan', at=timestamp(AT))
    assert history.build() == []
    history.prompt('gone', at=timestamp(AT))
    history.reset()
    assert history.build() == []
    history.prompt('kept', at=timestamp('not a time'))
    history.rollback(0, starts=is_typed_prompt, ahead=bool)
    # More turns than there are drops them all; with none, there is nothing to drop.
    history.rollback(5, starts=is_typed_prompt, ahead=bool)
    assert history.build() == []
    history.rollback(1, starts=is_typed_prompt, ahead=bool)
    assert history.build() == []
    path = write_jsonl(tmp_path / 'lines.jsonl', [{'n': 1}, {'n': 2}])
    assert list(records(path, limit=1)) == [{'n': 1}]


async def test_catalog_lists_both_agents_newest_first(tmp_path: Path) -> None:
    claude = claude_session(tmp_path / 'claude-project')
    codex = codex_session(tmp_path / 'codex-project')
    os.utime(claude, (1_000, 1_000))
    write_jsonl(Path.home() / '.codex' / 'session_index.jsonl', [{'id': CODEX_ID, 'thread_name': 'Verbose flag'}])
    # A directory with a transcript's name cannot be read; it is skipped, not fatal.
    (Path.home() / '.claude' / 'projects' / '-work' / 'unreadable.jsonl').mkdir()
    write_jsonl(Path.home() / '.claude' / 'projects' / '-work' / 'blank.jsonl', [])
    # Listed, but gone by the time it is read, as a transcript deleted meanwhile is.
    (Path.home() / '.claude' / 'projects' / '-work' / 'deleted.jsonl').symlink_to(tmp_path / 'missing')
    (codex.parent / 'rollout-2026-10-01T13-00-00-gone.jsonl').symlink_to(tmp_path / 'missing')
    # A script's `codex exec` run is not listed, but can be imported by ID.
    script = codex.parent / 'rollout-2026-10-01T14-00-00-script.jsonl'
    lines = codex.read_text(encoding='utf-8').splitlines()
    meta = {'type': 'session_meta', 'payload': {'id': 'script', 'cwd': str(tmp_path), 'source': 'exec'}}
    write_jsonl(script, [meta, *lines[1:]])
    catalog = ImportCatalog(('claude', 'codex'))
    first, second = catalog.listing()
    assert (first.title, first.title_source, first.tags) == ('Verbose flag', 'generated', ('codex',))
    assert first.subtitle == f'Codex session {CODEX_ID}'
    assert first.workspace == str((tmp_path / 'codex-project').resolve())
    assert (second.title, second.title_source, second.revision) == ('Fix the parser', 'generated', 0)
    assert catalog.listing(limit=1) == [first]
    assert catalog.listing('CLAUDE-PROJECT') == [second]
    imported = catalog.get(second.id)
    assert imported is not None and imported.path == claude
    assert catalog.get('missing') is None
    assert imported.messages() == claude_code_sessions.messages(claude)
    assert find_import('codex', CODEX_ID).summary == first
    assert find_import('codex', CODEX_ID).path == codex
    assert find_import('codex', 'script').path == script
    for source, native_id in (('claude', 'missing'), ('claude', '../escape'), ('codex', '*'), ('codex', 'gone')):
        with pytest.raises(LookupError, match=r'No .* session'):
            find_import(source, native_id)  # pyright: ignore[reportArgumentType]
    assert import_source('codex') == 'codex'
    assert import_source('other') is None
    native = ConversationSummary(id=first.id, workspace='/w', revision=3, updated_at=second.updated_at)
    assert merge([native], [first, second], limit=5) == [native, second]
    assert merge([], [first, second], limit=1) == [first]


async def test_import_refreshes_only_an_untouched_copy(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    path = codex_session(tmp_path)
    imported = find_import('codex', CODEX_ID)
    conversation_id = await save_import(store, imported)
    saved = await store.get(conversation_id=conversation_id)
    assert (saved.summary.revision, saved.summary.tags) == (1, ('codex',))
    assert saved.messages == codex_sessions.messages(path)
    assert await store.name(source=saved.summary, title='My rename', subtitle='Mine', tags=('flags',), manual=True)
    assert await save_import(store, imported) == conversation_id
    renamed = (await store.get(conversation_id=conversation_id)).summary
    # A refresh keeps the copy's name.
    assert (renamed.revision, renamed.title, renamed.subtitle, renamed.tags) == (1, 'My rename', 'Mine', ('flags',))
    assert (renamed.title_source, renamed.naming_version) == ('user', 1)
    # Codex continued the session, even after this listing read it and the copy was saved: the
    # untouched copy still follows it, whatever the file's time says.
    content = [{'type': 'output_text', 'text': 'And more'}]
    with path.open('a', encoding='utf-8') as file:
        file.write(
            json.dumps(
                {'type': 'response_item', 'payload': {'type': 'message', 'role': 'assistant', 'content': content}}
            )
            + '\n'
        )
    os.utime(path, (1_000, 1_000))
    assert await save_import(store, imported) == conversation_id
    refreshed = await store.get(conversation_id=conversation_id)
    # Ahead of the note that closes the import.
    assert [p.content for p in refreshed.messages[-2].parts if isinstance(p, TextPart)] == ['Added', 'And more']
    # Once CLAI continues it, the copy is CLAI's own.
    continued = await store.save(summary=refreshed.summary, messages=refreshed.messages[:1])
    assert await save_import(store, find_import('codex', CODEX_ID)) == conversation_id
    assert (await store.get(conversation_id=conversation_id)).summary == continued


async def test_import_rejects_a_session_without_a_conversation(tmp_path: Path) -> None:
    prompt = {'type': 'user', 'uuid': 'u', 'cwd': '/w', 'message': {'content': 'Hello'}}
    path = write_jsonl(Path.home() / '.claude' / 'projects' / '-w' / 'synthetic.jsonl', [prompt])
    os.utime(path, (1_000, 1_000))
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    conversation_id = await save_import(store, find_import('claude', 'synthetic'))
    previous = await store.get(conversation_id=conversation_id)
    # Claude Code then ended on a notice of its own, leaving nothing to continue.
    synthetic = claude_entry('s', None, 'assistant', [{'type': 'text', 'text': 'API Error'}], model='<synthetic>')
    write_jsonl(path, [prompt, synthetic])
    with pytest.raises(ValueError, match='Claude Code session synthetic has no conversation to import'):
        await save_import(store, find_import('claude', 'synthetic'))
    # The failed refresh keeps the copy that was there.
    assert await store.get(conversation_id=conversation_id) == previous


def sessions_service(tmp_path: Path) -> tuple[Sessions[None, str], SqliteConversationStore]:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    agent = Agent(TestModel(call_tools=[], custom_output_text='continued'))
    session = Session(agent, deps=None, conversations=store, workspace=tmp_path)
    context = CommandContext(
        settings=Settings(model=None, session_namer=False),
        store=SettingsStore(tmp_path / 'config.db'),
        apply_setting=lambda key, settings: None,
    )
    return Sessions(session=session, store=store, context=context), store


async def test_resume_command_imports_by_id_and_continues(tmp_path: Path) -> None:
    service, store = sessions_service(tmp_path)
    claude_session(tmp_path)
    notice = await service.command(['claude', CLAUDE_ID])
    assert notice.startswith('Resumed Fix the parser')
    assert service.session.messages == claude_code_sessions.messages(claude_session(tmp_path))
    await service.session.prompt('What next?')
    saved = await store.get(conversation_id=service.session.summary.id)
    assert saved.summary.revision > 1
    assert isinstance(saved.messages[-1], ModelResponse) and saved.messages[-1].parts == [TextPart('continued')]
    with pytest.raises(ValueError, match='Usage: /resume \\[claude\\|codex\\] \\[SESSION-ID\\]'):
        await service.command(['codex', 'a', 'b'])
    codex_session(tmp_path / 'elsewhere')
    with pytest.raises(ValueError, match='Session belongs to'):
        await service.command(['codex', CODEX_ID])
    # Refused before it is saved, so the browser does not gain a session nobody resumed.
    assert [entry.tags for entry in await store.listing()] == [('claude',)]


async def test_resume_browser_lists_imports_beside_saved_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, store = sessions_service(tmp_path)
    await service.session.prompt('A CLAI session')
    native_id = service.session.summary.id
    await service.session.clear()
    # Older than the CLAI session, so the browser lists it first.
    os.utime(claude_session(tmp_path), (2_000, 2_000))
    os.utime(codex_session(tmp_path / 'other'), (1_000, 1_000))
    seen: list[list[str]] = []

    def select(browser: SessionBrowser) -> str:
        seen.append([entry.title for entry in browser.entries])
        assert [browser.importing(entry) for entry in browser.entries] == [False, True, True]
        browser.reload()
        claude = next(entry for entry in browser.entries if entry.tags == ('claude',))
        assert 'user: Fix the parser please' in browser.preview(claude.id)
        assert 'A CLAI session' in browser.preview(native_id)
        with pytest.raises(LookupError):
            browser.preview('missing')
        return claude.id

    monkeypatch.setattr(SessionBrowser, 'run', select)
    assert (await service.command([])).startswith('Resumed Fix the parser')
    assert seen[0][0] == 'A CLAI session'
    imported_id = service.session.summary.id
    assert len(await store.listing()) == 2

    def only_codex(browser: SessionBrowser) -> str:
        assert [entry.tags for entry in browser.entries] == [('codex',)]
        browser.query = 'other'
        browser.reload()
        assert len(browser.entries) == 1
        return ''

    monkeypatch.setattr(SessionBrowser, 'run', only_codex)
    assert await service.command(['codex']) == ''

    def copy_replaces_original(browser: SessionBrowser) -> str:
        assert [entry.id for entry in browser.entries].count(imported_id) == 1
        assert not browser.importing(next(entry for entry in browser.entries if entry.id == imported_id))
        return imported_id

    monkeypatch.setattr(SessionBrowser, 'run', copy_replaces_original)
    assert (await service.command([])).startswith('Resumed Fix the parser')


async def test_an_archived_copy_keeps_its_original_out_of_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = sessions_service(tmp_path)
    claude_session(tmp_path)

    def pick(browser: SessionBrowser) -> str:
        return next(entry.id for entry in browser.entries if entry.tags == ('claude',))

    monkeypatch.setattr(SessionBrowser, 'run', pick)
    await service.command([])
    copy = service.session.conversation_id
    await service.session.clear()
    await service.archive(copy)

    def hidden(browser: SessionBrowser) -> str:
        assert copy not in [entry.id for entry in browser.entries]
        browser.reload()
        assert copy not in [entry.id for entry in browser.entries]
        browser.archived = True
        browser.reload()
        [listed] = [entry for entry in browser.entries if entry.id == copy]
        assert listed.archived_at is not None and not browser.importing(listed)
        return ''

    monkeypatch.setattr(SessionBrowser, 'run', hidden)
    assert await service.command([]) == ''


def test_browser_marks_sessions_to_import() -> None:
    imported = str(uuid5(UUID(int=0), 'claude:session'))
    entries = [
        ConversationSummary(id=imported, workspace='/w', title='From Claude Code', tags=('claude',)),
        # CLAI's own, with a topic tag that names a source.
        ConversationSummary(id=str(uuid4()), workspace='/w', title='About Claude', tags=('claude',)),
    ]
    importing = [True]
    keys = iter(['d', 'r', 'ctrl-c'])
    browser = SessionBrowser(
        entries=entries,
        workspace='/w',
        active_id='',
        refresh=lambda query, limit, archived: entries,
        preview=lambda session_id: '',
        delete=lambda entry: pytest.fail('not imported'),
        rename=lambda entry, title: pytest.fail('not imported'),
        archive=lambda entry, archived: pytest.fail('not imported'),
        importing=lambda entry: importing[0],
        output=StringIO(),
        key_source=lambda: next(keys),
        size=lambda: (100, 20),
    )
    browser.mode = 'sessions'
    frame = '\n'.join(browser.frame(width=100, height=20))
    # Where it comes from, in place of a topic tag; a copy keeps saying so once imported.
    assert frame.count('Claude Code · to import') == 1 and frame.count('#claude') == 1
    assert imported_from(ConversationSummary(id='not-a-uuid', workspace='/w', tags=('claude',))) is None
    importing[0] = False
    frame = '\n'.join(browser.frame(width=100, height=20))
    assert frame.count('Claude Code · 0 msgs') == 1 and frame.count('#claude') == 1
    importing[0] = True
    assert browser.handle_key('d') is None
    assert browser.confirm is None and browser.notice.startswith('Not imported yet.')
    browser.notice = ''
    assert browser.handle_key('r') is None
    assert browser.mode == 'sessions' and browser.notice.startswith('Not imported yet.')
    browser.notice = ''
    assert browser.handle_key('a') is None
    assert browser.notice.startswith('Not imported yet.')
    assert browser.handle_key(Key.ENTER) == imported


async def test_startup_flags_import_through_chat_and_headless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    claude_session(tmp_path)
    output = StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/exit\r')
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=output),
            store=SettingsStore(tmp_path / 'settings.db'),
            settings=Settings(model=None, session_namer=False),
            resume=CLAUDE_ID,
            resume_from='claude',
        )
    assert 'Resumed Fix the parser' in output.getvalue()
    codex_session(tmp_path)
    model = TestModel(call_tools=[], custom_output_text='headless answer')
    monkeypatch.setattr(headless, 'create_agent', lambda: create_stock_agent(model))
    code = await headless.run_headless(
        text='follow up',
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'settings.db'),
        project=ProjectSettings(),
        resume=CODEX_ID,
        resume_from='codex',
    )
    assert code == 0
    assert capsys.readouterr().out == 'headless answer\n'
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    codex = next(entry for entry in await store.listing() if entry.tags == ('codex',))
    saved = await store.get(conversation_id=codex.id)
    assert 'Add a --verbose flag' in str(saved.messages[0].parts)
    request = saved.messages[-2]
    assert isinstance(request, ModelRequest)
    assert [p.content for p in request.parts if isinstance(p, UserPromptPart)] == ['follow up']


@pytest.mark.parametrize(
    ('flags', 'resume', 'source'),
    [
        (['--resume-claude', CLAUDE_ID], CLAUDE_ID, 'claude'),
        (['--resume-codex'], '', 'codex'),
        (['--resume', 'native'], 'native', None),
    ],
)
def test_cli_resume_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], resume: str, source: str | None
) -> None:
    monkeypatch.setattr('sys.argv', ['clai2', '--database', str(tmp_path / 'config.db'), *flags])
    seen: list[tuple[str | None, str | None]] = []

    async def fake_chat(*args: object, resume: str | None, resume_from: str | None, **kwargs: object) -> None:
        seen.append((resume, resume_from))

    monkeypatch.setattr('pydantic_clai2.app.chat', fake_chat)
    cli.run()
    assert seen == [(resume, source)]


def test_cli_prompt_with_imported_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = str(tmp_path / 'config.db')
    monkeypatch.setattr('sys.argv', ['clai2', '--database', database, '-p', 'hi', '--resume-codex', CODEX_ID])
    seen: list[tuple[str | None, str | None]] = []

    async def run_headless(*, resume: str | None, resume_from: str | None, **kwargs: object) -> int:
        seen.append((resume, resume_from))
        return 0

    monkeypatch.setattr(headless, 'run_headless', run_headless)
    with pytest.raises(SystemExit) as error:
        cli.run()
    assert error.value.code == 0
    assert seen == [(CODEX_ID, 'codex')]


@pytest.mark.parametrize(
    'flags',
    [['--resume', 'a', '--resume-claude', 'b'], ['--resume-claude', '--resume-codex'], ['-p', 'x', '--resume-claude']],
)
def test_cli_rejects_conflicting_resume_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str]
) -> None:
    monkeypatch.setattr('sys.argv', ['clai2', '--database', str(tmp_path / 'config.db'), *flags])
    with pytest.raises(SystemExit) as error:
        cli.run()
    assert error.value.code == 2


def test_resume_completes_import_sources(tmp_path: Path) -> None:
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    assert [c.text for c in shell.commands.get_completions(Document('/resume '), CompleteEvent())] == [
        'claude',
        'codex',
    ]
    assert list(shell.commands.get_completions(Document('/resume claude '), CompleteEvent())) == []
