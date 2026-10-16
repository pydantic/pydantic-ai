"""Exercise real module reloads in a process isolated from pytest's imported class identities."""

import asyncio
import io
import sys
from dataclasses import replace
from pathlib import Path
from typing import Generic, TypeVar

import prompt_toolkit
import pytest
from prompt_toolkit.completion import CompleteEvent, Completer
from prompt_toolkit.document import Document
from rich.console import Console
from rich.text import Text

import pydantic_clai2
from pydantic_ai import Agent, ModelRequestContext, RunContext, models
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2 import DEFAULT_PLUGINS, chat
from pydantic_clai2._app import STOCK_PLUGINS
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime._session import StockAgent
from pydantic_clai2.ui.rendering import theme

models.ALLOW_MODEL_REQUESTS = False
PromptT = TypeVar('PromptT')


async def main(root: Path, mode: str) -> None:
    package = root / 'pydantic_clai2'
    app = package / '_app.py'
    original = app.read_text()
    updated = original.replace("prompt_async('> ')", "prompt_async('updated> ')")
    updated = updated.replace("'Goodbye.'", "'Updated goodbye.'")
    commands = package / 'commands.py'
    commands.write_text(commands.read_text().replace('Use /help.', 'Use updated /help.'))
    if mode == 'new_imports':
        commands.write_text(commands.read_text() + '\nRELOAD_MARKER = "source graph"\n')
        (package / 'reload_bridge.py').write_text('from .commands import RELOAD_MARKER\n')
        updated += '\nfrom .reload_bridge import RELOAD_MARKER\nassert RELOAD_MARKER == "source graph"\n'
    session = package / 'runtime' / '_session.py'
    session.write_text(
        session.read_text().replace('                        content,', "                        content + ' updated',")
    )
    plugin = root / 'reload_plugin.py'
    plugin.write_text(
        'from pydantic_clai2.commands import Command\n'
        'from pydantic_clai2.plugins import Plugin, SessionStart, SessionEnd, TurnStart\n'
        'class Example(Plugin):\n'
        '    async def on_session_start(self, event):\n'
        '        assert isinstance(event, SessionStart)\n'
        '        self.host.console.print(f"plugin start {event.settings.model} {event.settings.thinking}")\n'
        '    async def on_session_end(self, event):\n'
        '        self.host.console.print("plugin end")\n'
        '    async def on_turn_start(self, event):\n'
        '        assert isinstance(event, TurnStart)\n'
        '        self.host.console.print("plugin turn " + event.text)\n'
        '    def get_commands(self):\n'
        '        return [Command(name="example", description="Example", handler=lambda _: "plugin command")]\n'
    )
    store = SettingsStore(root / 'config.db')
    declaration = PluginSettings(id='example', factory='reload_plugin')
    store.save_plugin(declaration)
    store.save_plugin(PluginSettings(id='disabled', factory='must_not_be_imported', enabled=False))
    sys.path.append(str(root))
    prompts = iter(
        [
            '/help',
            '/set model test',
            '/set display.thinking false',
            '/theme github_light',
            'first',
            '/reload extra',
            '/reload',
            '/help',
            '/missing',
            '/example',
            '/plugins list',
            'second',
            '/reload',
            '/help',
            'third',
            '/new',
            '/exit',
        ]
    )
    labels: list[str] = []
    reloads = 0

    class Prompt(Generic[PromptT]):
        def __init__(self, **kwargs: object) -> None:
            completer = kwargs['completer']
            assert isinstance(completer, Completer)
            self.completer = completer

        async def prompt_async(self, label: str, **kwargs: object) -> str:
            nonlocal reloads
            palette = theme.current()
            assert (palette.name if palette is not None else 'default') == store.load().theme
            assert [item.text for item in self.completer.get_completions(Document('/theme '), CompleteEvent())] == list(
                theme.names()
            )
            assert not list(self.completer.get_completions(Document('/theme github_light '), CompleteEvent()))
            assert [item.text for item in self.completer.get_completions(Document('/rel'), CompleteEvent())] == [
                'reload'
            ]
            labels.append(label)
            text = next(prompts)
            if text == '/reload' and mode != 'unchanged':
                reloads += 1
                sys.path.insert(0, str(root))
                pydantic_clai2.__path__.insert(0, str(package))
                app.write_text(updated)
                if reloads == 1:
                    if mode == 'syntax':
                        app.write_text(updated + '\ninvalid syntax!\n')
                    elif mode == 'import':
                        (package / 'new_module.py').write_text('VALUE = 1\n')
                        app.write_text(updated + '\nfrom . import new_module\nraise RuntimeError("bad import")\n')
                    elif mode == 'harness':
                        app.write_text(updated + '\nfrom pydantic_ai_harness.subagents import _missing_reload_symbol\n')
                    elif mode == 'build':
                        app.write_text(
                            updated.replace('    commands = Commands()', '    raise RuntimeError("bad build")')
                        )
            return text

    deps = object()
    seen: list[list[str]] = []
    hooks = Hooks[object]()

    @hooks.on.before_model_request
    async def observe(ctx: RunContext[object], request_context: ModelRequestContext) -> ModelRequestContext:
        assert ctx.deps is deps
        seen.append(
            [
                str(part.content)
                for message in ctx.messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, UserPromptPart)
            ]
        )
        if mode == 'stock':
            assert 'delegate_task' in {tool.name for tool in request_context.model_request_parameters.function_tools}
            return replace(request_context, model=TestModel(call_tools=[], custom_output_text='hello'))
        return request_context

    output = io.StringIO()
    model = TestModel(call_tools=[], custom_output_text='hello')
    agent = (
        StockAgent(model, deps_type=object, output_type=str, capabilities=[])
        if mode == 'stock'
        else Agent(model, deps_type=object)
    )
    with (
        pytest.MonkeyPatch.context() as patch,
        agent.override(model=TestModel(call_tools=[], custom_output_text='hello')),
    ):
        patch.setattr(prompt_toolkit, 'PromptSession', Prompt)
        patch.setattr('pydantic_clai2._app.PromptSession', Prompt)
        await chat(
            agent,
            deps=deps,
            plugins=[hooks],
            store=store,
            console=Console(file=output, width=200),
            builtin_plugins=(declaration,)
            if mode == 'custom'
            else STOCK_PLUGINS
            if mode == 'stock'
            else DEFAULT_PLUGINS,
            project=ProjectSettings(plugins=(PluginSettings(id='unapproved', factory='unapproved', enabled=False),)),
        )
    conversations = SqliteConversationStore(database=root / 'sessions.db')
    summaries = await conversations.listing()
    assert len(summaries) == 1, summaries
    saved = await conversations.get(conversation_id=summaries[0].id)
    assert [
        str(part.content)
        for message in saved.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ] == seen[2]
    text = output.getvalue()
    assert '/reload: Reload CLAI2 code without restarting' in text, text
    assert 'Usage: /reload' in text, text
    assert ('Goodbye.' if mode == 'unchanged' else 'Updated goodbye.') in text, text
    assert text.count('plugin end') == 3, text
    assert text.count('plugin start test False') == 2, text
    assert all(f'plugin turn {prompt}' in text for prompt in ('first', 'second', 'third')), text
    assert 'plugin command' in text, text
    assert 'disabled: must_not_be_imported (disabled)' in text, text
    assert 'unapproved: unapproved (project) (disabled)' in text, text
    if mode == 'custom':
        assert 'example: reload_plugin (built-in) (enabled, loaded)' in text, text
    assert seen[0] == ['first'], seen
    assert seen[1] == [
        'first',
        'second updated' if mode in ('success', 'custom', 'new_imports', 'stock') else 'second',
    ], seen
    assert seen[2] == [*seen[1], 'third' if mode == 'unchanged' else 'third updated'], seen
    assert labels[-1] == ('> ' if mode == 'unchanged' else 'updated> ')
    assert store.load().model == 'test' and not store.load().thinking
    assert store.load().theme == 'github_light'
    if mode in ('unchanged', 'success', 'custom', 'new_imports', 'stock'):
        assert text.count('CLAI2 reloaded. Conversation preserved.') == 2, text
        assert ('Use /help.' if mode == 'unchanged' else 'Use updated /help.') in text, text
    else:
        assert text.count('Reload failed:') == 1, text
        assert text.count('CLAI2 reloaded. Conversation preserved.') == 1, text
        assert 'Use /help.' in text, text
        assert 'pydantic_clai2.new_module' not in sys.modules
    assert ('Harness is not refreshed by /reload.' in text) == (mode == 'harness'), text
    if mode == 'harness':
        assert '--resume to continue this session.' in text, text
        assert 'Keep the worktree if asked to remove it.' in text, text


async def transcript_reload(root: Path) -> None:
    """Reload live transcript instances, including their nested lines and Markdown streams."""
    import pydantic_clai2._app as app
    from pydantic_clai2.runtime.reloading import reload_clai
    from pydantic_clai2.ui.prompt import prompt_transcript

    transcript = prompt_transcript.TranscriptBuffer(max_lines=3, max_chars=100)
    transcript.write('before\n')

    def render(*, source: str, width: int) -> str:
        return source + '\n'

    block = transcript.markdown(render=render, width=80, changed=lambda: None)
    block.extend('markdown')
    block.write('markdown\n')
    model = TestModel(call_tools=[], custom_output_text='hello')
    agent = Agent(model)
    output = io.StringIO()
    for _ in range(2):
        previous_type = type(transcript)
        shell = reload_clai(
            lambda: app.create_shell(
                agent,
                deps=None,
                console=Console(file=output),
                store=SettingsStore(root / 'transcript-config.db'),
                transcript=transcript,
                plugins=(),
                usage_limits=None,
                settings=None,
                builtin_plugins=(),
                project=ProjectSettings(),
                load_plugins=False,
            )
        )
        assert shell.transcript is transcript
        assert type(transcript) is prompt_transcript.TranscriptBuffer
        assert type(transcript) is not previous_type
        assert type(block) is prompt_transcript.MarkdownBlock
        block.write('continued\n')
        with transcript.capture(shell.console):
            shell.console.print('notice')
        # Eviction exercises isinstance checks on objects created before module reload.
        transcript.write('after\n')
        frame = transcript.frame(width=80, height=24)
        assert Text.from_ansi('\n'.join(frame.rows)).plain.splitlines()[-2:] == ['notice', 'after']


asyncio.run(
    transcript_reload(Path(sys.argv[1])) if sys.argv[2] == 'transcript' else main(Path(sys.argv[1]), sys.argv[2])
)
