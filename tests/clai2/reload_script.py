"""Exercise real module reloads in a process isolated from pytest's imported class identities."""

import asyncio
import io
import sys
from pathlib import Path
from typing import Generic, TypeVar

import prompt_toolkit
import pytest
from prompt_toolkit.completion import CompleteEvent, Completer
from prompt_toolkit.document import Document
from pydantic_ai import Agent, ModelRequestContext, RunContext, models
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from rich.console import Console

import pydantic_clai2
from pydantic_clai2 import DEFAULT_PLUGINS, chat
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.project_settings import ProjectSettings
from pydantic_clai2.settings_store import SettingsStore

models.ALLOW_MODEL_REQUESTS = False
PromptT = TypeVar('PromptT')


async def main(root: Path, mode: str) -> None:
    package = root / 'pydantic_clai2'
    app = package / '_app.py'
    original = app.read_text()
    updated = original.replace("prompt_async('> ',", "prompt_async('updated> ',")
    updated = updated.replace('New session started.', 'Updated session started.')
    commands = package / 'commands.py'
    commands.write_text(commands.read_text().replace('Use /help.', 'Use updated /help.'))
    session = package / '_session.py'
    session.write_text(
        session.read_text().replace('                        text,', "                        text + ' updated',")
    )
    plugin = root / 'reload_plugin.py'
    plugin.write_text(
        'from pydantic_clai2.commands import Command\n'
        'from pydantic_clai2.plugins import SessionStart, SessionEnd, TurnStart\n'
        'def activate(host):\n'
        '    @host.on("session_start")\n'
        '    async def start(event):\n'
        '        assert isinstance(event, SessionStart)\n'
        '        host.console.print(f"plugin start {event.settings.model} {event.settings.thinking}")\n'
        '    @host.on("session_end")\n'
        '    async def end(event):\n'
        '        host.console.print("plugin end")\n'
        '    @host.on("turn_start")\n'
        '    async def turn(event):\n'
        '        assert isinstance(event, TurnStart)\n'
        '        host.console.print("plugin turn " + event.text)\n'
        '    host.commands.register(Command(name="example", description="Example", handler=lambda _: "plugin command"))\n'
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
        return request_context

    output = io.StringIO()
    agent = Agent(TestModel(call_tools=[], custom_output_text='hello'), deps_type=object)
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
            builtin_plugins=(declaration,) if mode == 'custom' else DEFAULT_PLUGINS,
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
    assert ('New session started.' if mode == 'unchanged' else 'Updated session started.') in text, text
    assert text.count('plugin end') == 3, text
    assert text.count('plugin start test False') == 2, text
    assert all(f'plugin turn {prompt}' in text for prompt in ('first', 'second', 'third')), text
    assert 'plugin command' in text, text
    assert 'disabled: must_not_be_imported (disabled)' in text, text
    assert 'unapproved: unapproved (project) (disabled)' in text, text
    if mode == 'custom':
        assert 'example: reload_plugin (built-in) (enabled, loaded)' in text, text
    assert seen[0] == ['first'], seen
    assert seen[1] == ['first', 'second updated' if mode in ('success', 'custom') else 'second'], seen
    assert seen[2] == [*seen[1], 'third' if mode == 'unchanged' else 'third updated'], seen
    assert labels[-1] == ('> ' if mode == 'unchanged' else 'updated> ')
    assert store.load().model == 'test' and not store.load().thinking
    if mode in ('unchanged', 'success', 'custom'):
        assert text.count('CLAI2 reloaded. Conversation preserved.') == 2, text
        assert ('Use /help.' if mode == 'unchanged' else 'Use updated /help.') in text, text
    else:
        assert text.count('Reload failed:') == 1, text
        assert text.count('CLAI2 reloaded. Conversation preserved.') == 1, text
        assert 'Use /help.' in text, text
        assert 'pydantic_clai2.new_module' not in sys.modules


asyncio.run(main(Path(sys.argv[1]), sys.argv[2]))
