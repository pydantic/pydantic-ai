"""Interactive application error, cancellation, and input boundaries."""

import asyncio
import io
import signal
from pathlib import Path
from typing import Generic, TypeVar

import pytest
from prompt_toolkit.styles import BaseStyle
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from rich.console import Console

from pydantic_clai2 import chat
from pydantic_clai2.command_context import CommandContext
from pydantic_clai2.commands import Command
from pydantic_clai2.config import Settings
from pydantic_clai2.settings_store import SettingsStore

PromptT = TypeVar('PromptT')


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def inputs(monkeypatch: pytest.MonkeyPatch, values: list[str | BaseException]) -> None:
    class Prompt(Generic[PromptT]):
        def __init__(self, **kwargs: object) -> None:
            style = kwargs['style']
            assert isinstance(style, BaseStyle)
            for selector in ('class:bottom-toolbar', 'class:bottom-toolbar.text'):
                assert style.get_attrs_for_style_str(selector).color == '9B77FF'

        async def prompt_async(self, label: str) -> str:
            value = values.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value

    monkeypatch.setattr('pydantic_clai2._app.PromptSession', Prompt)


@pytest.mark.parametrize('mode', ['eof', 'interrupt', 'error', 'cancel', 'double', 'structured'])
async def test_chat_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    output = io.StringIO()
    values: list[str | BaseException] = [' ', '/bad', 'run', '/exit']
    if mode == 'eof':
        values = [EOFError()]
    elif mode == 'interrupt':
        values = [KeyboardInterrupt(), KeyboardInterrupt()]

    class Behaviour(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if mode == 'error':
                raise ValueError('broken provider')
            if mode in ('cancel', 'double'):
                signal.raise_signal(signal.SIGINT)
                if mode == 'double':
                    signal.raise_signal(signal.SIGINT)
                await asyncio.sleep(0)
            return request_context

    inputs(monkeypatch, values)
    console = Console(file=output, width=20 if mode == 'eof' else 120)
    store = SettingsStore(tmp_path / 'config.db')
    if mode == 'structured':
        await chat(Agent(TestModel(), output_type=list[int]), deps=None, console=console, store=store)
    else:
        await chat(
            Agent(TestModel(), deps_type=type(None), capabilities=[Behaviour()]),
            deps=None,
            console=console,
            store=store,
        )
    if mode == 'error':
        assert 'broken provider' in output.getvalue()
    elif mode == 'cancel':
        assert 'Turn cancelled' in output.getvalue()
    elif mode == 'interrupt':
        assert 'Input cleared' in output.getvalue()


async def test_model_string_and_non_command_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs(monkeypatch, ['/set', '/set display.thinking', '/config show', '/plugins list', '/new', '/exit'])

    class Provider(AbstractCapability[None]):
        def get_commands(self, context: CommandContext) -> list[Command]:
            return [Command(name='legacy', description='Legacy command', handler=lambda args: 'ok')]

    await chat(
        Agent('test'),
        deps=None,
        plugins=[AbstractCapability(), Provider()],
        settings=Settings(model='test'),
        console=Console(file=io.StringIO()),
        store=SettingsStore(tmp_path / 'config.db'),
    )


@pytest.mark.parametrize('provider', ['openrouter', 'vllm'])
async def test_connected_provider_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    inputs(monkeypatch, ['hello', '/exit'])

    def model(name: str) -> TestModel:
        assert name == f'{provider}:test'
        return TestModel(custom_output_text='Connected response')

    monkeypatch.setattr(f'pydantic_clai2.{provider}.model', model)
    output = io.StringIO()
    await chat(
        Agent(TestModel()),
        deps=None,
        settings=Settings(model=f'{provider}:test'),
        store=SettingsStore(tmp_path / 'config.db'),
        console=Console(file=output),
    )
    assert 'Connected response' in output.getvalue()
