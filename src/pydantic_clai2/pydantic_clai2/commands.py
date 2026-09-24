"""One command registry for execution, help, and Termflow completion."""

import json
import shlex
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue, TypeAdapter
from pydantic_ai.models import known_model_names
from termflow.tui.completion import (  # pyright: ignore[reportMissingTypeStubs]
    CompleteEvent,
    Completer,
    Completion,
    Document,
)

from .config import SETTING_FIELDS, STRING_SETTINGS, PluginSettings
from .settings_store import SettingsStore
from .spinners import BUILTIN_SPINNERS
from .theme import names as theme_names


def is_command_input(text: str) -> bool:
    """Recognize slash commands without routing path-like prefixes to the registry.

    A slash, dot, or backslash in the first token after `/` denotes a path, not a command
    name. This is lexical: it neither reads local files nor parses prompt text
    as shell arguments. Unknown command-shaped names still reach the registry.
    """
    if not text.startswith('/'):
        return False
    name = text.split(maxsplit=1)[0][1:]
    return not any(marker in name for marker in ('/', '.', '\\'))


def expand_bare_command(text: str) -> str:
    """Map bare `clear` to `/clear`, as Code Puppy does, so every input path dispatches it as a command."""
    return '/clear' if text.strip().lower() == 'clear' else text


@dataclass(frozen=True, kw_only=True)
class Command:
    """A command available to both dispatch and autocomplete."""

    name: str
    description: str
    handler: Callable[[list[str]], str | Awaitable[str]]
    complete: Callable[[list[str]], Iterable[str]] = lambda _: ()


class Commands(Completer):
    """Instance-owned registry, based on Code Puppy's registry/completer pattern."""

    def __init__(self) -> None:
        """Create an empty registry and filesystem completer."""
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        """Register one command, rejecting ambiguous duplicate names."""
        self.register_many((command,))

    def register_many(self, commands: Iterable[Command]) -> None:
        """Validate a provider's declarations atomically, including collisions."""
        pending: dict[str, Command] = {}
        for command in commands:
            if command.name in self._commands or command.name in pending or not command.name.isidentifier():
                raise ValueError(f'Invalid or duplicate command: {command.name}')
            pending[command.name] = command
        self._commands.update(pending)

    def unregister(self, names: Iterable[str]) -> None:
        """Remove commands a plugin registered; unknown names are ignored."""
        for name in names:
            self._commands.pop(name, None)

    def __iter__(self) -> Iterator[Command]:
        """Iterate a snapshot, so callers may register or unregister while looping."""
        return iter(list(self._commands.values()))

    def execute(self, text: str) -> str | Awaitable[str]:
        """Parse shell-style arguments and dispatch without invoking a shell."""
        words = shlex.split(text.removeprefix('/'))
        if not words:
            return self.help([])
        name, *args = words
        command = self._commands.get(name)
        if command is None:
            raise ValueError(f'Unknown command /{name}. Use /help.')
        return command.handler(args)

    async def execute_async(self, text: str) -> str:
        """Await asynchronous plugin commands without blocking the event loop."""
        result = self.execute(text)
        return result if isinstance(result, str) else await result

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterator[Completion]:
        """Complete slash commands, contextual arguments, and @file paths."""
        text = document.text_before_cursor
        if not is_command_input(text):
            word = document.get_word_before_cursor(WORD=True)
            if word.startswith('@'):
                path = Path(word[1:]).expanduser()
                directory = path if word.endswith('/') else path.parent
                prefix = '' if word.endswith('/') else path.name
                try:
                    for child in sorted(directory.iterdir()):
                        if prefix in child.name:
                            yield Completion(child.name + ('/' if child.is_dir() else ''), start_position=-len(prefix))
                except OSError:
                    return
            return
        words = text[1:].split()
        if len(words) <= 1 and not text.endswith(' '):
            prefix = text[1:]
            for command in list(self._commands.values()):
                if prefix in command.name:
                    yield Completion(
                        command.name,
                        start_position=-len(prefix),
                        display='/' + command.name,
                        display_meta=command.description,
                    )
            return
        if not words:
            return
        command = self._commands.get(words[0])
        if command is None:
            return
        args = words[1:]
        if text.endswith(' '):
            args.append('')
        prefix = args[-1] if args else ''
        for candidate in command.complete(args):
            if prefix in candidate:
                yield Completion(candidate, start_position=-len(prefix))

    def help(self, _: list[str]) -> str:
        """Generate help from the same registry used for completion."""
        return '\n'.join(f'/{command.name}: {command.description}' for command in self._commands.values())


def config_command(store: SettingsStore, args: list[str]) -> str:
    """Share settings commands between the terminal and CLI."""
    if args == ['show'] or not args:
        return store.load().model_dump_json(indent=2)
    if len(args) == 2 and args[0] == 'get':
        field = SETTING_FIELDS.get(args[1])
        if field is None:
            raise ValueError(f'Unknown setting: {args[1]}')
        return json.dumps(store.load().model_dump()[field])
    if len(args) == 2 and args[0] == 'reset':
        store.reset(args[1])
    elif len(args) == 3 and args[0] == 'set':
        adapter: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
        value: JsonValue = args[2] if args[1] in STRING_SETTINGS else adapter.validate_json(args[2])
        store.set(args[1], value)
    else:
        raise ValueError('Usage: config show|get KEY|set KEY VALUE|reset KEY')
    return 'Saved. Applies when you restart CLAI.'


def set_completions(args: list[str]) -> Iterable[str]:
    """Complete setting names and values without network calls or credentials."""
    if len(args) <= 1:
        return (*SETTING_FIELDS, 'api_key')
    if len(args) == 2 and args[0] == 'display.theme':
        return theme_names()
    if len(args) == 2 and args[0] == 'display.spinner':
        return tuple(BUILTIN_SPINNERS)
    if len(args) == 2 and args[0] == 'model':
        from .model_catalog import CODEX_MODELS  # noqa: PLC0415

        names = known_model_names()
        providers = sorted({name.partition(':')[0] + ':' for name in names} | {'openai-codex:'})
        return tuple(dict.fromkeys((*providers, *CODEX_MODELS, *names)))
    if len(args) == 2 and args[0] in ('display.thinking', 'display.splash'):
        return ('true', 'false')
    return ()


def config_completions(args: list[str]) -> Iterable[str]:
    """Suggest operations, setting names, and boolean values."""
    if len(args) <= 1:
        return ('show', 'get', 'set', 'reset')
    if len(args) == 2 and args[0] in ('get', 'set', 'reset'):
        return SETTING_FIELDS
    if len(args) == 3 and args[0] == 'set':
        return set_completions(args[1:])
    return ()


def plugins_command(store: SettingsStore, args: list[str]) -> str:
    """Manage explicit plugin declarations without importing plugins."""
    declarations = store.plugins()
    if not args or args == ['list']:
        return (
            '\n'.join(f'{p.id}: {p.factory} ({"enabled" if p.enabled else "disabled"})' for p in declarations)
            or 'No plugins.'
        )
    if len(args) in (3, 4) and args[0] == 'add':
        settings = TypeAdapter(dict[str, JsonValue]).validate_json(args[3]) if len(args) == 4 else {}
        store.save_plugin(PluginSettings(id=args[1], factory=args[2], settings=settings))
    elif len(args) == 2 and args[0] in ('enable', 'disable'):
        plugin = next((p for p in declarations if p.id == args[1]), None)
        if plugin is None:
            raise ValueError(f'Unknown plugin: {args[1]}')
        store.save_plugin(plugin.model_copy(update={'enabled': args[0] == 'enable'}))
    elif len(args) == 2 and args[0] == 'remove':
        store.delete_plugin(args[1])
    else:
        raise ValueError('Usage: plugins list|add ID MODULE[:ATTR] [JSON]|enable ID|disable ID|remove ID')
    return 'Saved. Plugin code is trusted and loads on next startup.'
