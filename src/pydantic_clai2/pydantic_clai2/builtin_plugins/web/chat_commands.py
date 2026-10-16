"""What the web UI's composers complete and run: CLAI's commands, and the orchestrator chat's own.

The session's composer enters every `/command` at the prompt. The orchestrator chat runs its own (`OWN`)
on its conversation, and enters at the prompt only CLAI's commands in `SHARED`, showing their results
in its chat instead of the session's.
"""

from pydantic_clai2.builtin_plugins.web.protocol import CompletionInfo, Notice
from pydantic_clai2.commands import is_command_input
from pydantic_clai2.plugins import CommandEnd, PluginHost

SHARED = frozenset(
    {
        'accounts',
        'config',
        'keys',
        'login',
        'mcp',
        'plugins',
        'set',
        'settings',
        'spinner',
        'theme',
        'web',
        'workspaces',
    }
)
"""CLAI's commands the orchestrator chat runs too: those that change CLAI itself, not the session's conversation.

`Command` declares no such scope, so this list is the split. The others, such as `/fork`, `/compact`, a
skill, or a plugin's own command, act on the session or may, so the orchestrator chat refuses them.
"""

OWN = {
    'model': "Show or pick the orchestrator's model; /model session follows the session's again",
    'new': 'Start a new orchestrator chat; its workspaces are kept',
    'clear': 'Alias of /new',
    'help': 'List the commands that work in the orchestrator chat',
    'stop': "Stop the orchestrator's running turn",
}
"""The orchestrator chat's own commands, acting on its conversation in place of the session's of the same name."""

FOLLOW = ('session', 'default')
"""What `/model` takes in the orchestrator chat to follow the session's model again."""


def parse(text: str) -> tuple[str, list[str]]:
    """A `/command`'s name and arguments; a bare `/` asks for help, as at the prompt."""
    name, *args = text.removeprefix('/').split() or ['help']
    return name, args


def result_notice(event: CommandEnd) -> Notice | None:
    """What a chat shows for a command that ended: its error, its output, or nothing."""
    if event.error is not None:
        return Notice(text=str(event.error), level='error', command=event.text)
    return Notice(text=event.output, command=event.text) if event.output else None


async def completions(host: PluginHost[None], text: str) -> list[CompletionInfo]:
    """What Tab offers at the prompt for `text`."""
    return [
        CompletionInfo(
            text=completion.text,
            start=-completion.start_position,
            display=completion.display or completion.text,
            meta=completion.display_meta,
        )
        for completion in await host.complete(text)
    ]


async def saved_models(host: PluginHost[None]) -> list[str]:
    """The saved models and chains `/model` offers, as `PROVIDER:NAME`."""
    return [completion.text for completion in await completions(host, '/model ') if ':' in completion.text]


async def registered(host: PluginHost[None], name: str) -> bool:
    """Whether CLAI has a command called `name` that is available now."""
    return any(completion.text == name for completion in await host.complete(f'/{name}'))


async def orchestrator_completions(host: PluginHost[None], text: str) -> list[CompletionInfo]:
    """What Tab offers in the orchestrator chat: its own commands and the `SHARED` ones, with their arguments."""
    words = text[1:].split()
    if not is_command_input(text) or (not words and text.endswith(' ')):
        return []
    if len(words) <= 1 and not text.endswith(' '):
        prefix = text[1:]
        own = [
            CompletionInfo(text=name, start=len(prefix), display=f'/{name}', meta=meta)
            for name, meta in OWN.items()
            if prefix in name
        ]
        shared = [found for found in await completions(host, text) if found.text in SHARED]
        # Best match first, as at the prompt: the exact name, then names starting with the fragment.
        return sorted([*own, *shared], key=lambda found: (found.text != prefix, not found.text.startswith(prefix)))
    name, args = words[0], [*words[1:], *([''] if text.endswith(' ') else [])]
    if name == 'model' and len(args) == 1:
        choices = (FOLLOW[0], *await saved_models(host))
        return [
            CompletionInfo(text=choice, start=len(args[0]), display=choice) for choice in choices if args[0] in choice
        ]
    return await completions(host, text) if name in SHARED else []


async def orchestrator_help(host: PluginHost[None]) -> str:
    """`/help` in the orchestrator chat: its own commands, then CLAI's that it runs too."""
    shared = [found for found in await completions(host, '/') if found.text in SHARED]
    return '\n'.join(
        [
            *(f'/{name}: {meta}' for name, meta in OWN.items()),
            *(f'/{found.text}: {found.meta}' for found in shared),
            "The session's own commands, such as /fork or /compact, work in a session, not here.",
        ]
    )
