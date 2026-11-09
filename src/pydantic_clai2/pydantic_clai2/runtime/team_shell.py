"""One session's team in the shell: `/team` and `/board`, the roster panel, the team keys, and its plugins' feed."""

from collections.abc import Awaitable, Callable, Iterable, Mapping

from pydantic_ai_harness.subagents import DelegationTask
from pydantic_ai_harness.teams import TeamSnapshot
from pydantic_clai2.plugins import TeamProgress
from pydantic_clai2.runtime.forks import parse_fork_args
from pydantic_clai2.runtime.tasks import Tasks
from pydantic_clai2.runtime.team import SessionTeam, item_label, roster_rows
from pydantic_clai2.ui.menus.team_menu import TeamAction, member_label, open_team

TEAMS_OFF = 'Teams are off. /set team.enabled true turns them on.'
TEAM_USAGE = 'Usage: /team [list | spawn NAME [@MODEL] PROMPT | msg NAME TEXT | stop NAME]'
BOARD_USAGE = 'Usage: /board [list | assign ID NAME | unclaim ID | cancel ID]'


class ShellTeam:
    """The team of the conversation a session holds, as its shell shows and drives it."""

    def __init__(
        self,
        *,
        tasks: Tasks,
        enabled: Callable[[], bool],
        interactive: Callable[[], bool],
        fire: Callable[[TeamProgress], Awaitable[None]],
        send_input: Callable[[str], None],
        models: Callable[[], Iterable[str]],
        team_models: Callable[[], Mapping[str, str]],
    ) -> None:
        """`enabled` says whether the session leads a team; `interactive` whether its terminal can show a menu.

        `fire` tells plugins about a change, `send_input` enters a command as typed, and `models` lists the
        saved models a spawn can name, and `team_models` the configured model of each agent type.
        """
        self.tasks = tasks
        self.teams = tasks.router.teams
        self.catalog = SessionTeam(
            self.teams, conversation_id=tasks.conversation_id, history=tasks.history, models=team_models
        )
        """What plugins reach as `host.team` while `enabled`."""
        self.enabled = enabled
        self.interactive = interactive
        self._fire = fire
        self._send_input = send_input
        self._models = models
        tasks.on_team = self.deliver

    async def deliver(self, progress: TeamProgress) -> None:
        """A change to the team of the session's conversation, in order: plugins hear it while teams are on."""
        if not self.enabled():
            return
        # A teammate's request for the user's approval of its plan will open its prompt here.
        await self._fire(progress)

    # The roster panel and keys.

    def rows(self, glyph: str) -> tuple[str, ...]:
        """The roster panel's rows, from the cached team; none while teams are off."""
        if not self.enabled():
            return ()
        started = {record.id: record.started_at for record in self.tasks.records() if record.name is not None}
        return roster_rows(self.catalog.snapshot(), glyph=glyph, progress=self.tasks.progress, started=started)

    def key(self, command: str) -> str:
        """`ctrl-x a` and `ctrl-t`: open the roster or the board, as typing `command` does."""
        if not self.enabled():
            return TEAMS_OFF
        self._send_input(command)
        return ''

    # What the menus read and do.

    def snapshot(self) -> TeamSnapshot:
        return self.catalog.snapshot()

    def member(self, name: str) -> DelegationTask | None:
        return next((record for record in self.tasks.snapshots() if record.name == name), None)

    async def act(self, action: TeamAction) -> str:
        """Do what the user asked in a menu or a command, returning what to tell them."""
        team, target, value = self.catalog, action.target, action.value
        match action.kind:
            case 'message':
                return f'Message {await team.send(target, value)} for {target}.'
            case 'stop':
                await team.stop(target)
                return f'Stopped {target}; your next message wakes it.'
            case 'assign':
                await team.assign(target, value)
                return f'Assigned #{target} to {value}.'
            case 'unclaim':
                await team.unclaim(target)
                return f'#{target} is pending again.'
            case 'cancel':  # pragma: no branch -- every kind has its case.
                await team.cancel(target)
                return f'Cancelled #{target}.'

    # `/team` and `/board`.

    async def _snapshot(self) -> TeamSnapshot:
        return await self.teams.refresh(self.tasks.conversation_id())

    async def _menu(self, *, board: bool) -> str | None:
        """Open a menu when the terminal can show it and teams are on; `None` to print a listing instead."""
        if not (self.interactive() and self.enabled()):
            return None
        await self._snapshot()
        return await open_team(self, board=board)

    async def command(self, args: list[str]) -> str:
        """`/team`: the roster menu, or `list`, `spawn NAME [@MODEL] PROMPT`, `msg NAME TEXT`, or `stop NAME`."""
        text = args[0].strip() if args else ''
        action, _, rest = text.partition(' ')
        name, _, rest = rest.strip().partition(' ')
        rest = rest.strip()
        if not action and (opened := await self._menu(board=False)) is not None:
            return opened
        if action in ('', 'list') and not name:
            members = (await self._snapshot()).members
            listing = '\n'.join(member_label(member) for member in members)
            return listing or 'No teammates in this conversation. /team spawn NAME [@MODEL] PROMPT starts one.'
        if action == 'spawn' and name:
            if not self.enabled():
                raise ValueError(TEAMS_OFF)
            model, prompt = parse_fork_args(rest)
            if not prompt:
                raise ValueError('Usage: /team spawn NAME [@MODEL] PROMPT')
            member = await self.catalog.spawn(name, prompt, model=model)
            return f'Started {member.name} on {member.model or "the session model"}.'
        if action == 'msg' and name and rest:
            return await self.act(TeamAction(kind='message', target=name, value=rest))
        if action == 'stop' and name and not rest:
            return await self.act(TeamAction(kind='stop', target=name))
        raise ValueError(TEAM_USAGE)

    def complete(self, args: list[str]) -> Iterable[str]:
        """Suggest the subcommands, then teammates' names, then saved models as `@model` after a spawn's name."""
        if len(args) <= 1:
            return ('list', 'spawn', 'msg', 'stop')
        if args[0] in ('msg', 'stop') and len(args) == 2:
            return tuple(member.name for member in self.catalog.snapshot().members)
        if args[0] == 'spawn' and len(args) == 3:
            return tuple(f'@{name}' for name in self._models())
        return ()

    async def board_command(self, args: list[str]) -> str:
        """`/board`: the board menu, or `list`, `assign ID NAME`, `unclaim ID`, or `cancel ID`."""
        if not args and (opened := await self._menu(board=True)) is not None:
            return opened
        if args in ([], ['list']):
            board = (await self._snapshot()).board
            return '\n'.join(item_label(item, board=board) for item in board) or 'The board is empty.'
        if not self.enabled():
            raise ValueError(TEAMS_OFF)
        match args:
            case ['assign', item_id, name]:
                return await self.act(TeamAction(kind='assign', target=item_id.removeprefix('#'), value=name))
            case ['unclaim' | 'cancel' as kind, item_id]:
                return await self.act(TeamAction(kind=kind, target=item_id.removeprefix('#')))
            case _:
                raise ValueError(BOARD_USAGE)

    def complete_board(self, args: list[str]) -> Iterable[str]:
        """Suggest the subcommands, then board items, then teammates to assign one to."""
        if len(args) <= 1:
            return ('list', 'assign', 'unclaim', 'cancel')
        snapshot = self.catalog.snapshot()
        if len(args) == 2 and args[0] in ('assign', 'unclaim', 'cancel'):
            return tuple(item.id for item in snapshot.board)
        if len(args) == 3 and args[0] == 'assign':
            return tuple(member.name for member in snapshot.members)
        return ()
