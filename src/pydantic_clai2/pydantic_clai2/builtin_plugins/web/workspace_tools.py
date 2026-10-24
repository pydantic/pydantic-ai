"""The orchestrator's tools: make a workspace, start a session in it with a prompt, watch it, steer it, and report back.

They work over the `workspaces` plugin's registry, whose settings are the kinds of workspace the model may
make. Each workspace's session is a live session of this CLAI in its checkout, followed through
`local_client.LocalClient` as a page follows it, so the sidebar lists it under its repository like any
other. Only the orchestrator's own agent gets these tools (`web.orchestrator`), never a session's.
"""

import asyncio
import functools
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ParamSpec

import anyio
from pydantic import BaseModel, TypeAdapter, ValidationError

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai_harness.ask_user import AskUserAnswer
from pydantic_clai2.builtin_plugins.web.client import SessionView
from pydantic_clai2.builtin_plugins.web.local_client import LocalClient, live_session
from pydantic_clai2.builtin_plugins.web.protocol import Item, SessionsChanged
from pydantic_clai2.builtin_plugins.web.workspaces import required_workspaces, workspace_session
from pydantic_clai2.builtin_plugins.workspaces import Workspaces, describe
from pydantic_clai2.builtin_plugins.workspaces.manager import works_in

if TYPE_CHECKING:
    from pydantic_clai2.builtin_plugins.web.service import WebService

ASK_USER_TOOL = 'ask_user_question'
CONNECT_TIMEOUT = 15.0
MAX_WAIT = 3600.0
ITEM_CHARS = 600
ANSWER_CHARS = 4000


class SessionAnswer(BaseModel):
    """The answer to one question a session asks: labels of the options picked, or `custom` text instead."""

    header: str
    selected: list[str] = []
    custom: str | None = None


@dataclass(kw_only=True)
class _Link:
    """A workspace's session, followed in a task."""

    view: SessionView
    task: asyncio.Task[None]
    live_id: str


def _cut(text: str, chars: int) -> str:
    text = text.strip()
    return text if len(text) <= chars else f'{text[:chars]}… ({len(text) - chars:,} more characters)'


def summarize(item: Item) -> str:
    """One item as a short line: what it is, then its text, cut to `ITEM_CHARS`."""
    if item.kind == 'tool':
        return f'- tool `{item.name}` ({item.state}): {_cut(item.args, 200)} -> {_cut(item.text, ITEM_CHARS)}'
    return f'- {item.kind}: {_cut(item.text, ITEM_CHARS)}'


def _asked(view: SessionView) -> str:
    lines: list[str] = []
    for request in view.questions.values():
        lines.append(f'Question request {request.id}:')
        lines += [
            f'  - {question.header}: {question.question} Options: '
            + ', '.join(option.label for option in question.options)
            for question in request.questions
        ]
    return '\n'.join(lines)


_PICKED: TypeAdapter[dict[str, list[str]]] = TypeAdapter(dict[str, list[str]])


def confirmed(messages: Sequence[ModelMessage], workspace_id: str) -> bool:
    """Whether the user's latest pick of `Destroy ID` or `Keep ID` in an `ask_user_question` answer was `Destroy ID`.

    A later `Keep ID` takes an earlier `Destroy ID` back.
    """
    destroy, keep = f'Destroy {workspace_id}', f'Keep {workspace_id}'
    approved = False
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, ToolReturnPart) or part.tool_name != ASK_USER_TOOL:
                continue
            try:
                picked = _PICKED.validate_python(part.content)
            except ValidationError:  # Declined: the user picked nothing.
                continue
            for labels in picked.values():
                if destroy in labels:
                    approved = True
                elif keep in labels:
                    approved = False
    return approved


_PROBLEMS = (LookupError, ValueError, RuntimeError, OSError)
"""What the workspace and session layers raise for a problem the model can act on, such as an unknown workspace."""
_BUGS = (KeyError, IndexError, NotImplementedError, RecursionError)
"""Subclasses of those that mean a bug here, which fails the turn instead."""
P = ParamSpec('P')


def reported(tool: Callable[P, Awaitable[str]]) -> Callable[P, Awaitable[str]]:
    """`tool`, with a problem the model can act on returned as an error line instead of failing the run."""

    @functools.wraps(tool)
    async def run(*args: P.args, **kwargs: P.kwargs) -> str:
        try:
            return await tool(*args, **kwargs)
        except _BUGS:
            raise
        except _PROBLEMS as exc:
            return f'Error: {exc}'

    return run


class WorkspaceTools:
    """The tools, and the workspaces' sessions they follow between calls."""

    def __init__(self, service: 'WebService') -> None:
        self.service = service
        self.links: dict[str, _Link] = {}
        self._locks: defaultdict[str, anyio.Lock] = defaultdict(anyio.Lock)
        """One per workspace, so tool calls the model makes at once follow its session once, not once each."""

    def workspaces(self) -> Workspaces:
        """The registry with the kinds the `workspaces` plugin is configured with; `RuntimeError` while it is off."""
        return required_workspaces(self.service.host.plugins)

    async def view(self, workspace_id: str, *, start: bool = False) -> SessionView:
        """The workspace's session, followed from first use until it closes; with `start`, a new one if it has none."""
        async with self._locks[workspace_id]:
            link = self.links.get(workspace_id)
            if link is not None and not link.view.closed and await self._in_checkout(workspace_id, link.live_id):
                return link.view
            await self.disconnect(workspace_id)
            channel = await workspace_session(self.service, workspace_id, start=start)
            view = SessionView(LocalClient(self.service, channel.session.id))
            task = asyncio.create_task(view.follow(), name=f'clai2-orchestrator-{workspace_id}')
            self.links[workspace_id] = _Link(view=view, task=task, live_id=channel.session.id)
            with anyio.move_on_after(CONNECT_TIMEOUT):
                await view.ready()
            if view.session is None:
                await self.disconnect(workspace_id)
                raise RuntimeError(f'Could not follow the session of workspace {workspace_id}.')
            return view

    async def _in_checkout(self, workspace_id: str, live_id: str) -> bool:
        """Whether the live session still works in the checkout, which `/workspace` or a `/resume` there may change.

        The session itself, not the view: it moves before its snapshot reaches the view.
        """
        session = live_session(self.service, live_id)
        if session is None:
            return False
        return works_in(session.workspace, (await self.workspaces().get(workspace_id)).path)

    async def disconnect(self, workspace_id: str) -> None:
        """Stop following a workspace's session, which stays open until it is idle."""
        link = self.links.pop(workspace_id, None)
        if link is None:
            return
        link.task.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(link.task, return_exceptions=True)

    async def close(self) -> None:
        """Stop following every workspace's session."""
        for workspace_id in list(self.links):
            await self.disconnect(workspace_id)

    def tools(self) -> list[Callable[..., Awaitable[str]]]:
        """Every tool, `reported`, for an agent."""
        return [
            reported(self.workspace_list),
            reported(self.workspace_create),
            reported(self.workspace_destroy),
            reported(self.session_start),
            reported(self.session_send),
            reported(self.session_read),
            reported(self.session_wait),
            reported(self.session_interrupt),
            reported(self.session_answer),
        ]

    # The tools. Each returns text for the model; a problem it can act on comes back as an error line.

    async def workspace_list(self) -> str:
        """List the workspaces, with their kind, branch, and folder, and the kinds you can make."""
        workspaces = self.workspaces()
        listing = await workspaces.listing()
        lines = [describe(workspace) for workspace in listing] or ['No workspaces yet.']
        return '\n'.join([*lines, f'Kinds: {", ".join(sorted(workspaces.kinds)) or "none"}.'])

    async def workspace_create(self, kind: str, repo: str = '', ref: str = '') -> str:
        """Make a workspace of `kind`: `repo` (a local path or a URL; empty for the kind's own) checked out at `ref`.

        `ref` is a branch, tag, or commit, such as `main`; empty for the kind's own or the repository's `HEAD`.
        The workspace gets a new branch. Nothing runs in it until `session_start`.
        """
        workspace = await self.workspaces().create(kind=kind, repo=repo, ref=ref, base=self.service.host.workspace)
        self.service.hub.publish(SessionsChanged())  # The sidebar lists its checkout.
        return f'Created workspace {workspace.id} at {workspace.path} on branch {workspace.branch}.'

    async def workspace_destroy(self, ctx: RunContext[None], workspace: str) -> str:
        """Remove its checkout, closing the sessions working there first, unless its kind keeps it; then forget it.

        Only after the user picked `Destroy ID` in `ask_user_question`, with ID the workspace's.
        """
        if not confirmed(ctx.messages, workspace):
            return (
                f'Not destroyed. Ask the user first with `ask_user_question`, offering `Destroy {workspace}` '
                f'and `Keep {workspace}`, and call this again only if they pick `Destroy {workspace}`.'
            )
        service = self.service
        # Neither a tool call nor the page opens its session again meanwhile.
        async with self._locks[workspace], service.workspace_locks[workspace]:
            await self.disconnect(workspace)
            destroyed = await self.workspaces().destroy(workspace, live=service.live)
        del self._locks[workspace], service.workspace_locks[workspace]
        service.hub.publish(SessionsChanged())
        kept = 'removed' if destroyed.teardown == 'remove' else 'kept'
        return f'Destroyed workspace {destroyed.id}; its checkout at {destroyed.path} was {kept}, its branch kept.'

    async def session_start(self, workspace: str, prompt: str) -> str:
        """Start a session in the workspace, unless it has one, and send it `prompt`; then `session_wait`."""
        _prompt_only(prompt)
        view = await self.view(workspace, start=True)
        await view.send(prompt)
        assert view.session is not None, 'a followed view has its snapshot'
        return (
            f'Workspace {workspace} runs session {view.session.id}, listed in the sidebar under its repository; '
            'the prompt is sent. Wait with `session_wait`.'
        )

    async def session_send(self, workspace: str, text: str) -> str:
        """Send another prompt to the workspace's session; it queues behind a running turn."""
        _prompt_only(text)
        await (await self.view(workspace)).send(text)
        return 'Sent. Wait with `session_wait`.'

    async def session_read(self, workspace: str, last: int = 5) -> str:
        """The workspace's session in short: whether a turn runs, what it asks, and its `last` items."""
        view = await self.view(workspace)
        items = view.items
        title = view.session.title if view.session is not None else None
        lines = [f'Workspace {workspace}: {"a turn is running" if view.running else "idle"}; session {title!r}.']
        if view.last_turn is not None:
            lines.append(
                f'Last turn: {view.last_turn.outcome}'
                + (f' ({view.last_turn.error})' if view.last_turn.error else '')
                + '.'
            )
        if asked := _asked(view):
            lines.append(asked)
        shown = items[-last:] if last > 0 else []
        lines.append(f'Last {len(shown)} of {len(items)} items:')
        lines += [summarize(item) for item in shown]
        return '\n'.join(lines)

    async def session_wait(self, workspace: str, timeout: float = 300) -> str:
        """Wait up to `timeout` seconds until the turn ends or the session asks something; then its final answer."""
        view = await self.view(workspace)
        outcome = await view.wait(timeout=min(max(timeout, 0), MAX_WAIT))
        if outcome == 'asking':
            return f'Workspace {workspace} asks:\n{_asked(view)}\nAnswer with `session_answer`.'
        if outcome == 'timeout':
            return f'Workspace {workspace} is still running. Wait again, `session_read` it, or `session_interrupt` it.'
        if outcome == 'closed':
            link = self.links.get(workspace)
            await self.disconnect(workspace)
            # Following also ends when it falls too far behind, while the session goes on.
            if link is None or live_session(self.service, link.live_id) is None:
                return f'The session of workspace {workspace} closed.'
            return f'Lost track of the session of workspace {workspace}, which is still open. Wait again to follow it.'
        answer = next((item.text for item in reversed(view.items) if item.kind == 'assistant'), '')
        ended = view.last_turn.outcome if view.last_turn is not None else 'no turn ran'
        return f'Workspace {workspace} is idle; last turn: {ended}. Its final answer:\n{_cut(answer, ANSWER_CHARS)}'

    async def session_interrupt(self, workspace: str) -> str:
        """Stop the workspace's running turn, as Esc does."""
        view = await self.view(workspace)
        stopped = await view.client.interrupt()
        return 'Interrupted.' if stopped else 'Nothing was running.'

    async def session_answer(self, workspace: str, request: str, answers: list[SessionAnswer] | None = None) -> str:
        """Answer a question request the session asks with `answers`, one per question; none declines them all."""
        view = await self.view(workspace)
        if request not in view.questions:
            return f'Workspace {workspace} has no open question {request!r}.'
        given = None if answers is None else [_answer(answer) for answer in answers]
        await view.answer(request, given)
        return 'Answered. Wait with `session_wait`.'


def _prompt_only(text: str) -> None:
    """Refuse a `/command`: in a workspace's session one would run with no one there to confirm it."""
    if text.lstrip().startswith('/'):
        raise ValueError(
            'Send prompts only. A `/command`, such as `/workspaces destroy`, would run in the workspace with no one '
            'to confirm it; ask the user to type it themselves.'
        )


def _answer(answer: SessionAnswer) -> AskUserAnswer:
    if answer.custom is not None:
        return AskUserAnswer(header=answer.header, custom_answer=answer.custom)
    return AskUserAnswer(header=answer.header, selected=tuple(answer.selected))
