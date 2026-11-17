"""Workspaces in this CLAI's web UI: the session the orchestrator runs in each, and their checkouts in the sidebar.

A workspace's session is a live session of this CLAI in its checkout, as one started with New session there
is, so the sidebar lists it under its repository, by branch. Its conversation is kept in the workspace's
record, following a `/new` or `/resume` there, so it opens again after it closed for being idle or CLAI restarted.
"""

from typing import TYPE_CHECKING

from pydantic_clai2.builtin_plugins.web.channel import SessionChannel
from pydantic_clai2.builtin_plugins.workspaces import Workspaces, configured
from pydantic_clai2.builtin_plugins.workspaces.manager import works_in
from pydantic_clai2.plugins import ConversationChanged, PluginCatalog

if TYPE_CHECKING:
    from pydantic_clai2.builtin_plugins.web.service import WebService


def required_workspaces(plugins: PluginCatalog | None) -> Workspaces:
    """The configured workspaces; `RuntimeError` while the `workspaces` plugin is off."""
    workspaces = configured(plugins)
    if workspaces is None:
        raise RuntimeError('The `workspaces` plugin is off. Ask the user to turn it on.')
    return workspaces


async def workspace_session(service: 'WebService', workspace_id: str, *, start: bool) -> SessionChannel:
    """The live session the orchestrator runs in the workspace, opened again if need be.

    With `start`, a workspace without one in its checkout, or whose session was never saved, is deleted, or
    moved to another folder, gets a new one there; without, that is a `RuntimeError`. `ValueError` when the
    checkout is gone. Under a per-workspace lock, so the page and the orchestrator asking at once start one.
    """
    async with service.workspace_locks[workspace_id]:
        workspaces = required_workspaces(service.host.plugins)
        workspace = await workspaces.get(workspace_id)
        session = None
        if workspace.conversation is not None:
            try:
                session = await service.live.open(workspace.conversation)
            except LookupError:
                pass
        if session is None or not works_in(session.workspace, workspace.path):
            if not start:
                raise RuntimeError(f'Workspace {workspace_id} has no session yet. Start one with `session_start`.')
            session = await service.live.new(workspace=workspace.path)
        if (conversation := session.conversation.conversation_id) != workspace.conversation:
            # A new session, or a copy where another process holds the saved one.
            await workspaces.attach(workspace_id, conversation)
        return service.channel(session)


async def conversation_switched(service: 'WebService', event: ConversationChanged) -> None:
    """Have a workspace name the conversation its session switched to with `/new` or `/resume`, in its checkout.

    Call it before the page and the orchestrator hear of the switch. A session that left the checkout, with
    `/workspace` or a `/resume` of a session from another folder, is no longer the workspace's.
    """
    workspaces = configured(service.host.plugins)
    switched = [
        (channel.shown, channel.session)
        for channel in service.channels()
        if channel.id == event.conversation_id and channel.shown != channel.id
    ]
    if workspaces is None or not switched:
        return
    for workspace in await workspaces.listing():
        for previous, session in switched:
            if workspace.conversation == previous and works_in(session.workspace, workspace.path):
                await workspaces.attach(workspace.id, event.conversation_id)


async def open_workspace(service: 'WebService', workspace_id: str) -> str:
    """Make the workspace's session live, starting one if it has none, for the page to show; its conversation ID."""
    channel = await workspace_session(service, workspace_id, start=True)
    return service.choose(channel.session)


async def checkouts(plugins: PluginCatalog | None) -> list[str]:
    """The folders of the workspaces, which the sidebar lists under their repository before any session runs there."""
    workspaces = configured(plugins)
    return [] if workspaces is None else [str(workspace.path) for workspace in await workspaces.listing()]
