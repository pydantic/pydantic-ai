"""Workspaces in this CLAI's web UI: the session the orchestrator runs in each, and their checkouts in the sidebar.

A workspace's session is a live session of this CLAI in its checkout, as one started with New session there
is, so the sidebar lists it under its repository, by branch. Its conversation is kept in the workspace's
record, so it opens again after it closed for being idle or CLAI restarted.
"""

from typing import TYPE_CHECKING

from pydantic_clai2.builtin_plugins.web.channel import SessionChannel
from pydantic_clai2.builtin_plugins.workspaces import Workspaces, configured
from pydantic_clai2.plugins import PluginCatalog

if TYPE_CHECKING:
    from pydantic_clai2.builtin_plugins.web.service import WebService


def enabled(plugins: PluginCatalog | None) -> Workspaces:
    """The configured workspaces; `RuntimeError` while the `workspaces` plugin is off."""
    workspaces = configured(plugins)
    if workspaces is None:
        raise RuntimeError('The `workspaces` plugin is off. Ask the user to turn it on.')
    return workspaces


async def workspace_session(service: 'WebService', workspace_id: str, *, start: bool) -> SessionChannel:
    """The live session the orchestrator runs in the workspace, opened again if need be.

    With `start`, a workspace without one, or whose session was never saved or is deleted, gets a new one
    in its checkout; without, that is a `RuntimeError`. `ValueError` when the checkout is gone.
    """
    workspaces = enabled(service.host.plugins)
    workspace = await workspaces.get(workspace_id)
    if workspace.conversation is not None:
        try:
            session = await service.live.open(workspace.conversation)
        except LookupError:
            pass
        else:
            if (conversation := session.conversation.conversation_id) != workspace.conversation:
                # Another process holds it, so this one goes on in a copy: the copy is the workspace's from now on.
                await workspaces.attach(workspace_id, conversation)
            return service.channel(session)
    if not start:
        raise RuntimeError(f'Workspace {workspace_id} has no session yet. Start one with `session_start`.')
    session = await service.live.new(workspace=workspace.path)
    await workspaces.attach(workspace_id, session.conversation.conversation_id)
    return service.channel(session)


async def open_workspace(service: 'WebService', workspace_id: str) -> str:
    """Make the workspace's session live, starting one if it has none, for the page to show; its conversation ID."""
    channel = await workspace_session(service, workspace_id, start=True)
    return service.choose(channel.session)


async def checkouts(plugins: PluginCatalog | None) -> list[str]:
    """The folders of the workspaces, which the sidebar lists under their repository before any session runs there."""
    workspaces = configured(plugins)
    return [] if workspaces is None else [str(workspace.path) for workspace in await workspaces.listing()]
