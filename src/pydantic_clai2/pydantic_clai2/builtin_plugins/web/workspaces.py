"""The sidebar's Workspaces section: the `workspaces` plugin's registry, and opening a workspace's own page.

Only this page, which has its own token, gets a workspace's token: in the address `open` returns.
"""

from pydantic_clai2.builtin_plugins.web.protocol import WorkspaceOpened, WorkspaceRow, WorkspacesInfo
from pydantic_clai2.plugins import PluginCatalog


async def listing(plugins: PluginCatalog | None) -> WorkspacesInfo:
    """Every workspace while the `workspaces` plugin is on, with each one's status checked now."""
    # Imported here: the `workspaces` plugin imports this package for the token header.
    from pydantic_clai2.builtin_plugins.workspaces import configured

    workspaces = configured(plugins)
    if workspaces is None:
        return WorkspacesInfo(enabled=False, workspaces=[])
    rows = [
        WorkspaceRow(
            id=workspace.id,
            kind=workspace.kind,
            repo=workspace.repo,
            branch=workspace.branch,
            path=str(workspace.path),
            status=workspace.status,
            url=workspace.session.url if workspace.status == 'running' and workspace.session else None,
        )
        for workspace in await workspaces.listing()
    ]
    return WorkspacesInfo(enabled=True, workspaces=rows)


async def open_workspace(plugins: PluginCatalog | None, workspace_id: str) -> WorkspaceOpened:
    """Start the workspace's CLAI unless it runs, and return its page's address with its token."""
    from pydantic_clai2.builtin_plugins.workspaces import configured

    workspaces = configured(plugins)
    if workspaces is None:
        raise LookupError('The `workspaces` plugin is off.')
    workspace = await workspaces.start(workspace_id)
    assert workspace.session is not None, 'a started workspace has a session'
    return WorkspaceOpened(url=f'{workspace.session.url}?token={await workspaces.token(workspace_id)}')
