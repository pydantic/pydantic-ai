"""Modal sandbox capability: gives agents an isolated cloud sandbox as their workspace.

`ModalSandbox` is the supported entry point; build an agent with it and add tools that
use `ctx.workspace`, such as `Shell` and `FileSystem`. `ModalSandboxBackend` is the Modal
implementation of Pydantic AI's workspace backend protocol, public for applications that
want to create or attach to a sandbox themselves and pass it to a run as `workspace=`.
"""

from pydantic_ai_harness.modal_sandbox._backend import ModalSandboxBackend
from pydantic_ai_harness.modal_sandbox._capability import UPGRADE_DOCS_URL as _UPGRADE_DOCS_URL
from pydantic_ai_harness.modal_sandbox._capability import ModalSandbox

__all__ = ['ModalSandbox', 'ModalSandboxBackend']

# Public names of the previous `ModalSandbox` release, each mapped to what replaces it. Importing
# one raises an `ImportError` that says where to go, instead of Python's bare "cannot import name".
_REMOVED_NAMES: dict[str, str] = {
    'ModalSandboxSession': (
        'To share a sandbox you own across runs, pass `ModalSandboxBackend(sandbox=<modal.Sandbox>)` '
        'as `workspace=` to `agent.run()`.'
    ),
    'ModalSandboxExecResult': '`backend.run(...)` returns a `pydantic_ai.workspaces.CommandResult`.',
    'ModalSandboxError': 'Catch `pydantic_ai.workspaces.WorkspaceError`.',
    'ModalSandboxTerminalError': (
        'Catch `pydantic_ai.workspaces.WorkspaceUnavailableError`; there is no separate terminal error base.'
    ),
    'ModalSandboxUnavailableError': 'Catch `pydantic_ai.workspaces.WorkspaceUnavailableError`.',
    'ModalSandboxAuthError': (
        'Rejected credentials raise `pydantic_ai.workspaces.WorkspaceUnavailableError`; catch that.'
    ),
}


def __getattr__(name: str) -> object:
    if (replacement := _REMOVED_NAMES.get(name)) is not None:
        raise ImportError(
            f'`{name}` was removed from `pydantic_ai_harness.modal_sandbox`. {replacement} See {_UPGRADE_DOCS_URL}',
            name=name,
        )
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
