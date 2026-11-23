"""Every workspace backend Pydantic AI and the harness ship is checked by `WorkspaceBackendSuite` in CI, or says why not.

`BACKENDS` classifies each `WorkspaceBackend` class the two packages define, found by walking them, so a new backend
fails here until it names the suite subclass that checks it on every PR: a fake or local tier, not a live one. A
backend whose provider SDK needs asyncio says so, and must then refuse Trio with a `UserError` before it makes any
request, rather than fail somewhere inside the SDK.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import Workspace, WorkspaceBackend
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite

from ..capability_cases import caused_by_missing_dependency
from ..manifests import Exempt, assert_classified


@dataclass(frozen=True)
class Checked:
    """A backend that the `WorkspaceBackendSuite` subclass `suite`, as `<module>::<class>`, checks on every PR."""

    suite: str
    asyncio_only: bool = False
    """Whether the backend's provider SDK needs asyncio."""
    arguments: Mapping[str, str] = field(default_factory=dict[str, str])
    """What the Trio check constructs an asyncio-only backend with."""

    @property
    def reason(self) -> str:
        return f'checked by `{self.suite}`'


_WRAPPER = Exempt("a facade or wrapper over another backend, which that backend's own suite checks")


def _supplied(backend: str) -> Exempt:
    return Exempt(f'the `{backend}` its capability builds for one run and closes when the run ends')


BACKENDS: dict[str, Checked | Exempt] = {
    'pydantic_ai.workspaces._host.HostWorkspaceBackend': Exempt(
        'the base of `LocalWorkspaceBackend` and `WindowsWorkspaceBackend`, checked through them'
    ),
    'pydantic_ai.workspaces.local.LocalWorkspaceBackend': Checked(
        'tests.test_workspace_conformance::TestLocalWorkspaceBackend'
    ),
    'pydantic_ai.workspaces.windows.WindowsWorkspaceBackend': Checked(
        'tests.test_windows_workspace::TestWindowsWorkspaceBackend'
    ),
    'pydantic_ai.workspaces.unavailable.UnavailableWorkspace': Exempt(
        'every operation raises `WorkspaceUnavailableError`'
    ),
    'pydantic_ai.workspaces.workspace.Workspace': _WRAPPER,
    'pydantic_ai.workspaces.workspace.WrapperWorkspace': _WRAPPER,
    'pydantic_ai.workspaces.authority.AuthorityWorkspace': _WRAPPER,
    'pydantic_ai.workspaces.readonly.ReadOnlyWorkspace': _WRAPPER,
    'pydantic_ai_harness.boxd_sandbox._backend.BoxdSandboxBackend': Checked(
        'tests.harness.boxd_sandbox.test_conformance::TestFakeBoxdSandboxBackend', asyncio_only=True
    ),
    'pydantic_ai_harness.boxd_sandbox._capability._SuppliedBackend': _supplied('BoxdSandboxBackend'),
    'pydantic_ai_harness.boxlite_sandbox._backend.BoxLiteSandboxBackend': Checked(
        'tests.harness.boxlite_sandbox.test_conformance::TestFakeBoxLiteSandboxBackend', asyncio_only=True
    ),
    'pydantic_ai_harness.bubblewrap_sandbox._workspace.BubblewrapWorkspace': Checked(
        'tests.harness.bubblewrap_sandbox.test_conformance::TestBubblewrapAroundSSH'
    ),
    'pydantic_ai_harness.bubblewrap_sandbox._workspace._SandboxedCommands': Exempt(
        'the command half of `BubblewrapWorkspace`, checked through it'
    ),
    'pydantic_ai_harness.daytona_sandbox._backend.DaytonaSandboxBackend': Checked(
        'tests.harness.daytona_sandbox.test_conformance::TestFakeDaytonaSandboxBackend', asyncio_only=True
    ),
    'pydantic_ai_harness.daytona_sandbox._capability._SuppliedBackend': _supplied('DaytonaSandboxBackend'),
    'pydantic_ai_harness.docker_container._backend.DockerContainerBackend': Checked(
        'tests.harness.docker_container.test_conformance::TestFakeDocker'
    ),
    'pydantic_ai_harness.e2b_sandbox._backend.E2BSandboxBackend': Checked(
        'tests.harness.e2b_sandbox.test_conformance::TestFakeE2BSandboxBackend', asyncio_only=True
    ),
    'pydantic_ai_harness.kubernetes_sandbox._backend.KubernetesSandboxBackend': Checked(
        'tests.harness.kubernetes_sandbox.test_conformance::TestFakeKubernetesSandboxBackend',
        asyncio_only=True,
        arguments={'warm_pool': 'pool'},
    ),
    'pydantic_ai_harness.kubernetes_sandbox._capability._SuppliedBackend': _supplied('KubernetesSandboxBackend'),
    'pydantic_ai_harness.modal_sandbox._backend.ModalSandboxBackend': Checked(
        'tests.harness.modal_sandbox.test_conformance::TestFakeModalSandboxBackend', asyncio_only=True
    ),
    'pydantic_ai_harness.opensandbox_sandbox._backend.OpenSandboxBackend': Checked(
        'tests.harness.opensandbox_sandbox.test_conformance::TestFakeOpenSandboxBackend',
        asyncio_only=True,
        arguments={'image': 'python:3.13'},
    ),
    'pydantic_ai_harness.opensandbox_sandbox._capability._SuppliedBackend': _supplied('OpenSandboxBackend'),
    'pydantic_ai_harness.render_sandbox._backend.RenderSandboxBackend': Checked(
        'tests.harness.render_sandbox.test_conformance::TestFakeRenderSandboxBackend', asyncio_only=True
    ),
    'pydantic_ai_harness.smol_sandbox._backend.SmolSandboxBackend': Checked(
        'tests.harness.smol_sandbox.test_conformance::TestFakeSmolSandboxBackend', asyncio_only=True
    ),
    'pydantic_ai_harness._smolvm_cli.SmolvmCliBackend': Checked(
        'tests.harness.smol_sandbox.test_smolvm_cli::TestFakeSmolvmCliBackend'
    ),
    'pydantic_ai_harness.sprites_sandbox._backend.SpritesSandboxBackend': Checked(
        'tests.harness.sprites_sandbox.test_conformance::TestFakeSpritesSandboxBackend', asyncio_only=True
    ),
    'pydantic_ai_harness.sprites_sandbox._capability._SuppliedBackend': _supplied('SpritesSandboxBackend'),
    'pydantic_ai_harness.ssh_workspace._backend.SSHWorkspaceBackend': Checked(
        'tests.harness.ssh_workspace.test_conformance::TestSSHWorkspaceBackend'
    ),
}
"""Each shipped backend class, by `<module>.<qualname>`."""


def _shipped_backends() -> tuple[set[str], bool]:
    """Every `WorkspaceBackend` class the packages define, by name, and whether every module could be imported."""
    found: set[str] = set()
    complete = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for package_name in ('pydantic_ai.workspaces', 'pydantic_ai_harness'):
            package = importlib.import_module(package_name)
            for module_info in pkgutil.walk_packages(package.__path__, f'{package_name}.'):
                try:
                    module = importlib.import_module(module_info.name)
                except Exception as error:  # pragma: no cover - CI installs every extra
                    if not caused_by_missing_dependency(error):
                        raise
                    complete = False
                    continue
                for value in vars(module).values():
                    if (
                        isinstance(value, type)
                        and value is not WorkspaceBackend
                        and WorkspaceBackend in value.__mro__
                        and value.__module__ == module.__name__
                    ):
                        found.add(f'{value.__module__}.{value.__qualname__}')
    return found, complete


def test_every_shipped_backend_is_classified() -> None:
    found, complete = _shipped_backends()
    assert_classified(found, BACKENDS, complete_walk=complete, table_name='BACKENDS')


def _resolve(path: str, separator: str) -> object:
    module, _, name = path.rpartition(separator)
    try:
        return getattr(importlib.import_module(module), name)
    except ImportError as error:  # pragma: lax no cover - CI installs every extra on the platforms that have it
        assert caused_by_missing_dependency(error)
        pytest.skip(str(error))


@pytest.mark.parametrize('suite', sorted({entry.suite for entry in BACKENDS.values() if isinstance(entry, Checked)}))
def test_each_checking_suite_exists(suite: str) -> None:
    suite_class = _resolve(suite, '::')
    assert isinstance(suite_class, type) and issubclass(suite_class, WorkspaceBackendSuite)
    assert not suite_class.__name__.startswith('TestLive'), 'a live tier does not run on every PR'


@pytest.mark.parametrize('anyio_backend', ['trio'])
@pytest.mark.parametrize('operation', ['working_dir', 'run', 'exists'])
@pytest.mark.parametrize(
    'backend', [name for name, entry in BACKENDS.items() if isinstance(entry, Checked) and entry.asyncio_only]
)
async def test_an_asyncio_only_backend_refuses_trio_before_any_request(backend: str, operation: str) -> None:
    entry = BACKENDS[backend]
    assert isinstance(entry, Checked)
    backend_class = _resolve(backend, '.')
    assert isinstance(backend_class, type)
    try:
        workspace = backend_class(**entry.arguments)
    except ImportError as error:  # pragma: lax no cover - CI installs every extra on the platforms that have it
        assert caused_by_missing_dependency(error)
        pytest.skip(str(error))
    assert isinstance(workspace, WorkspaceBackend)
    operations: dict[str, Callable[[], Awaitable[object]]] = {
        'working_dir': workspace.working_dir,
        'run': lambda: Workspace(workspace).run(['true']),
        # A file operation goes to the backend's own file API when it has one, else through its shell.
        'exists': lambda: Workspace(workspace).exists('/pydantic-ai-trio-check'),
    }
    with pytest.raises(UserError, match='asyncio event loop'):
        await operations[operation]()
