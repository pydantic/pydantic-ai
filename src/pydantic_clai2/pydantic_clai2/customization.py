"""On-demand, installed-package guidance for customizing the terminal."""

from importlib.resources import files

from pydantic_ai.capabilities import Capability

CAPABILITY_ID = 'clai_customization'

_DESCRIPTION = (
    'Use when asked to customize CLAI itself (plugins, CLI UX/UI, commands, rendering, TUI menus, '
    'models or providers). It documents supported APIs and boundaries; do not assume plugin APIs exist.'
)


def _guide() -> str:
    """The packaged guide, read only once the model loads the capability."""
    return files('pydantic_clai2').joinpath('customization.md').read_text(encoding='utf-8')


def customization_guide() -> Capability[None]:
    """CLAI's plugin authoring guide as a deferred capability: one catalog line until the model loads it."""
    return Capability[None](id=CAPABILITY_ID, description=_DESCRIPTION, instructions=_guide, defer_loading=True)
