"""On-demand, installed-package guidance for customizing the terminal."""

from importlib.resources import files

from pydantic_ai import Tool
from pydantic_ai.capabilities import Capability, CapabilityOrdering
from pydantic_ai_harness.ask_user import AskUser
from pydantic_ai_harness.repo_context import RepoContext

_HINT = (
    'When asked to customize CLAI itself (plugins, CLI UX/UI, commands, rendering, '
    'TUI menus, models or providers), first call read_clai_customization_guide. '
    'It documents supported APIs and boundaries. Do not assume plugin APIs exist.'
)


def read_clai_customization_guide(**_ignored: object) -> str:
    """Read CLAI's plugin authoring guide, including CLI UX, TUI menus, and custom model providers.

    Read before implementing or advising on CLAI customization. Includes supported
    extension points, examples, installation, testing, and source-change boundaries.
    No arguments are needed; unexpected arguments are ignored.
    """
    return files('pydantic_clai2').joinpath('customization.md').read_text(encoding='utf-8')


class _GuideHint(Capability[None]):
    """The discovery hint, placed between the working guidance and the repository's instructions.

    Instructions follow capability order. This one is bound to the agent while the coding tools
    arrive as plugins for each run, so unconstrained it would lead the system prompt: a
    conditional aside about customizing CLAI, read before the agent learns what it is here to do.
    `wrapped_by=[AskUser, Capability]` puts it after the guidance the plugins contribute, whether
    that is the coding instructions (`Capability` carries the shipped ones) or a general aside
    (`AskUser`), and `wraps=[RepoContext]` puts it before the repository instruction file, which
    belongs next to the task, not above it. An edge to an absent capability simply does not apply;
    with none of them installed the hint keeps the default position, first, which is what it had
    before the edges existed.
    """

    def __init__(self) -> None:
        super().__init__(instructions=_HINT, tools=[Tool(read_clai_customization_guide, strict=False)])

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wrapped_by=[AskUser, Capability], wraps=[RepoContext])


def customization_guide() -> Capability[None]:
    """Offer a small discovery hint without reading or injecting the guide yet."""
    return _GuideHint()
