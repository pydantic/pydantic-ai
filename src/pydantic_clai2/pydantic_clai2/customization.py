"""On-demand, installed-package guidance for customizing the terminal."""

from importlib.resources import files

from pydantic_ai.capabilities import Capability


def read_clai_customization_guide() -> str:
    """Read CLAI's plugin authoring guide, including CLI UX, TUI menus, and custom model providers.

    Read before implementing or advising on CLAI customization. Includes supported
    extension points, examples, installation, testing, and source-change boundaries.
    """
    return files('pydantic_clai2').joinpath('customization.md').read_text(encoding='utf-8')


def customization_guide() -> Capability[None]:
    """Offer a small discovery hint without reading or injecting the guide yet."""
    return Capability(
        instructions=(
            'When asked to customize CLAI itself (plugins, CLI UX/UI, commands, rendering, '
            'TUI menus, models or providers), first call read_clai_customization_guide. '
            'It documents supported APIs and boundaries. Do not assume plugin APIs exist.'
        ),
        tools=[read_clai_customization_guide],
    )
