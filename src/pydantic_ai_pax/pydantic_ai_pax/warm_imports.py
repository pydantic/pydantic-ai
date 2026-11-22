"""Import the modules startup defers on a background thread once the prompt is ready.

Startup skips these so the prompt appears sooner. Without warming, the first prompt pays for
them instead: the OpenAI and Anthropic model modules load their SDKs, which takes about a second.
The import system locks each module, so a turn that needs a module the thread is still importing
waits for it instead of importing it twice.
"""

import importlib
import logging
import sys
import types
from collections.abc import Sequence
from threading import Thread

LAZY_IMPORTERS = frozenset({'pydantic_ai', 'pydantic_ai_harness', 'pydantic_ai_pax', 'pydantic_graph'})
"""Packages whose module-level imports Python 3.15 makes lazy for a launch: this repository's, not a dependency's."""
FIRST_USE_MODULES = (
    'pydantic_ai.models.openai',
    'pydantic_ai.models.anthropic',
    'pydantic_ai_pax.models.model_settings',
    'pydantic_ai_pax.auth',
    'pydantic_ai_pax.models.openrouter',
    'pydantic_ai_pax.models.vllm',
    'pydantic_ai_pax.models.github_copilot',
    'pydantic_ai_pax.ui.menus.model_menu',
    'pydantic_ai.mcp',
    'pydantic_ai_pax.mcp.client',
)
"""Every turn loads `model_settings` and the model's provider module, usually OpenAI or Anthropic; the rest serve
model resolution, `/login`, the model menus, and the first turn or sign-in with an MCP server (FastMCP and the MCP SDK)."""


def start(modules: Sequence[str] = FIRST_USE_MODULES) -> Thread:
    """Import `modules` on a background thread. A failure is logged; first use imports it again and raises."""
    # Not a daemon: finalizing the interpreter mid-import of a native extension aborts the process.
    thread = Thread(target=_import_all, args=(tuple(modules),), name='clai-warm-imports')
    thread.start()
    return thread


def _import_all(modules: tuple[str, ...]) -> None:
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:
            logging.getLogger(__name__).debug('Could not warm %s', name, exc_info=True)
    lazy_type = getattr(types, 'LazyImportType', None)
    if lazy_type is not None:  # pragma: no cover - Python 3.15, where coverage is not collected
        _bind_lazy_imports(lazy_type)


def _bind_lazy_imports(lazy_type: type) -> None:  # pragma: no cover - Python 3.15
    """Bind every import startup left lazy in `LAZY_IMPORTERS`, and those of the modules that loads, as an eager startup would have.

    Otherwise the first keypress, menu or turn that uses one imports it on the event loop.
    """
    scanned: set[str] = set()
    while pending := [
        (name, module)
        for name, module in list(sys.modules.items())
        if name not in scanned and name.partition('.')[0] in LAZY_IMPORTERS
    ]:
        for name, module in pending:
            scanned.add(name)
            namespace: dict[str, object] = getattr(module, '__dict__', {})
            for attribute, value in list(namespace.items()):
                if isinstance(value, lazy_type):
                    try:
                        namespace[attribute] = value.resolve()  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
                    except Exception:
                        logging.getLogger(__name__).debug('Could not warm %s.%s', name, attribute, exc_info=True)
