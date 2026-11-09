"""Import the modules startup defers on a background thread once the prompt is ready.

Startup skips these so the prompt appears sooner. Without warming, the first prompt pays for
them instead: the OpenAI and Anthropic model modules load their SDKs, which takes about a second.
Input typed meanwhile is handled once the thread finishes (`ClaiProcess.imports_warmed()`): the
import locks do not make two threads importing one package graph safe, as CPython can fail one of
the imports partway to break a lock cycle between them.
"""

import importlib
import logging
from collections.abc import Sequence
from threading import Thread

FIRST_USE_MODULES = (
    'pydantic_ai.models.openai',
    'pydantic_ai.models.anthropic',
    'pydantic_clai2.models.model_settings',
    'pydantic_clai2.auth',
    'pydantic_clai2.models.openrouter',
    'pydantic_clai2.models.vllm',
    'pydantic_clai2.models.github_copilot',
    'pydantic_clai2.ui.menus.model_menu',
    'pydantic_ai.mcp',
    'pydantic_clai2.mcp.client',
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
