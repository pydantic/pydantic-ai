"""Import the modules startup defers on a background thread once the prompt is ready.

Startup skips these so the prompt appears sooner. Without warming, the first prompt pays for
them instead: `model_settings` loads the OpenAI and Anthropic SDKs, which takes about a second.
The import system locks each module, so a turn that needs a module the thread is still importing
waits for it instead of importing it twice.
"""

import importlib
import logging
from collections.abc import Sequence
from threading import Thread

FIRST_USE_MODULES = (
    'pydantic_clai2.model_settings',
    'pydantic_clai2.auth',
    'pydantic_clai2.openrouter',
    'pydantic_clai2.vllm',
    'pydantic_clai2.github_copilot',
    'pydantic_clai2.model_menu',
)
"""Every turn loads `model_settings`; the rest serve model resolution, `/login`, and the model menus."""


def start(modules: Sequence[str] = FIRST_USE_MODULES) -> Thread:
    """Import `modules` on a daemon thread. A failure is logged; first use imports it again and raises."""
    thread = Thread(target=_import_all, args=(tuple(modules),), name='clai-warm-imports', daemon=True)
    thread.start()
    return thread


def _import_all(modules: tuple[str, ...]) -> None:
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 -- warming is an optimization; the importing caller reports the error.
            logging.getLogger(__name__).debug('Could not warm %s', name, exc_info=True)
