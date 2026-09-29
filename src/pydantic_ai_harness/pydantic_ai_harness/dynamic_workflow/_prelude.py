"""The sandbox helpers every workflow script can call: `agent`, `parallel`, `pipeline`, and friends.

They are fed to each sandbox session as a separate, untyped snippet before the script, so the
script's error line numbers and retry prompts refer to the script's own source. Host functions
only take keyword arguments; these helpers are what give the script positional forms.

A catalog sub-agent whose name matches a helper keeps its name: that helper is left out.
"""

from __future__ import annotations

from collections.abc import Collection

HOST_AGENT = '_dw_agent'
HOST_WORKFLOW = '_dw_workflow'
HOST_LOG = '_dw_log'
HOST_PHASE = '_dw_phase'
HOST_BUDGET = '_dw_budget'

HOST_ASYNC_NAMES = frozenset({HOST_AGENT, HOST_WORKFLOW})
"""Host functions the helpers `await`: dispatched concurrently."""

HOST_INLINE_NAMES = frozenset({HOST_LOG, HOST_PHASE, HOST_BUDGET})
"""Host functions the helpers call synchronously: answered inline, without a barrier on fan-out."""

ARGS_NAME = 'args'

_SHARED = """\
import asyncio


async def _dw_resolve(value):
    if type(value).__name__ == 'coroutine':
        return await value
    return value


async def _dw_settle(item):
    try:
        if type(item).__name__ != 'coroutine':
            item = item()
        return await _dw_resolve(item)
    except Exception:
        return None


async def _dw_chain(item, index, stages):
    prev = item
    for stage in stages:
        try:
            prev = await _dw_resolve(stage(prev, item, index))
        except Exception:
            return None
        if prev is None:
            return None
    return prev


def _dw_items(items, helper):
    items = list(items)
    if len(items) > {max_items}:
        raise ValueError(helper + '() got ' + str(len(items)) + ' items; the limit is {max_items} per call')
    return items
"""

_HELPERS: dict[str, tuple[str, str]] = {
    # name: (definition, type-check stub)
    'agent': (
        f"""\
async def agent(task, *, name=None, schema=None, model=None, phase=None):
    return await {HOST_AGENT}(task=task, name=name, schema=schema, model=model, phase=phase)
""",
        'async def agent(task: str, *, name: str | None = None, schema: dict[str, Any] | None = None, '
        'model: str | None = None, phase: str | None = None) -> Any: ...',
    ),
    'parallel': (
        """\
async def parallel(tasks):
    return list(await asyncio.gather(*[_dw_settle(task) for task in _dw_items(tasks, 'parallel')]))
""",
        'async def parallel(tasks: list[Any]) -> list[Any]: ...',
    ),
    'pipeline': (
        """\
async def pipeline(items, *stages):
    items = _dw_items(items, 'pipeline')
    return list(await asyncio.gather(*[_dw_chain(item, index, stages) for index, item in enumerate(items)]))
""",
        'async def pipeline(items: list[Any], *stages: Any) -> list[Any]: ...',
    ),
    'workflow': (
        f"""\
async def workflow(name, args=None):
    return await {HOST_WORKFLOW}(name=name, args=args)
""",
        'async def workflow(name: str, args: dict[str, Any] | None = None) -> Any: ...',
    ),
    'log': (
        f"""\
def log(message):
    {HOST_LOG}(message=str(message))
""",
        'def log(message: object) -> None: ...',
    ),
    'phase': (
        f"""\
def phase(title):
    {HOST_PHASE}(title=str(title))
""",
        'def phase(title: object) -> None: ...',
    ),
    'budget': (
        f"""\
def budget():
    return {HOST_BUDGET}()
""",
        'def budget() -> dict[str, int]: ...',
    ),
}

HELPER_NAMES = frozenset({*_HELPERS, ARGS_NAME})
"""Every name the prelude can bind in a script's globals."""


def render_prelude(*, shadowed: Collection[str], max_items: int) -> str:
    """The helper definitions, minus those a catalog sub-agent's name shadows."""
    definitions = [definition for name, (definition, _) in _HELPERS.items() if name not in shadowed]
    return '\n\n'.join([_SHARED.format(max_items=max_items), *definitions])


def render_helper_stubs(*, shadowed: Collection[str]) -> list[str]:
    """Type-check stubs for the helpers and the `args` global, minus shadowed names."""
    stubs = [stub for name, (_, stub) in _HELPERS.items() if name not in shadowed]
    if ARGS_NAME not in shadowed:
        stubs.insert(0, f'{ARGS_NAME}: dict[str, Any]')
    return stubs
