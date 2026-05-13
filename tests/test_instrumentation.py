"""Unit tests for `pydantic_ai._instrumentation`.

The bulk of the OTel-span / Logfire integration is covered in `test_logfire.py`;
this file holds focused unit-level invariants that don't need any of that machinery.
"""

from __future__ import annotations

import ast
import inspect


def test_current_otel_traceparent_has_no_lazy_imports():
    """`current_otel_traceparent` must keep all its imports at module top of
    `_instrumentation.py` — no `import` / `from ... import ...` inside the function body.

    The function is called from `AgentRun._traceparent` at the end of every agent run
    (via the `AgentRunResult` property accessor) when the graph run has no active span,
    which is the default for `Agent`'s internal graph (it sets `auto_instrument=False`).
    When that path runs inside a Temporal workflow, each `__import__` call from inside the
    function body trips Temporal's sandbox `"Module ... was imported after initial workflow
    load."` warning. The workflow worker reraises that warning as a workflow task failure
    and retries forever — the workflow run never finishes. Surfaces on any model path that
    doesn't itself open an OTel span (TestModel / FunctionModel / non-HTTP models); httpx-
    backed paths emit a request span so `_graph_run._traceparent` returns non-None and the
    fallback never fires, which is why `test_temporal.py::test_simple_agent_run_in_workflow`
    (OpenAI-backed) never caught the original regression.

    Module-top imports keep the function body to plain name lookups, no `__import__` calls
    inside the sandbox.
    """
    from pydantic_ai._instrumentation import current_otel_traceparent

    source = inspect.getsource(current_otel_traceparent)
    fn = ast.parse(source).body[0]
    assert isinstance(fn, ast.FunctionDef)
    lazy_imports = [
        ast.unparse(node) for node in ast.walk(fn) if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not lazy_imports, (
        f'`{fn.name}` must not lazy-import (would break Temporal workflow sandbox); '
        f'move these to module top of `_instrumentation.py`: {lazy_imports}'
    )
