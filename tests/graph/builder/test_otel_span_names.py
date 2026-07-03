"""Tests that GraphBuilder preformats OTel span names for non-Logfire backends.

Logfire interpolates ``{...}`` placeholders in a span message for its own display, but the
underlying OTel span *name* is left as the literal template unless it is preformatted. Other
OTel backends use that span name verbatim, so it must already contain the interpolated values.

See https://github.com/pydantic/pydantic-ai/issues/5862 and
https://github.com/pydantic/pydantic-ai/issues/3173.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

from pydantic_graph import GraphBuilder, StepContext

if TYPE_CHECKING:
    from logfire.testing import CaptureLogfire

pytestmark = pytest.mark.anyio

logfire_installed = importlib.util.find_spec('logfire') is not None


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
async def test_graph_and_node_span_names_are_preformatted(capfire: CaptureLogfire):
    """Span names must be the interpolated values, not the literal ``{...}`` templates."""
    g = GraphBuilder(name='my_graph', output_type=int)

    @g.step
    async def my_step(ctx: StepContext[None, None, None]) -> int:
        return 42

    g.add(
        g.edge_from(g.start_node).to(my_step),
        g.edge_from(my_step).to(g.end_node),
    )
    graph = g.build()
    await graph.run(state=None)

    span_names = {span['name'] for span in capfire.exporter.exported_spans_as_dict()}
    assert 'run graph my_graph' in span_names
    assert 'run node my_step' in span_names
    # The literal templates must not leak through to the OTel span name.
    assert 'run graph {graph.name}' not in span_names
    assert 'run node {node_id}' not in span_names
