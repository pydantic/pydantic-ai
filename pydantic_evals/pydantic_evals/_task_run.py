from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from .otel._context_subtree import context_subtree
from .otel.span_tree import SpanTree


@dataclass
class TaskRun:
    """Internal accumulator for attributes and metrics recorded during one run."""

    attributes: dict[str, Any] = field(init=False, default_factory=dict[str, Any])
    metrics: dict[str, int | float] = field(init=False, default_factory=dict[str, int | float])

    def record_metric(self, name: str, value: int | float) -> None:
        self.metrics[name] = value

    def increment_metric(self, name: str, amount: int | float) -> None:
        current_value = self.metrics.get(name, 0)
        incremented_value = current_value + amount
        if current_value == 0 and incremented_value == 0:
            return
        self.record_metric(name, incremented_value)

    def record_attribute(self, name: str, value: Any) -> None:
        self.attributes[name] = value


@contextmanager
def run_task() -> Iterator[Callable[[], dict[str, Any]]]:
    task_run = TaskRun()
    token = CURRENT_TASK_RUN.set(task_run)

    def get_eval_context_kwargs() -> dict[str, Any]:
        return {
            'attributes': task_run.attributes,
            'metrics': task_run.metrics,
            'duration': duration,
            '_span_tree': span_tree,
        }

    try:
        with context_subtree() as span_tree:
            t0 = time.perf_counter()
            yield get_eval_context_kwargs
            duration = time.perf_counter() - t0
    finally:
        CURRENT_TASK_RUN.reset(token)
    if isinstance(span_tree, SpanTree):  # pragma: no branch
        extract_span_tree_metrics(task_run, span_tree)


def extract_span_tree_metrics(task_run: TaskRun, span_tree: SpanTree) -> None:
    """Extract standard metrics (requests, cost, token usage) from a span tree."""
    for node in span_tree:
        if 'gen_ai.request.model' not in node.attributes:
            continue
        for k, v in node.attributes.items():
            if k == 'gen_ai.operation.name' and v == 'chat':
                task_run.increment_metric('requests', 1)
            elif not isinstance(v, int | float):
                continue
            elif k == 'operation.cost':
                task_run.increment_metric('cost', v)
            elif k.startswith('gen_ai.usage.details.'):
                task_run.increment_metric(k.removeprefix('gen_ai.usage.details.'), v)
            elif k.startswith('gen_ai.usage.'):
                task_run.increment_metric(k.removeprefix('gen_ai.usage.'), v)


CURRENT_TASK_RUN = ContextVar[TaskRun | None]('CURRENT_TASK_RUN', default=None)
