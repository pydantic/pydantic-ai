# Pydantic Graph

[![CI](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/pydantic/pydantic-ai.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/pydantic/pydantic-ai)
[![PyPI](https://img.shields.io/pypi/v/pydantic-graph.svg)](https://pypi.python.org/pypi/pydantic-graph)
[![python versions](https://img.shields.io/pypi/pyversions/pydantic-graph.svg)](https://github.com/pydantic/pydantic-ai)
[![license](https://img.shields.io/github/license/pydantic/pydantic-ai.svg)](https://github.com/pydantic/pydantic-ai/blob/main/LICENSE)

Graph and finite state machine library.

This library is developed as part of [Pydantic AI](https://ai.pydantic.dev), however it has no dependency
on `pydantic-ai` or related packages and can be considered as a pure graph-based state machine library. You may find it useful whether or not you're using Pydantic AI or even building with GenAI.

As with Pydantic AI, this library prioritizes type safety and use of common Python syntax over esoteric, domain-specific use of Python syntax.

`pydantic-graph` allows you to define graphs using standard Python syntax. In particular, edges are defined using the return type hint of nodes.

Full documentation is available at [ai.pydantic.dev/graph](https://ai.pydantic.dev/graph).

Here's a basic example:

```python {noqa="I001"}
from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class DivisibleBy5(BaseNode[None, None, int]):
    foo: int

    async def run(
        self,
        ctx: GraphRunContext,
    ) -> Increment | End[int]:
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            return Increment(self.foo)


@dataclass
class Increment(BaseNode):
    foo: int

    async def run(self, ctx: GraphRunContext) -> DivisibleBy5:
        return DivisibleBy5(self.foo + 1)


fives_graph = Graph(nodes=[DivisibleBy5, Increment])
result = fives_graph.run_sync(DivisibleBy5(4))
print(result.output)
#> 5
```
