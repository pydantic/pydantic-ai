# Graphs

!!! danger "Don't use a nail gun unless you need a nail gun"
    If PydanticAI [agents](agents.md) are a hammer, and [multi-agent workflows](multi-agent-applications.md) are a sledgehammer, then graphs are a nail gun, with flames down the side:

    * sure, nail guns look cooler than hammers
    * but nail guns take a lot more setup than hammers
    * and nail guns don't make you a better builder, they make you a builder with a nail gun
    * Lastly, (and at the risk of torturing this metaphor), if you're a fan of medieval tools like mallets and untyped Python, you probably won't like nail guns or PydanticAI approach to graphs. (But then again, if you're not a fan of type hints in Python, you've probably already bounced off PydanticAI to use one of the toy agent frameworks — good luck, and feel free to borrow my sledgehammer)

    In short, graphs are a powerful tool, but they're not the right tool for every job. Please consider other [multi-agent approaches](multi-agent-applications.md) before proceeding. Unless you're sure you need a graph, you probably don't.

Graphs and associated finite state machines (FSMs) are a powerful abstraction to model, control and visualize complex workflows.

Alongside PydanticAI, we've developed `pydantic-graph` — an async graph and state machine library for Python where nodes and edges are defined in pure Python using type hints.

While this library is developed as part of the PydanticAI; it has no dependency on `pydantic-ai` and can be considered as a pure graph library. You may find it useful whether or not you're using PydanticAI or even building with GenAI.

## Installation

`pydantic-graph` a required dependency of `pydantic-ai`, and an optional dependency of `pydantic-ai-slim`, see [installation instructions](install.md) for more information. You can also install it directly:

```bash
pip/uv-add pydantic-graph
```

## Basic Usage

TODO

## Typing

TODO

## Running Graphs

TODO

## State Machines

TODO

## Mermaid Diagrams

TODO
