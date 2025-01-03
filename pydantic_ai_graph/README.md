# PydanticAI Graph

[![CI](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/pydantic/pydantic-ai.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/pydantic/pydantic-ai)
[![PyPI](https://img.shields.io/pypi/v/pydantic-ai-graph.svg)](https://pypi.python.org/pypi/pydantic-ai-graph)
[![versions](https://img.shields.io/pypi/pyversions/pydantic-ai-graph.svg)](https://github.com/pydantic/pydantic-ai)
[![license](https://img.shields.io/github/license/pydantic/pydantic-ai-graph.svg?v)](https://github.com/pydantic/pydantic-ai/blob/main/LICENSE)

Graph and state machine library.

This library is developed as part of the [PydanticAI](https://ai.pydantic.dev), however it has no dependency
on `pydantic-ai` or related packages and can be considered as a pure graph library.

As with PydanticAI, this library prioritizes type safety and use of common Python syntax over esoteric, domain-specific use of Python syntax.

`pydantic-ai-graph` allows you to define graphs using simple Python syntax. In particular, edges are defined using the return type hint of nodes. 

When designing your graph and state machine, you need to identify the data types for the overall graph input, the final graph output, the graph dependency object and graph state. Then for each specific node in the graph, you have to identify the specific data type each node is expected to receive as the input type from the prior node in the graph during transitions.

Once the nodes in the graph are defined, you can use certain built-in methods on the Graph object to visualize the nodes 
and state transitions on the graph as mermaid diagrams.
