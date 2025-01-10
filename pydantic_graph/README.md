# Pydantic Graph

[![CI](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/pydantic/pydantic-ai.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/pydantic/pydantic-ai)
[![PyPI](https://img.shields.io/pypi/v/pydantic-graph.svg)](https://pypi.python.org/pypi/pydantic-graph)
[![versions](https://img.shields.io/pypi/pyversions/pydantic-graph.svg)](https://github.com/pydantic/pydantic-ai)
[![license](https://img.shields.io/github/license/pydantic/pydantic-ai.svg?v)](https://github.com/pydantic/pydantic-ai/blob/main/LICENSE)

Graph and finite state machine library.

This library is developed as part of the [PydanticAI](https://ai.pydantic.dev), however it has no dependency
on `pydantic-ai` or related packages and does and can be considered as a pure graph library.

As with PydanticAI, this library prioritizes type safety and use of common Python syntax over esoteric, domain-specific use of Python syntax.

`pydantic-graph` allows you to define graphs using simple Python syntax. In particular, edges are defined using the return type hint of nodes.
