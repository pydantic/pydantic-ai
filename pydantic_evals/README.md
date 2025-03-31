# Pydantic Evals

[![CI](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/pydantic/pydantic-ai.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/pydantic/pydantic-ai)
[![PyPI](https://img.shields.io/pypi/v/pydantic-evals.svg)](https://pypi.python.org/pypi/pydantic-evals)
[![python versions](https://img.shields.io/pypi/pyversions/pydantic-evals.svg)](https://github.com/pydantic/pydantic-ai)
[![license](https://img.shields.io/github/license/pydantic/pydantic-ai.svg)](https://github.com/pydantic/pydantic-ai/blob/main/LICENSE)

This is a library for evaluating non-deterministic (or "stochastic") functions in Python. It provides a simple,
Pythonic interface for defining and running stochastic functions, and analyzing the results of running those functions.

While this library is developed as part of [PydanticAI](https://ai.pydantic.dev), it only uses PydanticAI for a small
subset of generative functionality internally, and it is designed to be used with arbitrary "stochastic function"
implementations. In particular, it can be used with other (non-PydanticAI) AI libraries, agent frameworks, etc.

As with PydanticAI, this library prioritizes type safety and use of common Python syntax over esoteric, domain-specific
use of Python syntax.

Full documentation is available at [ai.pydantic.dev/evals](https://ai.pydantic.dev/evals).

[//]: # (TODO: Add a basic example here.)
[//]: # (TODO: Add a note about how you can view the results in the terminal or in any OTel sink, e.g. Logfire.)
