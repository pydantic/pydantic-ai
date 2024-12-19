# `pydantic_ai.models.gemini`

Custom interface to the `generativelanguage.googleapis.com` API using
[HTTPX](https://www.python-httpx.org/) and [Pydantic](https://docs.pydantic.dev/latest/).

The Google SDK for interacting with the `generativelanguage.googleapis.com` API
[`google-generativeai`](https://ai.google.dev/gemini-api/docs/quickstart?lang=python) reads like it was written by a
Java developer who thought they knew everything about OOP, spent 30 minutes trying to learn Python,
gave up and decided to build the library to prove how horrible Python is. It also doesn't use httpx for HTTP requests,
and tries to implement tool calling itself, but doesn't use Pydantic or equivalent for validation.

We therefore implement support for the API directly.

Despite these shortcomings, the Gemini model is actually quite powerful and very fast.

## Setup

For details on how to set up authentication with this model, see [model configuration for Gemini](../../models.md#gemini).

::: pydantic_ai.models.gemini
