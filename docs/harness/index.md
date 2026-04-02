# Pydantic Harness

Composable, reusable capabilities for [Pydantic AI](https://ai.pydantic.dev/) agents.

## What is it?

Pydantic Harness provides a library of **capabilities** -- self-contained bundles of
system prompts, tools, and lifecycle hooks -- that you can attach to any Pydantic AI
agent to give it new powers without writing boilerplate.

## Installation

```bash
pip install pydantic-harness
```

Or with `uv`:

```bash
uv add pydantic-harness
```

## Quick start

```python
from pydantic_ai import Agent
from pydantic_harness import Memory, Skills

agent = Agent('openai:gpt-4o', capabilities=[Memory(), Skills()])
```

## Learn more

- [Available capabilities](capabilities/index.md)
- [Pydantic AI documentation](https://ai.pydantic.dev/)
- [GitHub repository](https://github.com/pydantic/pydantic-harness)
