"""Integrations with managed-configuration services for Pydantic AI agents.

Each submodule connects a specific managed-configuration provider to agent
fields, letting you externalize runtime configuration (instructions, model
settings, and more over time) without redeploying.

Currently supported:

- [`pydantic_ai.managed.logfire`][pydantic_ai.managed.logfire]: Logfire managed variables.
"""
