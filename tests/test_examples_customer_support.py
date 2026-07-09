from __future__ import annotations as _annotations


def test_customer_support_agent_imports():
    from pydantic_ai_examples.customer_support import agent

    assert agent.name is None or isinstance(agent.name, str)
