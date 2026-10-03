"""Test that WebSearchTool with unsupported constraints raises UserError on Google."""
import pytest

from pydantic_ai import Agent
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.exceptions import UserError


def test_websearch_allowed_domains_raises_usererror():
    """WebSearchTool with allowed_domains should raise UserError on Google."""
    tool = WebSearchTool(allowed_domains=['docs.example.com'])
    agent = Agent('gemini-2.0-flash', tools=[tool])

    with pytest.raises(UserError, match='does not support.*allowed_domains'):
        # We can't actually run this without API keys, but the error should
        # occur during tool setup, not at runtime
        from pydantic_ai.models.google import GoogleModel
        model = GoogleModel('gemini-2.0-flash')
        # This would be called during agent initialization
        model._build_tools(agent._prepare_request_parameters().model_request_parameters)


def test_websearch_blocked_domains_raises_usererror():
    """WebSearchTool with blocked_domains should raise UserError on Google."""
    tool = WebSearchTool(blocked_domains=['spam.com'])
    agent = Agent('gemini-2.0-flash', tools=[tool])

    with pytest.raises(UserError, match='does not support.*blocked_domains'):
        from pydantic_ai.models.google import GoogleModel
        model = GoogleModel('gemini-2.0-flash')
        model._build_tools(agent._prepare_request_parameters().model_request_parameters)


def test_websearch_max_uses_raises_usererror():
    """WebSearchTool with max_uses should raise UserError on Google."""
    tool = WebSearchTool(max_uses=5)
    agent = Agent('gemini-2.0-flash', tools=[tool])

    with pytest.raises(UserError, match='does not support.*max_uses'):
        from pydantic_ai.models.google import GoogleModel
        model = GoogleModel('gemini-2.0-flash')
        model._build_tools(agent._prepare_request_parameters().model_request_parameters)


def test_websearch_external_web_access_raises_usererror():
    """WebSearchTool with external_web_access=False should raise UserError on Google."""
    tool = WebSearchTool(external_web_access=False)
    agent = Agent('gemini-2.0-flash', tools=[tool])

    with pytest.raises(UserError, match='does not support.*external_web_access'):
        from pydantic_ai.models.google import GoogleModel
        model = GoogleModel('gemini-2.0-flash')
        model._build_tools(agent._prepare_request_parameters().model_request_parameters)


def test_websearch_multiple_constraints_raises_usererror():
    """WebSearchTool with multiple unsupported constraints lists all in error."""
    tool = WebSearchTool(
        allowed_domains=['example.com'],
        blocked_domains=['spam.com'],
        max_uses=10
    )
    agent = Agent('gemini-2.0-flash', tools=[tool])

    with pytest.raises(UserError, match='allowed_domains, blocked_domains, max_uses'):
        from pydantic_ai.models.google import GoogleModel
        model = GoogleModel('gemini-2.0-flash')
        model._build_tools(agent._prepare_request_parameters().model_request_parameters)
