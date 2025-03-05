"""Tests for the _build_graph_deps method."""

from typing import Any, cast

import pytest
from opentelemetry.trace import NoOpTracer, Span

from pydantic_ai import Agent
from pydantic_ai._agent_graph import GraphAgentDeps
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

pytestmark = pytest.mark.anyio


class TestBuildGraphDeps:
    """Tests for the _build_graph_deps method that was added in the commit."""

    async def test_reflect_on_tool_call_passed_to_graph_deps(self) -> None:
        """Test that reflect_on_tool_call is passed correctly to GraphAgentDeps."""
        # Create an agent with reflect_on_tool_call=True (the default)
        agent_true = Agent(TestModel())

        # Create mock parameters for _build_graph_deps
        deps = None
        user_prompt = 'test prompt'
        new_message_index = 0
        model_used = TestModel()
        model_settings = None
        usage_limits = UsageLimits()
        result_schema = None
        result_validators: list[Any] = []
        # Create a mock Span instance instead of using the class
        run_span = cast(Span, object())
        tracer = NoOpTracer()

        # Call the method
        graph_deps = agent_true._build_graph_deps(  # pyright: ignore[reportPrivateUsage]
            deps,
            user_prompt,
            new_message_index,
            model_used,
            model_settings,
            usage_limits,
            result_schema,
            result_validators,
            run_span,
            tracer,
        )

        # Verify that reflect_on_tool_call was correctly passed to GraphAgentDeps
        assert isinstance(graph_deps, GraphAgentDeps)
        assert graph_deps.reflect_on_tool_call is True

        # Create an agent with reflect_on_tool_call=False
        agent_false = Agent(TestModel(), reflect_on_tool_call=False)

        # Call the method with the second agent
        graph_deps = agent_false._build_graph_deps(  # pyright: ignore[reportPrivateUsage]
            deps,
            user_prompt,
            new_message_index,
            model_used,
            model_settings,
            usage_limits,
            result_schema,
            result_validators,
            run_span,
            tracer,
        )

        # Verify that reflect_on_tool_call=False was correctly passed to GraphAgentDeps
        assert isinstance(graph_deps, GraphAgentDeps)
        assert graph_deps.reflect_on_tool_call is False
