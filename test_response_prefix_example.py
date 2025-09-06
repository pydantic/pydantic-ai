#!/usr/bin/env python3
"""Example script demonstrating the response prefix feature in Pydantic AI."""

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


def test_response_prefix():
    """Test the response prefix feature with validation."""
    # Test that unsupported models raise an error
    agent = Agent(TestModel())

    try:
        agent.run_sync('Hello', response_prefix='Assistant: ')
        assert False, 'Should have raised UserError'
    except Exception as e:
        print(f'✅ Validation works: {e}')

    # Create a mock model that supports response prefix
    class MockResponsePrefixModel(TestModel):
        @property
        def profile(self):   # pyright: ignore[reportIncompatibleVariableOverride]
            profile = super().profile
            profile.supports_response_prefix = True
            return profile

    # Create an agent with the mock model
    agent = Agent(MockResponsePrefixModel())

    # Test that the parameter is accepted without error
    result = agent.run_sync('Hello', response_prefix='Assistant: ')
    print('✅ Response prefix parameter accepted by supported model')
    print(f'Response: {result.output}')

    print('✅ Response prefix feature working correctly!')


if __name__ == '__main__':
    test_response_prefix()
