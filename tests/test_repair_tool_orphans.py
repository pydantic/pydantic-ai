"""Tests for the RepairToolOrphans capability."""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import RepairToolOrphans
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytestmark = [pytest.mark.anyio]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DANGLING_MSG = 'Tool call was not completed.'


def _ids_of(parts: list, cls: type) -> set[str]:
    """Extract tool_call_ids from parts of a given type."""
    return {p.tool_call_id for p in parts if isinstance(p, cls)}


# ---------------------------------------------------------------------------
# Unit tests for the repair logic
# ---------------------------------------------------------------------------


class TestRepairToolOrphansUnit:
    """Unit tests that directly call the internal _repair_messages function."""

    def _repair(self, messages: list[ModelMessage], *, warn: bool = False) -> list[ModelMessage]:
        from pydantic_ai.capabilities.repair_tool_orphans import _repair_messages

        return _repair_messages(messages, warn=warn)

    def test_empty_history(self):
        assert self._repair([]) == []

    def test_clean_history_unchanged(self):
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hello')]),
            ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='tc1')]),
            ModelRequest(parts=[ToolReturnPart(tool_name='t', content='ok', tool_call_id='tc1')]),
            ModelResponse(parts=[TextPart(content='done')]),
        ]
        result = self._repair(messages)
        assert len(result) == 4
        # No synthetic parts should have been added
        req = result[2]
        assert isinstance(req, ModelRequest)
        returns = [p for p in req.parts if isinstance(p, ToolReturnPart)]
        assert len(returns) == 1
        assert returns[0].tool_call_id == 'tc1'

    def test_dangling_regular_call_with_following_request(self):
        """ToolCallPart with no return -> synthetic ToolReturnPart injected."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='tc1')]),
            ModelRequest(parts=[UserPromptPart(content='continue')]),
        ]
        result = self._repair(messages)
        assert len(result) == 3
        req = result[2]
        assert isinstance(req, ModelRequest)
        synthetic = [p for p in req.parts if isinstance(p, ToolReturnPart)]
        assert len(synthetic) == 1
        assert synthetic[0].tool_call_id == 'tc1'
        assert synthetic[0].content == _DANGLING_MSG
        # Original user prompt preserved
        assert any(isinstance(p, UserPromptPart) for p in req.parts)

    def test_dangling_regular_call_trailing(self):
        """Trailing ModelResponse with dangling call -> synthetic request appended."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='tc1')]),
        ]
        result = self._repair(messages)
        assert len(result) == 3
        req = result[2]
        assert isinstance(req, ModelRequest)
        assert len(req.parts) == 1
        assert isinstance(req.parts[0], ToolReturnPart)
        assert req.parts[0].tool_call_id == 'tc1'

    def test_dangling_builtin_call(self):
        """BuiltinToolCallPart without return -> synthetic return in same response."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(
                parts=[BuiltinToolCallPart(tool_name='web_search', args='{}', tool_call_id='btc1')]
            ),
            ModelRequest(parts=[UserPromptPart(content='continue')]),
        ]
        result = self._repair(messages)
        resp = result[1]
        assert isinstance(resp, ModelResponse)
        builtin_returns = [p for p in resp.parts if isinstance(p, BuiltinToolReturnPart)]
        assert len(builtin_returns) == 1
        assert builtin_returns[0].tool_call_id == 'btc1'
        assert builtin_returns[0].content == _DANGLING_MSG

    def test_orphaned_tool_return_removed(self):
        """ToolReturnPart with no matching call in preceding response -> removed."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[TextPart(content='no tools called')]),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='t', content='stale', tool_call_id='orphan'),
                    UserPromptPart(content='continue'),
                ]
            ),
        ]
        result = self._repair(messages)
        req = result[2]
        assert isinstance(req, ModelRequest)
        # Orphaned return should be gone
        returns = [p for p in req.parts if isinstance(p, ToolReturnPart)]
        assert len(returns) == 0
        # User prompt preserved
        assert any(isinstance(p, UserPromptPart) for p in req.parts)

    def test_orphaned_retry_prompt_removed(self):
        """RetryPromptPart with no matching call -> removed."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[TextPart(content='done')]),
            ModelRequest(
                parts=[
                    RetryPromptPart(content='retry orphan', tool_call_id='orphan'),
                    UserPromptPart(content='next'),
                ]
            ),
        ]
        result = self._repair(messages)
        req = result[2]
        assert isinstance(req, ModelRequest)
        retries = [p for p in req.parts if isinstance(p, RetryPromptPart)]
        assert len(retries) == 0

    def test_empty_request_gets_placeholder(self):
        """If removing orphans empties the request, a placeholder is inserted."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[TextPart(content='done')]),
            ModelRequest(
                parts=[ToolReturnPart(tool_name='t', content='orphan', tool_call_id='orphan')]
            ),
        ]
        result = self._repair(messages)
        req = result[2]
        assert isinstance(req, ModelRequest)
        assert len(req.parts) == 1
        assert isinstance(req.parts[0], UserPromptPart)
        assert req.parts[0].content == '...'

    def test_multiple_dangling_calls(self):
        """Multiple tool calls without returns -> all get synthetic returns."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='a', args='{}', tool_call_id='tc1'),
                    ToolCallPart(tool_name='b', args='{}', tool_call_id='tc2'),
                ]
            ),
            ModelRequest(parts=[UserPromptPart(content='continue')]),
        ]
        result = self._repair(messages)
        req = result[2]
        assert isinstance(req, ModelRequest)
        returns = [p for p in req.parts if isinstance(p, ToolReturnPart)]
        assert len(returns) == 2
        assert _ids_of(returns, ToolReturnPart) == {'tc1', 'tc2'}

    def test_partial_answer(self):
        """One of two calls answered -> only the unanswered one gets synthetic return."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='a', args='{}', tool_call_id='tc1'),
                    ToolCallPart(tool_name='b', args='{}', tool_call_id='tc2'),
                ]
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name='a', content='ok', tool_call_id='tc1')]
            ),
        ]
        result = self._repair(messages)
        req = result[2]
        assert isinstance(req, ModelRequest)
        returns = [p for p in req.parts if isinstance(p, ToolReturnPart)]
        assert len(returns) == 2
        ids = {r.tool_call_id for r in returns}
        assert ids == {'tc1', 'tc2'}
        # tc1 should have real content, tc2 synthetic
        for r in returns:
            if r.tool_call_id == 'tc1':
                assert r.content == 'ok'
            else:
                assert r.content == _DANGLING_MSG

    def test_metadata_preserved(self):
        """Request metadata should survive repair."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='tc1')]),
            ModelRequest(
                parts=[UserPromptPart(content='next')],
                instructions='keep me',
                metadata={'key': 'value'},
            ),
        ]
        result = self._repair(messages)
        req = result[2]
        assert isinstance(req, ModelRequest)
        assert req.instructions == 'keep me'
        assert req.metadata == {'key': 'value'}

    def test_warning_emitted(self):
        """When warn=True, a UserWarning should be emitted on repair."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='tc1')]),
            ModelRequest(parts=[UserPromptPart(content='continue')]),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self._repair(messages, warn=True)
            assert len(w) == 1
            assert 'RepairToolOrphans' in str(w[0].message)

    def test_no_warning_when_clean(self):
        """No warning should be emitted when no repairs are needed."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[TextPart(content='done')]),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self._repair(messages, warn=True)
            assert len(w) == 0

    def test_no_warning_when_disabled(self):
        """No warning when warn=False even if repairs happen."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='tc1')]),
            ModelRequest(parts=[UserPromptPart(content='continue')]),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self._repair(messages, warn=False)
            assert len(w) == 0

    def test_multi_turn_poisoned_conversation(self):
        """Multiple rounds of damage: dangling calls + orphaned returns."""
        messages: list[ModelMessage] = [
            # Turn 1: normal
            ModelRequest(parts=[UserPromptPart(content='start')]),
            ModelResponse(parts=[ToolCallPart(tool_name='a', args='{}', tool_call_id='tc1')]),
            ModelRequest(parts=[ToolReturnPart(tool_name='a', content='ok', tool_call_id='tc1')]),
            # Turn 2: response has dangling call, next request has orphaned return
            ModelResponse(parts=[ToolCallPart(tool_name='b', args='{}', tool_call_id='tc2')]),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='c', content='stale', tool_call_id='tc_orphan'),
                    UserPromptPart(content='help'),
                ]
            ),
            # Turn 3: clean
            ModelResponse(parts=[TextPart(content='final')]),
        ]
        result = self._repair(messages)
        # Turn 2 request should have: synthetic for tc2, user prompt, orphan removed
        req = result[4]
        assert isinstance(req, ModelRequest)
        returns = [p for p in req.parts if isinstance(p, ToolReturnPart)]
        assert len(returns) == 1
        assert returns[0].tool_call_id == 'tc2'
        assert returns[0].content == _DANGLING_MSG
        # User prompt preserved
        assert any(isinstance(p, UserPromptPart) and p.content == 'help' for p in req.parts)


# ---------------------------------------------------------------------------
# Integration test with Agent
# ---------------------------------------------------------------------------


class TestRepairToolOrphansIntegration:
    """Test the capability end-to-end with a real Agent."""

    async def test_agent_with_poisoned_history(self):
        """Agent should successfully run with a poisoned message history when
        RepairToolOrphans is enabled."""

        def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='recovered')])

        agent = Agent(
            FunctionModel(model_func),
            capabilities=[RepairToolOrphans(warn=False)],
        )

        # Poisoned history: dangling tool call
        poisoned_history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hello')]),
            ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='tc1')]),
            # Missing ToolReturnPart for tc1
            ModelRequest(parts=[UserPromptPart(content='continue')]),
            ModelResponse(parts=[TextPart(content='partial')]),
        ]

        result = await agent.run('try again', message_history=poisoned_history)
        assert result.output == 'recovered'
