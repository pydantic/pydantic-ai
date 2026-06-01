from __future__ import annotations

import sys
from io import StringIO

import pytest

import pydantic_ai_slim.pydantic_ai._display as _display
from pydantic_ai import Agent


@pytest.fixture(autouse=True)
def _reset_banner_state():
    """Reset the once-per-process banner flag between tests."""
    _display._banner_displayed = False
    yield
    _display._banner_displayed = False


class TestDisplayAgentBanner:
    def test_smoke_prints_banner(self):
        """Banner prints to stderr on first call."""
        agent = Agent('test', name='my_agent', output_type=bool, defer_model_check=True)

        capture = StringIO()
        old = sys.stderr
        sys.stderr = capture
        try:
            _display.display_agent_banner(agent)
        finally:
            sys.stderr = old

        output = capture.getvalue()
        assert 'my_agent' in output
        assert 'test' in output
        assert 'bool' in output
        assert 'pydantic-ai v' in output
        assert 'Next steps' in output

    def test_once_per_process(self):
        """Second call is a no-op."""
        agent = Agent('test', name='agent1', defer_model_check=True)

        capture = StringIO()
        old = sys.stderr
        sys.stderr = capture
        try:
            _display.display_agent_banner(agent)
        finally:
            sys.stderr = old
        first = capture.getvalue()
        assert first, 'First call should produce output'

        capture2 = StringIO()
        sys.stderr = capture2
        try:
            _display.display_agent_banner(agent)
        finally:
            sys.stderr = old
        assert capture2.getvalue() == '', 'Second call should be silent'

    def test_hide_banner_env_var(self, monkeypatch: pytest.MonkeyPatch):
        """PYDANTIC_AI_HIDE_BANNER suppresses output."""
        _display._banner_displayed = False
        monkeypatch.setenv('PYDANTIC_AI_HIDE_BANNER', '1')

        agent = Agent('test', name='hidden', defer_model_check=True)

        capture = StringIO()
        old = sys.stderr
        sys.stderr = capture
        try:
            _display.display_agent_banner(agent)
        finally:
            sys.stderr = old
        assert capture.getvalue() == '', 'Banner should be suppressed by env var'

    def test_agent_display_banner_method(self):
        """Agent.display_banner() calls through to _display.display_agent_banner."""
        agent = Agent('test', name='method_test', defer_model_check=True)

        capture = StringIO()
        old = sys.stderr
        sys.stderr = capture
        try:
            agent.display_banner()
        finally:
            sys.stderr = old

        assert 'method_test' in capture.getvalue()

    def test_tools_appear_in_banner(self):
        """Function tools are listed in the banner."""
        agent = Agent('test', name='tool_agent', defer_model_check=True)

        @agent.tool_plain
        def my_tool() -> str:
            return 'ok'

        capture = StringIO()
        old = sys.stderr
        sys.stderr = capture
        try:
            _display.display_agent_banner(agent)
        finally:
            sys.stderr = old

        assert 'my_tool' in capture.getvalue()

    def test_logfire_status_not_configured(self):
        """Logfire shows 'not configured' when no instrumentation is set."""
        agent = Agent('test', name='no_logfire', defer_model_check=True)

        capture = StringIO()
        old = sys.stderr
        sys.stderr = capture
        try:
            _display.display_agent_banner(agent)
        finally:
            sys.stderr = old

        assert 'not configured' in capture.getvalue()

    def test_model_name_string(self):
        """Model displayed as plain string when set as a string."""
        agent = Agent('openai:gpt-5-mini', name='str_model', defer_model_check=True)

        capture = StringIO()
        old = sys.stderr
        sys.stderr = capture
        try:
            _display.display_agent_banner(agent)
        finally:
            sys.stderr = old

        assert 'openai:gpt-5-mini' in capture.getvalue()

    def test_model_name_none(self):
        """Model displayed as '(not set)' when model is None."""
        agent = Agent(None, name='no_model', defer_model_check=True)

        capture = StringIO()
        old = sys.stderr
        sys.stderr = capture
        try:
            _display.display_agent_banner(agent)
        finally:
            sys.stderr = old

        assert '(not set)' in capture.getvalue()
