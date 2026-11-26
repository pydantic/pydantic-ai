"""Claude Agent provider: the user's logged-in Claude account, through their own unmodified Claude Code.

The provider never reads, stores or forwards Claude credentials. It starts Claude Code through the Claude Agent SDK,
and Claude Code signs its requests with whatever login the user set up for it (`claude auth login`), the way it does
when the user runs it themselves.
"""

from __future__ import annotations as _annotations

from pathlib import Path
from types import TracebackType
from typing import Self

from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.claude_agent import claude_agent_model_profile

from .anthropic import AnthropicProvider

try:
    import httpx2
    from anthropic import AsyncAnthropic

    from ._claude_agent_sessions import ClaudeCodeOptions, ClaudeCodeSessions, ClaudeCodeTransport
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `anthropic` and `claude-agent-sdk` packages to use the Claude Agent provider, '
        'you can use the `claude-agent` optional group — `pip install "pydantic-ai-slim[claude-agent]"`'
    ) from _import_error

__all__ = ('ClaudeAgentProvider',)


class ClaudeAgentProvider(AnthropicProvider):
    """Provider for the user's logged-in Claude account, on any Claude subscription that includes Claude Code.

    Requests go to a Claude Code process started through the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview),
    not over the network from this process. Each conversation keeps its process while the provider, the model or an
    agent run using it is entered, so Claude Code's prompt cache carries over from one request to the next.
    """

    def __init__(self, *, cli_path: str | Path | None = None, cwd: str | Path | None = None) -> None:
        """Create a new Claude Agent provider.

        Args:
            cli_path: The `claude` executable to run. Defaults to the one bundled with the `claude-agent-sdk` package.
                Every `claude` on the machine shares the login stored in Claude Code's configuration directory.
            cwd: The working directory Claude Code runs in, which it tells the model about. Defaults to the current
                working directory.
        """
        self._sessions = ClaudeCodeSessions(ClaudeCodeOptions(cli_path=cli_path, cwd=cwd))
        transport = ClaudeCodeTransport(self._sessions)
        client = AsyncAnthropic(
            api_key='claude-agent',
            base_url='https://claude-agent.invalid',
            http_client=httpx2.AsyncClient(transport=transport, trust_env=False),
        )
        super().__init__(anthropic_client=client)

    @property
    def name(self) -> str:
        return 'claude-agent'

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        # The Claude API's deprecation dates still apply: Claude Code serves the same models.
        return merge_profile(AnthropicProvider.model_profile(model_name), claude_agent_model_profile(model_name))

    async def __aenter__(self) -> Self:
        await super().__aenter__()
        self._sessions.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        await self._sessions.release_lease()
        return await super().__aexit__(exc_type, exc_val, exc_tb)
