"""LOCKED: the behavior `ClaudeCodeModel` must keep to stay within Anthropic's terms for Claude subscriptions.

Anthropic lets a user sign in to the *unmodified* Claude Code with their own subscription, and forbids third parties to
collect, store or intermediate Claude credentials, or to route requests through a subscription on behalf of their users
(https://code.claude.com/docs/en/legal-and-compliance). `ClaudeCodeModel` stays within that by running the user's own
signed-in Claude Code through the Claude Agent SDK, and these tests pin each part of that:

- Claude Code is the SDK's own process for the user's `claude` executable, unmodified and with no extra CLI flags.
- Pydantic AI never reads, holds or forwards a Claude credential, and never opens Claude Code's configuration directory.
- Every Claude Code process identifies the client as Pydantic AI.
- Claude Code's own tools, settings and skills are off, and nothing can turn them on. The user's own MCP servers and
  claude.ai connectors are off unless the user's code opts in with `claude_code_inherit_mcp_servers`.
- The environment Claude Code gets can't switch it from the user's login to an API key, proxy or cloud provider.

Don't change, skip or delete these tests to make a change pass. A failure here means the change moves `ClaudeCodeModel`
outside what was agreed as compliant: bring it to a maintainer, who decides, and who updates this file in a PR of its
own.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent

from ..conftest import try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic
    from claude_agent_sdk import ClaudeAgentOptions

    import pydantic_ai.models.claude_code
    import pydantic_ai.profiles.claude_code
    import pydantic_ai.providers.claude_code
    from pydantic_ai import __version__
    from pydantic_ai.models.claude_code import ClaudeCodeModel, ClaudeCodeModelSettings
    from pydantic_ai.providers import _claude_code_mcp, _claude_code_sessions
    from pydantic_ai.providers._claude_code_sessions import ClaudeCodeTransport
    from pydantic_ai.providers.claude_code import ClaudeCodeProvider

    from .claude_code_fake import FakeClaudeCode, FakeResponse, text, tool_use

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='claude-agent-sdk not installed'),
    pytest.mark.usefixtures('allow_model_requests'),
]


def fake_claude_code(provider: ClaudeCodeProvider, *responses: FakeResponse) -> FakeClaudeCode:
    fake = FakeClaudeCode(list(responses))
    provider._sessions.transport_factory = fake  # pyright: ignore[reportPrivateUsage]
    return fake


def chosen_options(options: ClaudeAgentOptions) -> dict[str, Any]:
    """Every option Pydantic AI sets that differs from the SDK's default, with in-process objects named by shape."""
    chosen: dict[str, Any] = {}
    for option in dataclasses.fields(ClaudeAgentOptions):
        value = getattr(options, option.name)
        default = option.default_factory() if option.default_factory is not dataclasses.MISSING else option.default
        if value != default:
            chosen[option.name] = value
    chosen['mcp_servers'] = {name: server['type'] for name, server in chosen['mcp_servers'].items()}
    chosen['hooks'] = {
        event: [(matcher.matcher, len(matcher.hooks)) for matcher in matchers]
        for event, matchers in chosen['hooks'].items()
    }
    return chosen


async def test_claude_code_runs_with_exactly_these_options():
    """The whole set of options Claude Code gets: a new or changed option fails here, whatever it is."""
    provider = ClaudeCodeProvider()
    fake = fake_claude_code(
        provider,
        FakeResponse([tool_use('toolu_1', 'get_city', {})], stop_reason='tool_use'),
        FakeResponse([text('Lisbon.')]),
    )
    agent = Agent(ClaudeCodeModel('claude-sonnet-5-5', provider=provider), instructions='Be brief.')

    @agent.tool_plain
    def get_city() -> str:
        return 'Lisbon'

    await agent.run('Where?')

    [session] = fake.sessions
    assert chosen_options(session.options) == snapshot(
        {
            'tools': [],
            'system_prompt': 'Be brief.',
            'mcp_servers': {'pydantic_ai': 'sdk'},
            'strict_mcp_config': True,
            'model': 'claude-sonnet-5-5',
            'env': {
                'ANTHROPIC_API_KEY': '',
                'ANTHROPIC_AUTH_TOKEN': '',
                'ANTHROPIC_BASE_URL': '',
                'CLAUDE_CODE_USE_BEDROCK': '',
                'CLAUDE_CODE_USE_VERTEX': '',
                'CLAUDE_CODE_USE_FOUNDRY': '',
                'CLAUDE_CODE_USE_GATEWAY': '',
                'CLAUDE_CODE_USE_MANTLE': '',
                'CLAUDE_CODE_USE_ANTHROPIC_AWS': '',
                'CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD': '',
                'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
                'CLAUDE_CODE_DISABLE_ATTACHMENTS': '1',
                'CLAUDE_CODE_DISABLE_CLAUDE_MDS': '1',
                'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1',
                'DISABLE_AUTO_COMPACT': '1',
                'DISABLE_COMPACT': '1',
                'CLAUDE_CODE_MAX_RETRIES': '0',
                'CLAUDE_CODE_PROMPT_CACHE_TTL': '5m',
                'CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK': '1',
                'CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK': '1',
                'MCP_TOOL_TIMEOUT': '86400000',
                'MAX_MCP_OUTPUT_TOKENS': '10000000',
                'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '128000',
            },
            'hooks': {'PreToolUse': [('^mcp__pydantic_ai__', 1)]},
            'include_partial_messages': True,
            'verbatim_prompts': True,
            'setting_sources': [],
            'skills': [],
            'client_app': f'pydantic-ai/{__version__}',
        }
    )


async def test_claude_code_identifies_the_client_as_pydantic_ai():
    provider = ClaudeCodeProvider()
    fake = fake_claude_code(provider, FakeResponse([text('Hi.')]))

    await Agent(ClaudeCodeModel('claude-sonnet-5-5', provider=provider)).run('Hello')

    assert fake.sessions[0].options.client_app == f'pydantic-ai/{__version__}'


async def test_only_the_opt_in_brings_the_users_own_mcp_servers():
    """`claude_code_inherit_mcp_servers` loosens strict MCP config alone: tools, settings and skills stay off."""
    provider = ClaudeCodeProvider()
    fake = fake_claude_code(provider, FakeResponse([text('Hi.')]))
    fake.mcp_servers = ['claude.ai Docs']
    settings = ClaudeCodeModelSettings(claude_code_inherit_mcp_servers=True)

    await Agent(ClaudeCodeModel('claude-sonnet-5-5', provider=provider), model_settings=settings).run('Hello')

    options = fake.sessions[0].options
    assert (options.strict_mcp_config, options.tools, options.setting_sources, options.skills) == snapshot(
        (False, [], [], [])
    )


def test_claude_code_is_the_sdk_process_for_the_users_own_executable():
    """Outside tests, the SDK starts the `claude` executable it bundles, or the one the user names, as is."""
    assert ClaudeCodeProvider()._sessions.transport_factory is None  # pyright: ignore[reportPrivateUsage]
    assert ClaudeCodeProvider(cli_path='/opt/claude/bin/claude')._sessions.options.cli_path == '/opt/claude/bin/claude'  # pyright: ignore[reportPrivateUsage]


def test_nothing_can_loosen_the_options_or_take_a_credential():
    """Neither the provider nor the model takes a credential, or an option that would turn Claude Code's own on."""
    assert list(inspect.signature(ClaudeCodeProvider.__init__).parameters) == snapshot(['self', 'cli_path', 'cwd'])
    assert list(inspect.signature(ClaudeCodeModel.__init__).parameters) == snapshot(
        ['self', 'model_name', 'provider', 'profile', 'settings']
    )


def test_the_anthropic_client_never_leaves_the_process(monkeypatch: pytest.MonkeyPatch):
    """`AnthropicModel`'s client holds a placeholder key, ignores the environment, and only reaches Claude Code."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-from-the-environment')
    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://proxy.example.com')

    client = ClaudeCodeProvider().client

    assert isinstance(client, AsyncAnthropic)
    assert (client.api_key, client.auth_token, str(client.base_url)) == snapshot(
        ('claude-code', None, 'https://claude-code.invalid')
    )
    assert isinstance(client._client._transport, ClaudeCodeTransport)  # pyright: ignore[reportPrivateUsage]
    assert client._client.trust_env is False  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def opened_paths() -> Iterator[list[str]]:
    """Every file the process opens while the fixture is active, as an audit hook sees it."""
    opened: list[str] = []
    active = True

    def hook(event: str, args: tuple[Any, ...]) -> None:  # pragma: no cover (coverage doesn't trace audit hooks)
        if active and event == 'open' and isinstance(args[0], (str, bytes, Path)):
            opened.append(str(args[0]))

    sys.addaudithook(hook)  # An audit hook can't be removed, so it stops recording instead.
    yield opened
    active = False


async def test_no_claude_configuration_or_credential_file_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opened_paths: list[str]
):
    config_dir = tmp_path / 'claude-config'
    config_dir.mkdir()
    (config_dir / '.credentials.json').write_text('{}')
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(config_dir))
    already_opened = len(opened_paths)  # The test's own setup above.

    provider = ClaudeCodeProvider()
    fake_claude_code(provider, FakeResponse([text('Hi.')]))
    async with Agent(ClaudeCodeModel('claude-sonnet-5-5', provider=provider)) as agent:
        await agent.run('Hello')

    claude_dirs = (str(config_dir), str(Path.home() / '.claude'))
    assert [path for path in opened_paths[already_opened:] if path.startswith(claude_dirs)] == []


FORBIDDEN_NAMES = frozenset({'environ', 'getenv', 'open', 'read_text', 'read_bytes', 'subprocess', 'keyring'})
FORBIDDEN_TEXT = (
    'credential',
    'oauth',
    'keychain',
    'auth_token',
    'access_token',
    'refresh_token',
    'session_token',
    '.claude',
    'claude.json',
    'api_key',
)


def test_the_source_never_reaches_for_a_credential():
    """No code in the Claude Code modules reads the environment or a file, or names a credential.

    Docstrings may describe the guarantee; only names and non-docstring strings count. The names of the environment
    variables Claude Code gets as empty strings are allowed, since blanking them is the point.
    """
    blanked = {name for name, value in _claude_code_sessions._ENV.items() if value == ''}  # pyright: ignore[reportPrivateUsage]
    modules = [
        _claude_code_sessions,
        _claude_code_mcp,
        pydantic_ai.providers.claude_code,
        pydantic_ai.models.claude_code,
        pydantic_ai.profiles.claude_code,
    ]
    found: list[str] = []
    for module in modules:
        tree = ast.parse(inspect.getsource(module))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Name | ast.Attribute | ast.alias):
                name = (
                    node.id
                    if isinstance(node, ast.Name)
                    else node.attr
                    if isinstance(node, ast.Attribute)
                    else node.name
                )
                # `ClaudeCodeSessions.open` opens a session, not a file.
                sessions_open = isinstance(node, ast.Attribute) and ast.unparse(node) == 'self.sessions.open'
                if name.split('.')[-1] in FORBIDDEN_NAMES and not sessions_open:
                    found.append(f'{module.__name__}:{node.lineno} name {name!r}')  # pragma: no cover (a violation)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                if node.value not in blanked and any(text in node.value.lower() for text in FORBIDDEN_TEXT):
                    found.append(
                        f'{module.__name__}:{node.lineno} string {node.value!r}'
                    )  # pragma: no cover (a violation)
    assert found == []
