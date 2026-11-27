"""Tests for `parse_permission_rule` and `parse_permission_rules`.

Parsing makes no provider request, and the agent runs use `FunctionModel`, so there is nothing to record.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.permissions import (
    PermissionRule,
    Permissions,
    PermissionVerdict,
    parse_permission_rule,
    parse_permission_rules,
)
from pydantic_ai_harness.permissions._rules import decide
from pydantic_ai_harness.shell import Shell


def _calls(*calls: tuple[str, Mapping[str, object]]) -> FunctionModel:
    """A model that makes `calls` one per step, then answers with the last tool result it got."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step < len(calls):
            name, args = calls[step]
            return ModelResponse(parts=[ToolCallPart(name, dict(args), tool_call_id=f'call{step}')])
        last = messages[-1]
        assert isinstance(last, ModelRequest)
        returned = [part for part in last.parts if isinstance(part, ToolReturnPart)]
        return ModelResponse(parts=[TextPart(str(returned[0].content) if returned else 'done')])

    return FunctionModel(respond)


def _agent(
    tmp_path: Path, rules: Sequence[PermissionRule[None]], *calls: tuple[str, Mapping[str, object]]
) -> Agent[None, str | DeferredToolRequests]:
    return Agent[None, 'str | DeferredToolRequests'](
        _calls(*calls),
        deps_type=type(None),
        output_type=[str, DeferredToolRequests],
        capabilities=[
            LocalWorkspace(tmp_path),
            FileSystem(),
            Shell(denied_commands=[]),
            Permissions('read-only', rules=rules),
        ],
    )


class TestParsePermissionRule:
    @pytest.mark.parametrize(
        ('rule', 'verdict', 'expected'),
        [
            ('Bash(git push:*)', 'allow', PermissionRule('allow', command='git push')),
            ('Shell(git push:*)', 'deny', PermissionRule('deny', command='git push')),
            ('  Bash( git push :* )  ', 'allow', PermissionRule('allow', command='git push')),
            ('Bash(git push)', 'ask', PermissionRule('ask', command='git push')),
            ('Bash(rm -rf)', 'deny', PermissionRule('deny', command='rm -rf')),
            ('Bash(echo (x):*)', 'allow', PermissionRule('allow', command='echo (x)')),
            ('Bash', 'ask', PermissionRule('ask', tools={'code_arg_language': 'shell'})),
            ('write_file', 'deny', PermissionRule('deny', tools=['write_file'])),
            ('write_file(docs/**)', 'allow', PermissionRule('allow', tools=['write_file'], args={'path': 'docs/**'})),
            ('write_file(a=b/c)', 'deny', PermissionRule('deny', tools=['write_file'], args={'a': 'b/c'})),
            ('write_file(docs/a=b)', 'deny', PermissionRule('deny', tools=['write_file'], args={'path': 'docs/a=b'})),
            (
                'web_fetch(url=https://example.com/*)',
                'ask',
                PermissionRule('ask', tools=['web_fetch'], args={'url': 'https://example.com/*'}),
            ),
            ('mcp__github__create_issue', 'ask', PermissionRule('ask', tools=['mcp__github__create_issue'])),
            ('read', 'deny', PermissionRule('deny', tools=['read'])),
        ],
    )
    def test_compiles_to_a_rule(self, rule: str, verdict: PermissionVerdict, expected: PermissionRule[None]) -> None:
        assert parse_permission_rule(rule, verdict) == expected

    def test_keeps_the_reason(self) -> None:
        assert parse_permission_rule('Bash(rm -rf)', 'deny', reason='No recursive deletes.') == PermissionRule(
            'deny', command='rm -rf', reason='No recursive deletes.'
        )

    def test_an_exact_command_cannot_be_allowed(self) -> None:
        with pytest.raises(UserError, match=r'write `Bash\(git status:\*\)` to allow them'):
            parse_permission_rule('Bash(git status)', 'allow')

    @pytest.mark.parametrize(
        ('name', 'instead'),
        [
            ('Read', 'read_file'),
            ('Edit', 'edit_file'),
            ('MultiEdit', 'edit_file'),
            ('Write', 'write_file'),
            ('Glob', 'find_files'),
            ('Grep', 'grep'),
            ('WebFetch', 'web_fetch'),
        ],
    )
    def test_claude_code_tool_names_name_the_tool_to_use(self, name: str, instead: str) -> None:
        with pytest.raises(UserError, match=rf"Claude Code tool name '{name}': name `{instead}` instead"):
            parse_permission_rule(f'{name}(.env)', 'deny')
        with pytest.raises(UserError, match=rf"Claude Code tool name '{name}'"):
            parse_permission_rule(name, 'deny')

    @pytest.mark.parametrize(
        'rule', ['', '   ', 'write_file()', 'Bash(  )', 'write file', 'write_file(docs', '(x)', 'write_file)']
    )
    def test_malformed_strings_raise(self, rule: str) -> None:
        with pytest.raises(UserError, match=re.escape(f'Permission rule {rule!r} must be a tool name')):
            parse_permission_rule(rule, 'deny')

    @pytest.mark.parametrize('rule', ['shell(rm -rf:*)', 'run_command(git push:*)', 'bash(rm:*)'])
    def test_a_command_pattern_on_another_name_raises(self, rule: str) -> None:
        with pytest.raises(UserError, match=re.escape(f'Permission rule {rule!r} uses `:*`')):
            parse_permission_rule(rule, 'deny')

    def test_text_before_an_equals_sign_that_is_no_argument_name_is_part_of_the_path(self) -> None:
        assert parse_permission_rule('write_file(9=x)', 'deny') == PermissionRule(
            'deny', tools=['write_file'], args={'path': '9=x'}
        )

    @pytest.mark.parametrize('verdict', ['allow', 'deny'])
    def test_a_trailing_space_star_is_the_same_as_colon_star(self, verdict: PermissionVerdict) -> None:
        assert parse_permission_rule('Bash(git push *)', verdict) == parse_permission_rule('Bash(git push:*)', verdict)
        assert parse_permission_rule('Bash(*)', verdict) == parse_permission_rule('Bash', verdict)

    def test_a_star_inside_a_command_raises(self) -> None:
        with pytest.raises(UserError, match='has a `\\*` that is not at its end'):
            parse_permission_rule('Bash(git * main)', 'deny')

    @pytest.mark.parametrize('rule', ['read_file(./.env)', 'read_file(././.env)', 'read_file(path=./.env)'])
    def test_a_leading_dot_slash_is_dropped(self, rule: str) -> None:
        assert parse_permission_rule(rule, 'deny') == PermissionRule('deny', tools=['read_file'], args={'path': '.env'})

    def test_other_arguments_keep_their_glob_as_written(self) -> None:
        assert parse_permission_rule('web_fetch(url=./x)', 'deny').args == {'url': './x'}

    @pytest.mark.parametrize('rule', ['read_file(/secrets/**)', 'read_file(~/.ssh/*)', 'read_file(path=/etc/*)'])
    def test_an_absolute_or_home_path_raises(self, rule: str) -> None:
        with pytest.raises(UserError, match='starts its path with'):
            parse_permission_rule(rule, 'deny')

    @pytest.mark.parametrize('rule', ['mcp__github__*', 'mcp__github__create_?', 'tool[s]', 'mcp__*(x)'])
    def test_a_wildcard_in_a_tool_name_raises(self, rule: str) -> None:
        with pytest.raises(UserError, match='has a wildcard in its tool name'):
            parse_permission_rule(rule, 'deny')

    @pytest.mark.parametrize('rule', ['mcp__github', 'mcp__github(x)'])
    def test_a_claude_code_mcp_server_rule_raises(self, rule: str) -> None:
        with pytest.raises(UserError, match='names an MCP server the way Claude Code does'):
            parse_permission_rule(rule, 'deny')

    def test_a_full_mcp_tool_name_stays_literal(self) -> None:
        assert parse_permission_rule('mcp__github__create_issue', 'deny') == PermissionRule(
            'deny', tools=['mcp__github__create_issue']
        )

    @pytest.mark.parametrize(
        ('rule', 'glob'), [('read_file(**/.env)', '.env'), ('read_file(**/**/secrets/*)', 'secrets/*')]
    )
    def test_a_leading_double_star_is_dropped_for_ask_and_deny(self, rule: str, glob: str) -> None:
        assert parse_permission_rule(rule, 'deny').args == {'path': glob}
        assert parse_permission_rule(rule, 'ask').args == {'path': glob}

    def test_an_allow_keeps_its_leading_double_star(self) -> None:
        """Kept, it allows less, not more: a call at the top isn't allowed, so it asks."""
        assert parse_permission_rule('read_file(**/.env)', 'allow').args == {'path': '**/.env'}

    def test_an_unknown_verdict_raises(self) -> None:
        with pytest.raises(UserError, match="has the verdict 'denny'"):
            parse_permission_rule('read_file', 'denny')  # pyright: ignore[reportArgumentType]

    def test_an_empty_command_raises(self) -> None:
        with pytest.raises(UserError, match='must be a shell command'):
            parse_permission_rule('Bash(:*)', 'allow')


class TestParsePermissionRules:
    def test_parses_each_verdict(self) -> None:
        assert parse_permission_rules(
            {
                'allow': ['Bash(git status:*)', 'write_file(docs/**)'],
                'ask': ['Bash(git push)'],
                'deny': ['Bash(rm -rf)'],
            }
        ) == [
            PermissionRule('allow', command='git status'),
            PermissionRule('allow', tools=['write_file'], args={'path': 'docs/**'}),
            PermissionRule('ask', command='git push'),
            PermissionRule('deny', command='rm -rf'),
        ]

    def test_an_unknown_key_raises(self) -> None:
        with pytest.raises(UserError, match="has the verdict 'denny'"):
            parse_permission_rules({'denny': ['read_file']})  # pyright: ignore[reportArgumentType]

    def test_a_single_string_raises(self) -> None:
        with pytest.raises(UserError, match="for 'deny' must be a list of strings"):
            parse_permission_rules({'deny': 'read_file'})

    def test_a_recursive_deny_refuses_a_file_at_the_top(self) -> None:
        [rule] = parse_permission_rules({'deny': ['read_file(**/.env)']})
        for path in ('.env', 'app/.env', '/home/me/project/.env'):
            assert decide([rule], command=None, command_arg=None, args={'path': path}, posix=True) == rule

    async def test_rules_decide_real_shell_and_file_calls(self, tmp_path: Path) -> None:
        (tmp_path / 'docs').mkdir()
        (tmp_path / 'kept').mkdir()
        rules = parse_permission_rules(
            {'allow': ['Bash(echo:*)', 'write_file(docs/**)'], 'deny': ['Bash(rm -rf)', 'write_file(secrets/*)']}
        )

        ran = await _agent(tmp_path, rules, ('run_command', {'command': 'echo hi'})).run('go')
        assert isinstance(ran.output, str) and 'hi' in ran.output

        denied = await _agent(tmp_path, rules, ('run_command', {'command': 'rm -rf kept'})).run('go')
        assert denied.output == 'A permission rule denied this command, so it did not run. Do not retry it.'
        assert (tmp_path / 'kept').is_dir()

        await _agent(tmp_path, rules, ('write_file', {'path': 'docs/a.md', 'content': 'x'})).run('go')
        assert (tmp_path / 'docs' / 'a.md').read_text() == 'x'

        refused = await _agent(tmp_path, rules, ('write_file', {'path': 'secrets/key', 'content': 'x'})).run('go')
        assert refused.output == 'A permission rule denied this file write, so it did not run. Do not retry it.'

        held = await _agent(tmp_path, rules, ('write_file', {'path': 'notes.md', 'content': 'x'})).run('go')
        assert isinstance(held.output, DeferredToolRequests)
