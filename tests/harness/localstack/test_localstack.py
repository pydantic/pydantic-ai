"""Tests for the LocalStack capability and LocalStackToolset.

Tests drive public behavior: AWS commands through `aws_cli`, the health check
through `localstack_health`, and container management through the toolset's
async-context lifecycle (observing the `docker` commands it runs).
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest
import sniffio

import pydantic_ai_harness
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import ModelRetry, ToolFailed, UserError
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    CommandResult,
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceCommand,
    WorkspaceError,
)
from pydantic_ai_harness.localstack import LocalStack, LocalStack as Exported, LocalStackError, LocalStackToolset

from ._http_server import HttpResponse, http_server, unused_tcp_port


def _toolset(
    *,
    allowed_services: list[str] | None = None,
    denied_services: list[str] | None = None,
    default_timeout: float = 10.0,
    max_output_chars: int = 50_000,
    aws_cli_path: str = 'aws',
    endpoint_url: str = 'http://localhost:4566',
    manage_container: bool = False,
    docker_path: str = 'docker',
    startup_timeout: float = 120.0,
) -> LocalStackToolset[None]:
    return LocalStackToolset[None](
        endpoint_url=endpoint_url,
        region='us-east-1',
        access_key_id='test',
        secret_access_key='test',
        allowed_services=allowed_services or [],
        denied_services=denied_services or [],
        default_timeout=default_timeout,
        max_output_chars=max_output_chars,
        aws_cli_path=aws_cli_path,
        manage_container=manage_container,
        docker_path=docker_path,
        startup_timeout=startup_timeout,
    )


def _run_context(workspace: Workspace | Path | None = None) -> RunContext[None]:
    """A run context for calling tools directly, in a workspace or a local one at a directory."""
    if isinstance(workspace, Path):
        workspace = Workspace(LocalWorkspaceBackend(workspace))
    if workspace is None:
        return RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
    return RunContext[None](
        deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0, workspace=workspace
    )


class _FailingWorkspace(LocalWorkspaceBackend):
    """A workspace whose commands fail with a plain `WorkspaceError`."""

    async def run(self, command: WorkspaceCommand, **kwargs: object) -> CommandResult:
        raise WorkspaceError('sandbox refused the command')


async def _call_localstack_tool(
    toolset: LocalStackToolset[None], name: str, *, workspace_dir: Path | None = None, **tool_args: Any
) -> str:
    ctx = _run_context(workspace_dir)
    tools = await toolset.get_tools(ctx)
    result = await toolset.call_tool(name, tool_args, ctx, tools[name])
    assert isinstance(result, str)
    return result


def _make_stub(tmp_path: Path, body: str) -> str:
    """Write an executable shell script standing in for the AWS CLI."""
    stub = tmp_path / 'fake-aws'
    stub.write_text(f'#!/bin/sh\n{body}\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(stub)


def _docker_stub(tmp_path: Path) -> tuple[str, Path]:
    """A fake `docker` CLI that logs every invocation; `run` also prints a container id."""
    log = tmp_path / 'docker.log'
    stub = tmp_path / 'docker'
    stub.write_text(f'#!/bin/sh\necho "$@" >> {log}\nif [ "$1" = run ]; then echo managed-xyz; fi\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(stub), log


class TestConstruction:
    def test_allow_and_deny_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match=r'Specify allowed_services or denied_services, not both.'):
            _toolset(allowed_services=['s3'], denied_services=['dynamodb'])

    @pytest.mark.parametrize('default_timeout', [0.0, -1.0, float('inf'), float('nan')])
    def test_non_positive_or_non_finite_default_timeout_rejected(self, default_timeout: float) -> None:
        with pytest.raises(ValueError, match=r'default_timeout must be a positive number of seconds.'):
            _toolset(default_timeout=default_timeout)

    def test_non_positive_max_output_chars_rejected(self) -> None:
        with pytest.raises(ValueError, match=r'max_output_chars must be a positive integer.'):
            _toolset(max_output_chars=0)


class TestOutputCap:
    async def test_new_string_tool_is_capped_at_dispatch(self) -> None:
        toolset = _toolset(max_output_chars=1)

        def text() -> str:
            return 'xx'

        toolset.add_function(text)
        assert await _call_localstack_tool(toolset, 'text') == 'x'

    async def test_non_string_tool_result_is_unchanged(self) -> None:
        toolset = _toolset(max_output_chars=1)

        def number() -> int:
            return 42

        toolset.add_function(number)
        ctx = _run_context()
        tools = await toolset.get_tools(ctx)
        assert await toolset.call_tool('number', {}, ctx, tools['number']) == 42


class TestAwsCli:
    async def test_injects_endpoint_region_and_command(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        result = await _toolset(aws_cli_path=stub).aws_cli(_run_context(tmp_path), 's3 ls')
        assert '--endpoint-url http://localhost:4566' in result
        assert '--region us-east-1 s3 ls' in result
        assert '[stdout]' in result

    async def test_strips_redundant_leading_aws(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        result = await _toolset(aws_cli_path=stub).aws_cli(_run_context(tmp_path), 'aws s3 ls')
        assert '--region us-east-1 s3 ls' in result

    async def test_injects_credentials_into_environment(self, tmp_path: Path) -> None:
        stub = _make_stub(
            tmp_path,
            'printf "%s\\n" "$AWS_ENDPOINT_URL|$AWS_DEFAULT_REGION|$AWS_REGION|$AWS_ACCESS_KEY_ID|$AWS_SECRET_ACCESS_KEY"',
        )
        result = await _toolset(aws_cli_path=stub).aws_cli(_run_context(tmp_path), 's3 ls')
        assert 'http://localhost:4566|us-east-1|us-east-1|test|test' in result

    async def test_empty_command_retries(self, tmp_path: Path) -> None:
        with pytest.raises(ModelRetry, match='Provide an AWS CLI command'):
            await _toolset().aws_cli(_run_context(tmp_path), '   ')

    async def test_only_aws_retries(self, tmp_path: Path) -> None:
        with pytest.raises(ModelRetry, match='Provide an AWS CLI command'):
            await _toolset().aws_cli(_run_context(tmp_path), 'aws')

    async def test_unparsable_command_retries(self, tmp_path: Path) -> None:
        with pytest.raises(ModelRetry, match='Could not parse the AWS CLI command'):
            await _toolset().aws_cli(_run_context(tmp_path), "s3 cp 'unterminated")

    async def test_no_service_token_retries(self, tmp_path: Path) -> None:
        with pytest.raises(ModelRetry, match='Could not determine the AWS service'):
            await _toolset().aws_cli(_run_context(tmp_path), '--version')

    async def test_denied_service_retries(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match="'s3' is denied"):
            await _toolset(denied_services=['s3'], aws_cli_path=stub).aws_cli(_run_context(tmp_path), 's3 ls')

    async def test_denylist_allows_other_services(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo ok')
        result = await _toolset(denied_services=['s3'], aws_cli_path=stub).aws_cli(
            _run_context(tmp_path), 'dynamodb list-tables'
        )
        assert 'ok' in result

    async def test_service_outside_allowlist_retries(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match="'iam' is not in the allowed list"):
            await _toolset(allowed_services=['s3'], aws_cli_path=stub).aws_cli(_run_context(tmp_path), 'iam list-users')

    async def test_global_option_value_does_not_bypass_allowlist(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match="'iam' is not in the allowed list"):
            await _toolset(allowed_services=['s3'], aws_cli_path=stub).aws_cli(
                _run_context(tmp_path), '--output s3 iam list-users'
            )

    async def test_global_option_value_does_not_bypass_denylist(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match="'s3' is denied"):
            await _toolset(denied_services=['s3'], aws_cli_path=stub).aws_cli(
                _run_context(tmp_path), '--output prod s3 ls'
            )

    async def test_end_of_options_separator_does_not_bypass_denylist(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match="'s3' is denied"):
            await _toolset(denied_services=['s3'], aws_cli_path=stub).aws_cli(_run_context(tmp_path), '-- s3 ls')

    async def test_trailing_end_of_options_separator_resolves_no_service(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match='Could not determine the AWS service'):
            await _toolset(aws_cli_path=stub).aws_cli(_run_context(tmp_path), '--')

    async def test_allowed_service_permitted(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo permitted')
        result = await _toolset(allowed_services=['s3'], aws_cli_path=stub).aws_cli(_run_context(tmp_path), 's3 ls')
        assert 'permitted' in result

    @pytest.mark.parametrize(
        'command',
        [
            's3 ls --endpoint-url https://s3.amazonaws.com',
            's3 ls --endpoint-url=https://s3.amazonaws.com',
            's3 ls --profile prod',
            's3 ls --region us-west-2',
            's3 ls --no-sign-request',
        ],
    )
    async def test_rejects_model_supplied_global_endpoint_or_credential_options(
        self, command: str, tmp_path: Path
    ) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match='Do not pass AWS global options'):
            await _toolset(aws_cli_path=stub).aws_cli(_run_context(tmp_path), command)

    @pytest.mark.parametrize(
        'command',
        [
            's3 ls --endpoint http://evil',
            's3 ls --end http://evil',
            's3 ls --prof prod',
            's3 ls --no-sign',
            's3 ls --reg eu-west-1',
            's3api list-buckets --endpoint=http://evil',
        ],
    )
    async def test_rejects_abbreviated_global_endpoint_or_credential_options(
        self, command: str, tmp_path: Path
    ) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match='Do not pass AWS global options'):
            await _toolset(aws_cli_path=stub).aws_cli(_run_context(tmp_path), command)

    async def test_allows_safe_options_that_are_not_forbidden_prefixes(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        result = await _toolset(aws_cli_path=stub).aws_cli(_run_context(tmp_path), 's3 ls --no-paginate --output json')
        assert '--no-paginate' in result
        assert '--output json' in result

    async def test_scrubs_workspace_aws_environment_before_running_cli(self, tmp_path: Path) -> None:
        workspace_env = {
            'AWS_PROFILE': 'prod-profile',
            'AWS_CONFIG_FILE': '/tmp/real-config',
            'AWS_SHARED_CREDENTIALS_FILE': '/tmp/real-creds',
            'AWS_SESSION_TOKEN': 'real-session-token',
            'AWS_WEB_IDENTITY_TOKEN_FILE': '/tmp/web-token',
            'AWS_ROLE_ARN': 'arn:aws:iam::123:role/prod',
            'AWS_ACCESS_KEY_ID': 'real-key',
            'NOT_AWS': 'kept',
        }
        stub = _make_stub(
            tmp_path,
            'printf "%s\\n" '
            '"${AWS_PROFILE-unset}|${AWS_CONFIG_FILE-unset}|${AWS_SHARED_CREDENTIALS_FILE-unset}|'
            '${AWS_SESSION_TOKEN-unset}|${AWS_WEB_IDENTITY_TOKEN_FILE-unset}|${AWS_ROLE_ARN-unset}|'
            '$AWS_ACCESS_KEY_ID|$AWS_SECRET_ACCESS_KEY|$NOT_AWS"',
        )
        workspace = Workspace(LocalWorkspaceBackend(tmp_path, env=workspace_env))

        result = await _toolset(aws_cli_path=stub).aws_cli(_run_context(workspace), 's3 ls')

        assert 'unset|unset|unset|unset|unset|unset|test|test|kept' in result

    async def test_runs_in_the_workspace_working_directory(self, tmp_path: Path) -> None:
        workspace_dir = tmp_path / 'repo'
        workspace_dir.mkdir()
        stub = _make_stub(tmp_path, 'pwd -P')

        result = await _toolset(aws_cli_path=stub).aws_cli(_run_context(workspace_dir), 's3 ls')

        assert result == f'[stdout]\n{workspace_dir.resolve()}\n'

    async def test_workspace_path_lookup(self, tmp_path: Path) -> None:
        # A bare name resolves on the workspace's PATH, not the agent process's.
        bin_dir = tmp_path / 'bin'
        bin_dir.mkdir()
        (bin_dir / 'fake-aws').write_text('#!/bin/sh\necho from-workspace\n')
        (bin_dir / 'fake-aws').chmod(0o755)
        workspace = Workspace(LocalWorkspaceBackend(tmp_path, env={'PATH': f'{bin_dir}:/usr/bin:/bin'}))

        result = await _toolset(aws_cli_path='fake-aws').aws_cli(_run_context(workspace), 's3 ls')

        assert result == '[stdout]\nfrom-workspace\n'

    @pytest.mark.parametrize('timeout_seconds', [0.0, -1.0, float('inf'), float('nan')])
    async def test_non_positive_or_non_finite_timeout_retries(self, timeout_seconds: float, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo ran')
        with pytest.raises(ModelRetry, match='`timeout_seconds` must be a positive number'):
            await _toolset(aws_cli_path=stub).aws_cli(_run_context(tmp_path), 's3 ls', timeout_seconds=timeout_seconds)

    async def test_workspace_failure_fails_the_call(self, tmp_path: Path) -> None:
        with pytest.raises(ToolFailed, match='sandbox refused'):
            await _toolset().aws_cli(_run_context(Workspace(_FailingWorkspace(tmp_path))), 's3 ls')

    async def test_no_aws_cli_tool_without_command_support(self, tmp_path: Path) -> None:
        read_only = _run_context(ReadOnlyWorkspace(Workspace(LocalWorkspaceBackend(tmp_path))))
        assert list(await _toolset().get_tools(read_only)) == ['localstack_health']
        assert list(await _toolset().get_tools(_run_context(tmp_path))) == ['aws_cli', 'localstack_health']

    async def test_nonzero_exit_shows_code_and_stderr(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo boom >&2\nexit 3')
        result = await _toolset(aws_cli_path=stub).aws_cli(_run_context(tmp_path), 's3 ls')
        assert '[stderr]\nboom' in result
        assert '[exit code: 3]' in result

    async def test_no_output(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'exit 0')
        assert await _toolset(aws_cli_path=stub).aws_cli(_run_context(tmp_path), 's3 ls') == '(no output)'

    async def test_missing_cli_binary(self, tmp_path: Path) -> None:
        result = await _toolset(aws_cli_path='/no/such/aws-binary').aws_cli(_run_context(tmp_path), 's3 ls')
        assert result == (
            "[error: AWS CLI '/no/such/aws-binary' not found in the workspace. "
            'Install the AWS CLI there to use LocalStack tools.]'
        )

    async def test_timeout(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'sleep 5')
        result = await _toolset(aws_cli_path=stub).aws_cli(_run_context(tmp_path), 's3 ls', timeout_seconds=0.3)
        assert 'timed out after 0.3s' in result

    async def test_per_call_timeout_overrides_default(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        result = await _toolset(default_timeout=0.01, aws_cli_path=stub).aws_cli(
            _run_context(tmp_path), 's3 ls', timeout_seconds=10.0
        )
        assert '[stdout]' in result

    async def test_output_truncated(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'printf "%01000d" 0')
        result = await _call_localstack_tool(
            _toolset(max_output_chars=100, aws_cli_path=stub), 'aws_cli', workspace_dir=tmp_path, command='s3 ls'
        )
        assert 'output truncated' in result
        assert len(result) == 100

    async def test_truncation_respects_small_cap_without_marker(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'printf "%01000d" 0')
        result = await _call_localstack_tool(
            _toolset(max_output_chars=10, aws_cli_path=stub), 'aws_cli', workspace_dir=tmp_path, command='s3 ls'
        )
        assert len(result) <= 10
        assert 'output truncated' not in result

    async def test_truncation_keeps_tail_and_marker_for_normal_cap(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'printf "HEAD"; printf "%0500dTAIL" 0')
        result = await _call_localstack_tool(
            _toolset(max_output_chars=200, aws_cli_path=stub), 'aws_cli', workspace_dir=tmp_path, command='s3 ls'
        )
        assert len(result) == 200
        assert result.startswith('[... output truncated, showing last 153 chars]\n')
        assert result.endswith('TAIL')
        assert 'HEAD' not in result

    async def test_truncation_caps_complete_failure_response(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'printf "%0500d" 0; exit 7')
        result = await _call_localstack_tool(
            _toolset(max_output_chars=200, aws_cli_path=stub), 'aws_cli', workspace_dir=tmp_path, command='s3 ls'
        )
        assert len(result) == 200
        assert result.startswith('[... output truncated, showing last 153 chars]\n')
        assert result.endswith('[exit code: 7]')


class TestLocalStackHealth:
    async def test_success(self) -> None:
        with http_server([HttpResponse(200, '{"services": {"s3": "available"}}')]) as server:
            result = await _toolset(endpoint_url=server.endpoint_url).localstack_health()
        assert server.paths == ['/_localstack/health']
        assert '"s3": "available"' in result

    async def test_trailing_slash_endpoint(self) -> None:
        with http_server([HttpResponse(200)]) as server:
            await _toolset(endpoint_url=f'{server.endpoint_url}/').localstack_health()
        assert server.paths == ['/_localstack/health']

    async def test_non_200(self) -> None:
        with http_server([HttpResponse(503)]) as server:
            result = await _toolset(endpoint_url=server.endpoint_url).localstack_health()
        assert 'HTTP 503' in result

    async def test_at_size_limit_is_not_truncated(self) -> None:
        with http_server([HttpResponse(200, 'x' * 100)]) as server:
            result = await _call_localstack_tool(
                _toolset(endpoint_url=server.endpoint_url, max_output_chars=100), 'localstack_health'
            )
        assert result == 'x' * 100

    async def test_over_size_limit_keeps_tail(self) -> None:
        with http_server([HttpResponse(200, 'HEAD' + 'T' * 100)]) as server:
            result = await _call_localstack_tool(
                _toolset(endpoint_url=server.endpoint_url, max_output_chars=100), 'localstack_health'
            )
        assert len(result) <= 100
        assert result.endswith('T')
        assert 'HEAD' not in result
        assert 'output truncated' in result

    async def test_connection_error(self) -> None:
        result = await _toolset(endpoint_url=f'http://localhost:{unused_tcp_port()}').localstack_health()
        assert 'could not reach LocalStack' in result


class TestLocalStackCapability:
    def test_default_construction(self) -> None:
        cap = LocalStack()
        assert cap.endpoint_url == 'http://localhost.localstack.cloud:4566'
        assert cap.region == 'us-east-1'
        assert cap.access_key_id == 'test'
        assert cap.default_timeout == 60.0
        assert cap.aws_cli_path == 'aws'

    def test_custom_construction(self) -> None:
        cap = LocalStack(endpoint_url='http://ls:4566', allowed_services=['s3'], default_timeout=5.0)
        assert cap.endpoint_url == 'http://ls:4566'
        assert cap.allowed_services == ['s3']
        assert cap.default_timeout == 5.0

    def test_get_toolset_returns_toolset(self) -> None:
        assert isinstance(LocalStack().get_toolset(), LocalStackToolset)

    def test_instructions_included_by_default(self) -> None:
        instructions = LocalStack(endpoint_url='http://ls:4566').get_instructions()
        assert instructions is not None
        assert 'http://ls:4566' in instructions
        assert 'aws_cli' in instructions

    def test_instructions_can_be_disabled(self) -> None:
        assert LocalStack(include_instructions=False).get_instructions() is None

    async def test_agent_integration(self, tmp_path: Path) -> None:
        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        stub = _make_stub(tmp_path, 'echo "ran in $(pwd -P)"')
        model = TestModel(custom_output_text='done', call_tools=['aws_cli'])
        agent: Agent[None, str] = Agent(model, capabilities=[LocalWorkspace(tmp_path), LocalStack(aws_cli_path=stub)])
        result = await agent.run('create a bucket')
        assert result.output == 'done'
        returns = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'aws_cli'
        ]
        assert returns == [f'[stdout]\nran in {tmp_path.resolve()}\n']

    async def test_no_workspace_fails_the_run(self) -> None:
        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        agent: Agent[None, str] = Agent(TestModel(), capabilities=[LocalStack()])
        with pytest.raises(UserError, match='`LocalStack` needs a workspace'):
            await agent.run('create a bucket')

    def test_exported_from_submodule(self) -> None:

        assert Exported is LocalStack
        assert pydantic_ai_harness.LocalStack is LocalStack


class TestContainerManagement:
    def test_default_is_unmanaged(self) -> None:
        cap = LocalStack()
        assert cap.manage_container is False
        assert cap.image == 'localstack/localstack'
        assert cap.host_address == '127.0.0.1'
        assert cap.service_port_range is None
        assert cap.mount_docker_socket is False
        assert cap.docker_path == 'docker'
        assert cap.startup_timeout == 120.0
        assert dict(cap.container_env) == {}

    def test_custom_container_config(self) -> None:
        cap = LocalStack(manage_container=True, container_name='ls', container_env={'DEBUG': '1'})
        assert cap.manage_container is True
        assert cap.container_name == 'ls'
        assert dict(cap.container_env) == {'DEBUG': '1'}

    async def test_unmanaged_does_not_touch_docker(self, tmp_path: Path) -> None:
        docker, log = _docker_stub(tmp_path)
        async with _toolset(docker_path=docker):
            pass
        assert not log.exists()

    async def test_managed_starts_on_endpoint_port_and_stops(self, tmp_path: Path) -> None:
        docker, log = _docker_stub(tmp_path)
        with http_server([HttpResponse(200)]) as health:
            async with _toolset(endpoint_url=health.endpoint_url, manage_container=True, docker_path=docker):
                pass
        log_text = log.read_text()
        assert (f'run -d --rm -p 127.0.0.1:{health.port}:4566 -e GATEWAY_LISTEN localstack/localstack') in log_text
        assert 'stop managed-xyz' in log_text

    async def test_managed_defaults_to_edge_port_when_endpoint_has_no_port(self, tmp_path: Path) -> None:
        docker, log = _docker_stub(tmp_path)
        with pytest.raises(LocalStackError, match=r'did not become ready within 1.0s'):
            async with _toolset(
                endpoint_url='http://localhost',
                manage_container=True,
                docker_path=docker,
                startup_timeout=1.0,
            ):
                pass  # pragma: no cover
        log_text = log.read_text()
        assert '-p 127.0.0.1:4566:4566' in log_text
        assert 'localstack/localstack' in log_text
        assert 'stop managed-xyz' in log_text

    async def test_for_run_returns_fresh_managed_instance(self, test_model: TestModel, tmp_path: Path) -> None:
        docker, log = _docker_stub(tmp_path)
        with http_server([HttpResponse(200)]) as health:
            original = _toolset(endpoint_url=health.endpoint_url, manage_container=True, docker_path=docker)
            ctx = RunContext[None](deps=None, model=test_model, usage=RunUsage(), prompt=None, messages=[], run_step=0)
            fresh = await original.for_run(ctx)
            assert isinstance(fresh, LocalStackToolset)
            assert fresh is not original
            async with fresh:
                pass
        assert 'stop managed-xyz' in log.read_text()

    async def test_agent_integration_manages_container(self, tmp_path: Path) -> None:

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        docker, log = _docker_stub(tmp_path)
        with http_server([HttpResponse(200)]) as health:
            cap = LocalStack(endpoint_url=health.endpoint_url, manage_container=True, docker_path=docker)
            agent: Agent[None, str] = Agent(
                model=TestModel(custom_output_text='done', call_tools=[]),
                capabilities=[LocalWorkspace(tmp_path), cap],
            )
            result = await agent.run('provision a bucket')
        assert result.output == 'done'
        assert 'stop managed-xyz' in log.read_text()
