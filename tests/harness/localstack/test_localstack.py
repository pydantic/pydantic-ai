"""Tests for the LocalStack capability and LocalStackToolset.

Tests drive public behavior: AWS commands through `aws_cli`, the health check
through `localstack_health`, and container management through the toolset's
async-context lifecycle (observing the `docker` commands it runs).
"""

from __future__ import annotations

import math
import stat
from pathlib import Path
from typing import Any

import pytest
import sniffio
from pydantic import ValidationError

import pydantic_ai_harness
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.localstack import LocalStack, LocalStack as Exported, LocalStackError, LocalStackToolset
from tests.harness._recording_durability import RecordingDurability

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
    host_address: str = '127.0.0.1',
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
        host_address=host_address,
        docker_path=docker_path,
        startup_timeout=startup_timeout,
    )


def _run_context() -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


async def _call_localstack_tool(toolset: LocalStackToolset[None], name: str, **tool_args: Any) -> str:
    ctx = _run_context()
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
        with pytest.raises(ValueError, match=r'Specify `allowed_services` or `denied_services`, not both.'):
            _toolset(allowed_services=['s3'], denied_services=['dynamodb'])

    def test_non_positive_max_output_chars_rejected(self) -> None:
        with pytest.raises(ValueError, match=r'`max_output_chars` must be a positive integer.'):
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
        result = await _toolset(aws_cli_path=stub).aws_cli('s3 ls')
        assert '--endpoint-url http://localhost:4566' in result
        assert '--region us-east-1 s3 ls' in result
        assert '[stdout]' in result

    async def test_strips_redundant_leading_aws(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        result = await _toolset(aws_cli_path=stub).aws_cli('aws s3 ls')
        assert '--region us-east-1 s3 ls' in result

    async def test_injects_credentials_into_environment(self, tmp_path: Path) -> None:
        stub = _make_stub(
            tmp_path,
            'printf "%s\\n" "$AWS_ENDPOINT_URL|$AWS_DEFAULT_REGION|$AWS_REGION|$AWS_ACCESS_KEY_ID|$AWS_SECRET_ACCESS_KEY"',
        )
        result = await _toolset(aws_cli_path=stub).aws_cli('s3 ls')
        assert 'http://localhost:4566|us-east-1|us-east-1|test|test' in result

    async def test_empty_command_retries(self) -> None:
        with pytest.raises(ModelRetry, match='Provide an AWS CLI command'):
            await _toolset().aws_cli('   ')

    async def test_only_aws_retries(self) -> None:
        with pytest.raises(ModelRetry, match='Provide an AWS CLI command'):
            await _toolset().aws_cli('aws')

    async def test_unparsable_command_retries(self) -> None:
        with pytest.raises(ModelRetry, match='Could not parse the AWS CLI command'):
            await _toolset().aws_cli("s3 cp 'unterminated")

    async def test_no_service_token_retries(self) -> None:
        with pytest.raises(ModelRetry, match='Could not determine the AWS service'):
            await _toolset().aws_cli('--version')

    async def test_denied_service_retries(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match="'s3' is denied"):
            await _toolset(denied_services=['s3'], aws_cli_path=stub).aws_cli('s3 ls')

    async def test_denylist_allows_other_services(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo ok')
        result = await _toolset(denied_services=['s3'], aws_cli_path=stub).aws_cli('dynamodb list-tables')
        assert 'ok' in result

    async def test_service_outside_allowlist_retries(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match="'iam' is not in the allowed list"):
            await _toolset(allowed_services=['s3'], aws_cli_path=stub).aws_cli('iam list-users')

    async def test_global_option_value_does_not_bypass_allowlist(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match="'iam' is not in the allowed list"):
            await _toolset(allowed_services=['s3'], aws_cli_path=stub).aws_cli('--output s3 iam list-users')

    async def test_global_option_value_does_not_bypass_denylist(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match="'s3' is denied"):
            await _toolset(denied_services=['s3'], aws_cli_path=stub).aws_cli('--output prod s3 ls')

    async def test_end_of_options_separator_does_not_bypass_denylist(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match="'s3' is denied"):
            await _toolset(denied_services=['s3'], aws_cli_path=stub).aws_cli('-- s3 ls')

    async def test_trailing_end_of_options_separator_resolves_no_service(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match='Could not determine the AWS service'):
            await _toolset(aws_cli_path=stub).aws_cli('--')

    @pytest.mark.parametrize(
        ('command', 'allowed_services', 'denied_services', 'message'),
        [
            ('--out text s3 ls', [], ['s3'], "'s3' is denied"),
            ('--quer s3 iam list-users', ['s3'], [], "'iam' is not in the allowed list"),
            ('--co on s3 ls', [], ['s3'], "'s3' is denied"),
            ('--deb iam list-users', ['s3'], [], "'iam' is not in the allowed list"),
        ],
    )
    async def test_abbreviated_global_option_does_not_bypass_service_lists(
        self,
        tmp_path: Path,
        command: str,
        allowed_services: list[str],
        denied_services: list[str],
        message: str,
    ) -> None:
        """The CLI accepts a unique prefix of a global option, so the service parser must skip its value too."""
        stub = _make_stub(tmp_path, 'echo "$@"')
        toolset = _toolset(allowed_services=allowed_services, denied_services=denied_services, aws_cli_path=stub)
        with pytest.raises(ModelRetry, match=message):
            await toolset.aws_cli(command)

    async def test_ambiguous_global_option_prefix_takes_no_value(self, tmp_path: Path) -> None:
        """`--cli` starts several global options, so like the CLI it selects none and consumes no value."""
        stub = _make_stub(tmp_path, 'echo "$@"')
        with pytest.raises(ModelRetry, match="'iam' is not in the allowed list"):
            await _toolset(allowed_services=['s3'], aws_cli_path=stub).aws_cli('--cli iam list-users')

    async def test_abbreviated_global_flag_keeps_the_following_service(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        result = await _toolset(allowed_services=['s3'], aws_cli_path=stub).aws_cli('--deb s3 ls')
        assert '--deb s3 ls' in result

    async def test_allowed_service_permitted(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo permitted')
        result = await _toolset(allowed_services=['s3'], aws_cli_path=stub).aws_cli('s3 ls')
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
            await _toolset(aws_cli_path=stub).aws_cli(command)

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
            await _toolset(aws_cli_path=stub).aws_cli(command)

    async def test_allows_safe_options_that_are_not_forbidden_prefixes(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        result = await _toolset(aws_cli_path=stub).aws_cli('s3 ls --no-paginate --output json')
        assert '--no-paginate' in result
        assert '--output json' in result

    async def test_scrubs_aws_environment_before_running_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('AWS_PROFILE', 'prod-profile')
        monkeypatch.setenv('AWS_CONFIG_FILE', '/tmp/real-config')
        monkeypatch.setenv('AWS_SHARED_CREDENTIALS_FILE', '/tmp/real-creds')
        monkeypatch.setenv('AWS_SESSION_TOKEN', 'real-session-token')
        monkeypatch.setenv('AWS_WEB_IDENTITY_TOKEN_FILE', '/tmp/web-token')
        monkeypatch.setenv('AWS_ROLE_ARN', 'arn:aws:iam::123:role/prod')
        stub = _make_stub(
            tmp_path,
            'printf "%s\\n" '
            '"${AWS_PROFILE-unset}|${AWS_CONFIG_FILE-unset}|${AWS_SHARED_CREDENTIALS_FILE-unset}|'
            '${AWS_SESSION_TOKEN-unset}|${AWS_WEB_IDENTITY_TOKEN_FILE-unset}|${AWS_ROLE_ARN-unset}|'
            '$AWS_ACCESS_KEY_ID|$AWS_SECRET_ACCESS_KEY"',
        )

        result = await _toolset(aws_cli_path=stub).aws_cli('s3 ls')

        assert '/tmp/real-config' not in result
        assert '/tmp/real-creds' not in result
        assert '|unset|unset|unset|test|test' in result
        assert result.startswith('[stdout]\nunset|')

    async def test_cli_config_files_are_isolated_from_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`aws configure set` must not reach the `~/.aws` files of the user running the agent."""
        home = tmp_path / 'home'
        monkeypatch.setenv('HOME', str(home))
        stub = _make_stub(
            tmp_path,
            'printf "%s\\n" "$AWS_CONFIG_FILE" "$AWS_SHARED_CREDENTIALS_FILE"\n'
            'echo "[default]" > "$AWS_SHARED_CREDENTIALS_FILE"',
        )

        result = await _toolset(aws_cli_path=stub).aws_cli('configure set aws_access_key_id real')

        config_file, credentials_file = (Path(line) for line in result.removeprefix('[stdout]\n').splitlines())
        assert (config_file.name, credentials_file.name) == ('config', 'credentials')
        assert config_file.parent == credentials_file.parent
        assert home not in credentials_file.parents
        assert not credentials_file.parent.exists()
        assert not home.exists()

    async def test_stdin_is_closed(self, tmp_path: Path) -> None:
        """A command that reads stdin, like `s3 cp - s3://bucket/key`, sees EOF instead of hanging."""
        stub = _make_stub(tmp_path, 'cat')
        assert await _toolset(default_timeout=5.0, aws_cli_path=stub).aws_cli('s3 cp - s3://bucket/key') == (
            '(no output)'
        )

    async def test_nonzero_exit_shows_code_and_stderr(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo boom >&2\nexit 3')
        result = await _toolset(aws_cli_path=stub).aws_cli('s3 ls')
        assert '[stderr]\nboom' in result
        assert '[exit code: 3]' in result

    async def test_no_output(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'exit 0')
        assert await _toolset(aws_cli_path=stub).aws_cli('s3 ls') == '(no output)'

    async def test_missing_cli_binary(self) -> None:
        result = await _toolset(aws_cli_path='/no/such/aws-binary').aws_cli('s3 ls')
        assert 'not found' in result

    async def test_timeout(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'sleep 5')
        result = await _toolset(aws_cli_path=stub).aws_cli('s3 ls', timeout_seconds=0.3)
        assert 'timed out after 0.3s' in result

    @pytest.mark.parametrize('timeout_seconds', [0.0, -5.0, math.inf, math.nan])
    async def test_rejects_timeout_that_is_not_positive_and_finite(self, timeout_seconds: float) -> None:
        """Argument validation turns a bad model-supplied timeout into a retry before the command or a durable operation runs."""
        tools = await _toolset().get_tools(_run_context())
        with pytest.raises(ValidationError, match='timeout_seconds'):
            tools['aws_cli'].args_validator.validate_python({'command': 's3 ls', 'timeout_seconds': timeout_seconds})

    async def test_timeout_constraint_is_in_the_tool_schema(self) -> None:
        tools = await _toolset().get_tools(_run_context())
        assert tools['aws_cli'].tool_def.parameters_json_schema['properties']['timeout_seconds']['anyOf'] == [
            {'exclusiveMinimum': 0, 'type': 'number'},
            {'type': 'null'},
        ]

    async def test_per_call_timeout_overrides_default(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'echo "$@"')
        result = await _toolset(default_timeout=0.01, aws_cli_path=stub).aws_cli('s3 ls', timeout_seconds=10.0)
        assert '[stdout]' in result

    async def test_output_truncated(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'printf "%01000d" 0')
        result = await _call_localstack_tool(
            _toolset(max_output_chars=100, aws_cli_path=stub), 'aws_cli', command='s3 ls'
        )
        assert 'output truncated' in result
        assert len(result) == 100

    async def test_truncation_respects_small_cap_without_marker(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'printf "%01000d" 0')
        result = await _call_localstack_tool(
            _toolset(max_output_chars=10, aws_cli_path=stub), 'aws_cli', command='s3 ls'
        )
        assert len(result) <= 10
        assert 'output truncated' not in result

    async def test_truncation_keeps_tail_and_marker_for_normal_cap(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'printf "HEAD"; printf "%0500dTAIL" 0')
        result = await _call_localstack_tool(
            _toolset(max_output_chars=200, aws_cli_path=stub), 'aws_cli', command='s3 ls'
        )
        assert len(result) == 200
        assert result.startswith('[... output truncated, showing last 153 chars]\n')
        assert result.endswith('TAIL')
        assert 'HEAD' not in result

    async def test_durable_operation_returns_capped_output(self, tmp_path: Path) -> None:
        """The capability's toolset runs `aws_cli` as a durable operation, whose recorded result must be capped."""
        stub = _make_stub(tmp_path, 'printf "HEAD"; printf "%05000dTAIL" 0')
        toolset = LocalStack(aws_cli_path=stub, max_output_chars=100).get_toolset()
        assert isinstance(toolset, LocalStackToolset)
        result = await toolset.aws_cli('s3 ls')
        assert len(result) == 100
        assert result.endswith('TAIL')
        assert 'HEAD' not in result

    async def test_output_cap_drops_a_character_split_by_the_tail_read(self, tmp_path: Path) -> None:
        """Only the bytes the cap can use are read, which can start mid-character; the cap then drops that partial one."""
        stub = _make_stub(tmp_path, 'i=0; while [ $i -lt 500 ]; do printf "\\303\\251"; i=$((i+1)); done; printf x')
        result = await _toolset(max_output_chars=100, aws_cli_path=stub).aws_cli('s3 ls')
        assert len(result) == 100
        assert result.endswith('\u00e9' * 10 + 'x')
        assert '\ufffd' not in result

    async def test_truncation_caps_complete_failure_response(self, tmp_path: Path) -> None:
        stub = _make_stub(tmp_path, 'printf "%0500d" 0; exit 7')
        result = await _call_localstack_tool(
            _toolset(max_output_chars=200, aws_cli_path=stub), 'aws_cli', command='s3 ls'
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

    async def test_through_the_capability(self) -> None:
        with http_server([HttpResponse(200, '{"services": {"s3": "available"}}')]) as server:
            toolset = LocalStack(endpoint_url=server.endpoint_url).get_toolset()
            assert isinstance(toolset, LocalStackToolset)
            result = await toolset.localstack_health()
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

    async def test_connection_error_is_capped_before_dispatch(self) -> None:
        """The durable operation records the method's own result, so the error must fit the cap there too."""
        endpoint_url = f'http://localhost:{unused_tcp_port()}/' + 'p' * 1_000
        result = await _toolset(endpoint_url=endpoint_url, max_output_chars=100).localstack_health()
        assert len(result) <= 100
        assert 'output truncated' in result


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

    def test_instructions_name_the_managed_container_endpoint(self) -> None:
        cap = LocalStack(manage_container=True, host_address='127.0.0.2', endpoint_url='http://localhost:4599')
        instructions = cap.get_instructions()
        assert instructions is not None
        assert 'http://127.0.0.2:4599' in instructions
        assert 'localhost:4599' not in instructions

    def test_repr_hides_secrets(self) -> None:
        cap = LocalStack(secret_access_key='secret-key', container_env={'LOCALSTACK_AUTH_TOKEN': 'secret-token'})
        assert 'secret-key' not in repr(cap)
        assert 'secret-token' not in repr(cap)

    def test_invalid_settings_raise_at_construction(self) -> None:
        with pytest.raises(ValueError, match=r'Specify `allowed_services` or `denied_services`, not both.'):
            LocalStack(allowed_services=['s3'], denied_services=['sqs'])
        with pytest.raises(ValueError, match=r'`max_output_chars` must be a positive integer.'):
            LocalStack(max_output_chars=0)

    def test_invalid_managed_endpoint_port_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match=r"`endpoint_url` 'http://localhost:port' has an invalid port"):
            LocalStack(manage_container=True, endpoint_url='http://localhost:port')

    def test_managed_endpoint_port_zero_keeps_the_edge_port(self) -> None:
        cap = LocalStack(manage_container=True, endpoint_url='http://localhost:0')
        assert 'http://localhost.localstack.cloud:4566' in (cap.get_instructions() or '')

    def test_managed_scoped_ipv6_host_address_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match=r"`host_address` 'fe80::1%eth0' is a scoped IPv6 address"):
            LocalStack(manage_container=True, host_address='fe80::1%eth0')

    def test_instructions_reach_ipv6_all_interface_binding_at_loopback(self) -> None:
        cap = LocalStack(manage_container=True, host_address='::', endpoint_url='http://localhost:4599')
        assert 'http://[::1]:4599' in (cap.get_instructions() or '')

    def test_loads_from_agent_spec(self) -> None:
        agent = Agent.from_spec(
            {
                'model': 'test',
                'capabilities': [{'LocalStack': {'endpoint_url': 'http://ls:4566', 'allowed_services': ['s3']}}],
            },
            custom_capability_types=[LocalStack],
        )
        loaded = [c for c in agent.root_capability.capabilities if isinstance(c, LocalStack)]
        assert loaded == [LocalStack(endpoint_url='http://ls:4566', allowed_services=['s3'])]

    async def test_agent_integration(self) -> None:

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        model = TestModel(custom_output_text='done', call_tools=[])
        agent: Agent[None, str] = Agent(model, capabilities=[LocalStack()])
        result = await agent.run('create a bucket')
        assert result.output == 'done'

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

    async def test_reentry_stops_a_container_whose_stop_failed_before_starting_another(self, tmp_path: Path) -> None:
        log = tmp_path / 'docker.log'
        stop_failed = tmp_path / 'stop-failed'
        docker = _make_stub(
            tmp_path,
            f'echo "$@" >> {log}\n'
            'if [ "$1" = run ]; then echo managed-xyz; fi\n'
            f'if [ "$1" = stop ] && [ ! -e {stop_failed} ]; then touch {stop_failed}; echo busy >&2; exit 1; fi',
        )
        with http_server([HttpResponse(200), HttpResponse(200)]) as health:
            toolset = _toolset(endpoint_url=health.endpoint_url, manage_container=True, docker_path=docker)
            with pytest.raises(LocalStackError, match='Failed to stop LocalStack container managed-xyz: busy'):
                async with toolset:
                    pass
            async with toolset:
                pass
        assert [line.split()[0] for line in log.read_text().splitlines()] == ['run', 'stop', 'stop', 'run', 'stop']

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

    @pytest.mark.parametrize('manage_container', [False, True])
    async def test_only_commands_against_an_external_instance_are_recorded(
        self, tmp_path: Path, manage_container: bool
    ) -> None:
        """A managed container starts empty when a run recovers, so its commands run again to rebuild it."""
        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        docker, _ = _docker_stub(tmp_path)
        with http_server([HttpResponse(200)]) as health:
            cap = LocalStack(
                endpoint_url=health.endpoint_url,
                manage_container=manage_container,
                docker_path=docker,
                aws_cli_path=_make_stub(tmp_path, 'echo ok'),
            )
            agent: Agent[None, str] = Agent(
                model=TestModel(call_tools=['aws_cli']), name='localstack', capabilities=[cap, RecordingDurability()]
            )
            await agent.run('list buckets')
        bound = RecordingDurability.from_agent(agent)
        assert bound is not None
        recorded = 'localstack__capability__localstack.aws_cli' in {name for name, _ in bound.calls}
        assert recorded is not manage_container

    async def test_agent_integration_manages_container(self, tmp_path: Path) -> None:

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        docker, log = _docker_stub(tmp_path)
        with http_server([HttpResponse(200)]) as health:
            cap = LocalStack(endpoint_url=health.endpoint_url, manage_container=True, docker_path=docker)
            agent: Agent[None, str] = Agent(
                model=TestModel(custom_output_text='done', call_tools=[]), capabilities=[cap]
            )
            result = await agent.run('provision a bucket')
        assert result.output == 'done'
        assert 'stop managed-xyz' in log.read_text()

    async def test_entered_agent_runs_share_one_container(self, tmp_path: Path) -> None:
        """`async with agent:` starts the container once; without sharing, each run would publish the same port again."""
        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        docker, log = _docker_stub(tmp_path)
        with http_server([HttpResponse(200)]) as health:
            cap = LocalStack(endpoint_url=health.endpoint_url, manage_container=True, docker_path=docker)
            agent: Agent[None, str] = Agent(
                model=TestModel(custom_output_text='done', call_tools=[]), capabilities=[cap]
            )
            async with agent:
                await agent.run('first')
                await agent.run('second')
                assert [line.split()[0] for line in log.read_text().splitlines()] == ['run']
        assert [line.split()[0] for line in log.read_text().splitlines()] == ['run', 'stop']

    async def test_reentrant_entry_starts_and_stops_one_container(self, tmp_path: Path) -> None:
        docker, log = _docker_stub(tmp_path)
        with http_server([HttpResponse(200)]) as health:
            toolset = _toolset(endpoint_url=health.endpoint_url, manage_container=True, docker_path=docker)
            async with toolset:
                assert await toolset.for_run(_run_context()) is toolset
                async with toolset:
                    pass
                assert [line.split()[0] for line in log.read_text().splitlines()] == ['run']
        assert [line.split()[0] for line in log.read_text().splitlines()] == ['run', 'stop']

    async def test_for_run_shares_an_unmanaged_instance(self) -> None:
        toolset = _toolset()
        assert await toolset.for_run(_run_context()) is toolset

    async def test_for_run_keeps_tools_added_after_construction(self, tmp_path: Path) -> None:
        docker, _ = _docker_stub(tmp_path)
        toolset = _toolset(manage_container=True, docker_path=docker)

        def extra() -> str:
            return 'extra'

        toolset.add_function(extra)
        fresh = await toolset.for_run(_run_context())
        assert isinstance(fresh, LocalStackToolset)
        assert fresh is not toolset
        tools = await fresh.get_tools(_run_context())
        assert set(tools) == {'aws_cli', 'localstack_health', 'extra'}
        assert await _call_localstack_tool(fresh, 'extra') == 'extra'

    async def test_bound_tool_carried_into_a_run_targets_its_container(self, tmp_path: Path) -> None:
        docker, _ = _docker_stub(tmp_path)
        aws = _make_stub(tmp_path, 'echo "$AWS_ENDPOINT_URL"')
        with http_server([HttpResponse(200)]) as health:
            toolset = _toolset(
                endpoint_url=health.endpoint_url, manage_container=True, docker_path=docker, aws_cli_path=aws
            )
            toolset.add_function(toolset.aws_cli, name='aws_alias')
            fresh = await toolset.for_run(_run_context())
            assert isinstance(fresh, LocalStackToolset)
            async with fresh:
                via_alias = await _call_localstack_tool(fresh, 'aws_alias', command='s3 ls')
                direct = await _call_localstack_tool(fresh, 'aws_cli', command='s3 ls')
        assert via_alias == direct == f'[stdout]\nhttp://localhost.localstack.cloud:{health.port}\n'

    async def test_managed_ipv6_host_address_is_bracketed(self, tmp_path: Path) -> None:
        # `run` fails at once, so the published ports are checked without waiting for readiness.
        log = tmp_path / 'docker.log'
        docker = _make_stub(tmp_path, f'echo "$@" >> {log}\nexit 1')
        instructions = LocalStack(
            manage_container=True, host_address='::1', endpoint_url='http://localhost:4599'
        ).get_instructions()
        assert instructions is not None
        assert 'http://[::1]:4599' in instructions
        toolset = LocalStackToolset[None](
            endpoint_url='http://localhost:4599',
            region='us-east-1',
            access_key_id='test',
            secret_access_key='test',
            allowed_services=[],
            denied_services=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            aws_cli_path='aws',
            manage_container=True,
            host_address='::1',
            service_port_range='4510-4559',
            docker_path=docker,
        )
        with pytest.raises(LocalStackError, match='Failed to start LocalStack container'):
            async with toolset:
                pass  # pragma: no cover
        assert '-p [::1]:4599:4566 -p [::1]:4510-4559:4510-4559' in log.read_text()

    def test_invalid_managed_endpoint_port_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"`endpoint_url` 'http://localhost:port' has an invalid port"):
            _toolset(endpoint_url='http://localhost:port', manage_container=True)

    def test_managed_scoped_ipv6_host_address_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"`host_address` 'fe80::1%eth0' is a scoped IPv6 address"):
            _toolset(host_address='fe80::1%eth0', manage_container=True)
