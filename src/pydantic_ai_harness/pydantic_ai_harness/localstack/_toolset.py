"""LocalStack toolset — gives agents access to an emulated AWS environment."""

from __future__ import annotations

import math
import shlex
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx
from typing_extensions import Self

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool
from pydantic_ai.workspaces import WorkspaceError, WorkspaceTimeoutError
from pydantic_ai_harness._output import truncate_tail
from pydantic_ai_harness._workspace import raise_tool_failure, supports_commands
from pydantic_ai_harness.localstack._container import LocalStackContainer

_HEALTH_PATH = '/_localstack/health'
_DEFAULT_EDGE_PORT = 4566
_AWS_GLOBAL_OPTIONS_WITH_VALUE = {
    '--ca-bundle',
    '--cli-binary-format',
    '--cli-connect-timeout',
    '--cli-read-timeout',
    '--color',
    '--endpoint-url',
    '--output',
    '--profile',
    '--query',
    '--region',
}
_FORBIDDEN_MODEL_GLOBAL_OPTIONS = {
    '--endpoint-url',
    '--no-sign-request',
    '--profile',
    '--region',
}
_INJECTED_AWS_ENV = (
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'AWS_DEFAULT_REGION',
    'AWS_REGION',
    'AWS_ENDPOINT_URL',
)
"""The AWS variables the capability sets for every command; each maps to one of its settings."""

_NOT_FOUND_EXIT = 127
"""Exit status of `_LAUNCHER`, with no output, when the AWS CLI is not on the workspace's PATH."""

_LAUNCHER = (
    'AWS_ENV=$(env) || { echo "Could not read the workspace environment." >&2; exit 1; }\n'
    r"""AWS_NAMES=$(printf '%s\n' "$AWS_ENV" | sed -n 's/^\(AWS_[A-Za-z0-9_]*\)=.*/\1/p')"""
    ' || { echo "Could not filter AWS environment variables." >&2; exit 1; }\n'
    'for name in $AWS_NAMES; do\n'
    f'  case "$name" in {"|".join(_INJECTED_AWS_ENV)}) ;; *) unset "$name" ;; esac\n'
    'done\n'
    'unset AWS_ENV AWS_NAMES\n'
    f'command -v "$1" > /dev/null 2>&1 || exit {_NOT_FOUND_EXIT}\n'
    'exec "$@"'
)
"""Scrub the workspace's own AWS settings, look the CLI up, then replace the shell with it.

Every `AWS_*` variable except the injected ones is unset, so a profile, session token, or config
file meant for real AWS never reaches the CLI. `exec` lets the workspace's timeout reach the CLI.
"""


class LocalStackToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent the ability to drive an emulated AWS environment.

    Wraps the AWS CLI: `aws_cli` runs a command in the run's workspace (`ctx.workspace`)
    against a running LocalStack instance with the endpoint, region, and credentials
    injected, while `localstack_health` reports which emulated services are available.
    The health check and a managed container stay on the agent's host.

    Commands are executed as an argument vector (no shell), so shell operators
    and redirection in the command string have no effect.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        allowed_services: Sequence[str],
        denied_services: Sequence[str],
        default_timeout: float,
        max_output_chars: int,
        aws_cli_path: str,
        manage_container: bool = False,
        image: str = 'localstack/localstack',
        host_address: str = '127.0.0.1',
        service_port_range: str | None = None,
        mount_docker_socket: bool = False,
        container_name: str | None = None,
        container_env: Mapping[str, str] | None = None,
        docker_path: str = 'docker',
        startup_timeout: float = 120.0,
    ) -> None:
        super().__init__()
        if allowed_services and denied_services:
            raise ValueError('Specify allowed_services or denied_services, not both.')
        if max_output_chars <= 0:
            raise ValueError('max_output_chars must be a positive integer.')
        if not 0 < default_timeout < math.inf:
            raise ValueError('default_timeout must be a positive number of seconds.')

        self._endpoint_url = endpoint_url
        self._region = region
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._allowed_services = list(allowed_services)
        self._denied_services = list(denied_services)
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars
        self._aws_cli_path = aws_cli_path
        self._manage_container = manage_container
        self._image = image
        self._host_address = host_address
        self._service_port_range = service_port_range
        self._mount_docker_socket = mount_docker_socket
        self._container_name = container_name
        self._container_env = dict(container_env or {})
        self._docker_path = docker_path
        self._startup_timeout = startup_timeout
        self._container: LocalStackContainer | None = None

        self.add_function(self.aws_cli, name='aws_cli')
        self.add_function(self.localstack_health, name='localstack_health')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh instance per run so a managed container is isolated and torn down.

        `get_toolset` builds one shared instance at agent construction. When this
        toolset manages a Docker container it holds per-run lifecycle state, so each
        run gets its own instance (and its own container) that `__aexit__` can stop.
        """
        return LocalStackToolset[AgentDepsT](
            endpoint_url=self._endpoint_url,
            region=self._region,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            allowed_services=self._allowed_services,
            denied_services=self._denied_services,
            default_timeout=self._default_timeout,
            max_output_chars=self._max_output_chars,
            aws_cli_path=self._aws_cli_path,
            manage_container=self._manage_container,
            image=self._image,
            host_address=self._host_address,
            service_port_range=self._service_port_range,
            mount_docker_socket=self._mount_docker_socket,
            container_name=self._container_name,
            container_env=self._container_env,
            docker_path=self._docker_path,
            startup_timeout=self._startup_timeout,
        )

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Offer `aws_cli` only when the workspace can run commands."""
        tools = await super().get_tools(ctx)
        if not supports_commands(ctx.workspace):
            tools.pop('aws_cli')
        return tools

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        """Enforce the model-visible output cap at the tool dispatch seam.

        Only `str` results are capped; a future tool returning rich content
        (e.g. `ToolReturn`) needs this seam extended.
        """
        result = await super().call_tool(name, tool_args, ctx, tool)
        if not isinstance(result, str):
            return result
        return truncate_tail(result, self._max_output_chars)

    def _host_port(self) -> int:
        """Host port to bind the container's edge port to, parsed from `endpoint_url`."""
        return urlsplit(self._endpoint_url).port or _DEFAULT_EDGE_PORT

    async def __aenter__(self) -> Self:
        """Start the managed LocalStack container, if configured, before tools run."""
        if self._manage_container:
            container = LocalStackContainer(
                image=self._image,
                host_port=self._host_port(),
                host_address=self._host_address,
                service_port_range=self._service_port_range,
                mount_docker_socket=self._mount_docker_socket,
                container_name=self._container_name,
                environment=self._container_env,
                docker_path=self._docker_path,
                startup_timeout=self._startup_timeout,
            )
            await container.__aenter__()
            self._container = container
            self._endpoint_url = container.endpoint_url
        return self

    async def __aexit__(self, *args: object) -> None:
        """Stop the managed LocalStack container, if one was started."""
        if self._container is not None:
            container = self._container
            self._container = None
            await container.__aexit__(*args)

    def _normalize_command(self, command: str) -> list[str]:
        """Split the command into tokens, dropping a redundant leading `aws`.

        Raises `ModelRetry` for an empty or unparsable command so the model can
        correct itself instead of aborting the run.
        """
        try:
            tokens = shlex.split(command)
        except ValueError as e:
            raise ModelRetry(f'Could not parse the AWS CLI command: {e}') from e
        if tokens and tokens[0] == 'aws':
            tokens = tokens[1:]
        if not tokens:
            raise ModelRetry('Provide an AWS CLI command, e.g. "s3 ls" or "dynamodb list-tables".')
        self._check_global_options(tokens)
        return tokens

    def _service_name(self, tokens: Sequence[str]) -> str | None:
        """Return the first non-flag token, which is the AWS service name."""
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == '--':
                return tokens[index + 1] if index + 1 < len(tokens) else None
            if token.startswith('--'):
                name = token.split('=', 1)[0]
                if '=' not in token and name in _AWS_GLOBAL_OPTIONS_WITH_VALUE:
                    index += 2
                else:
                    index += 1
                continue
            return token
        return None

    def _check_global_options(self, tokens: Sequence[str]) -> None:
        """Reject model-supplied AWS globals that can override the injected target or credentials.

        The AWS CLI accepts any unambiguous prefix of a global option name (`--endpoint`
        for `--endpoint-url`, `--prof` for `--profile`) and a later value overrides an
        earlier one, so an exact-name check is not enough. Reject any `--` token whose
        name is a prefix of a forbidden option (the exact name is the full-length prefix).
        """
        for token in tokens:
            if not token.startswith('--'):
                continue
            name = token.split('=', 1)[0]
            if len(name) < 3:
                continue
            if any(forbidden.startswith(name) for forbidden in _FORBIDDEN_MODEL_GLOBAL_OPTIONS):
                forbidden = ', '.join(sorted(_FORBIDDEN_MODEL_GLOBAL_OPTIONS))
                raise ModelRetry(
                    f'Do not pass AWS global options that change the LocalStack target or credentials '
                    f'({forbidden}), including abbreviations of them; the capability injects them.'
                )

    def _check_service(self, tokens: Sequence[str]) -> None:
        """Validate the command's service against the allow/deny lists.

        These checks are best-effort and are not a security boundary. Restrict
        what LocalStack itself emulates for hard enforcement.
        """
        service = self._service_name(tokens)
        if service is None:
            raise ModelRetry('Could not determine the AWS service from the command.')
        if self._denied_services and service in self._denied_services:
            raise ModelRetry(f'AWS service {service!r} is denied.')
        if self._allowed_services and service not in self._allowed_services:
            raise ModelRetry(f'AWS service {service!r} is not in the allowed list.')

    def _aws_env(self) -> dict[str, str]:
        """The LocalStack AWS settings, set on top of the workspace's environment."""
        values = (self._access_key_id, self._secret_access_key, self._region, self._region, self._endpoint_url)
        return dict(zip(_INJECTED_AWS_ENV, values, strict=True))

    async def aws_cli(self, ctx: RunContext[AgentDepsT], command: str, *, timeout_seconds: float | None = None) -> str:
        """Run an AWS CLI command against the emulated AWS environment.

        Pass the command without the leading `aws` and without `--endpoint-url`;
        the endpoint, region, and credentials are injected automatically. For
        example `s3 mb s3://my-bucket`, `s3 ls`, or `dynamodb list-tables`.

        Args:
            ctx: The current agent run context.
            command: The AWS CLI command to run (e.g. `s3 ls`).
            timeout_seconds: Maximum seconds to wait (default: the configured timeout).

        Returns:
            Labelled stdout/stderr output, with an exit code on non-zero exit.
        """
        tokens = self._normalize_command(command)
        self._check_service(tokens)
        if timeout_seconds is not None and not 0 < timeout_seconds < math.inf:
            raise ModelRetry('`timeout_seconds` must be a positive number of seconds.')
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout
        argv = [
            self._aws_cli_path,
            '--endpoint-url',
            self._endpoint_url,
            '--region',
            self._region,
            *tokens,
        ]
        return await self._run(ctx, argv, timeout)

    async def _run(self, ctx: RunContext[AgentDepsT], argv: list[str], timeout: float) -> str:
        """Execute the AWS CLI argument vector in the workspace and format its output.

        The workspace enforces `timeout` and stops the command when it expires.
        """
        try:
            result = await ctx.workspace.run(['sh', '-c', _LAUNCHER, 'sh', *argv], env=self._aws_env(), timeout=timeout)
        # Before `WorkspaceError`: a timeout is reported to the model, other workspace failures fail the call.
        except WorkspaceTimeoutError:
            return f'[command timed out after {timeout}s]'
        except WorkspaceError as e:
            raise_tool_failure(e)

        if result.exit_code == _NOT_FOUND_EXIT and not result.stdout and not result.stderr:
            return (
                f'[error: AWS CLI {self._aws_cli_path!r} not found in the workspace. '
                'Install the AWS CLI there to use LocalStack tools.]'
            )

        parts: list[str] = []
        if result.stdout:
            parts.append(f'[stdout]\n{result.stdout}')
        if result.stderr:
            parts.append(f'[stderr]\n{result.stderr}')
        output = '\n'.join(parts) if parts else '(no output)'
        if result.exit_code:
            output = f'{output}\n[exit code: {result.exit_code}]'
        return output

    async def localstack_health(self) -> str:
        """Report the health and availability of the emulated AWS services.

        Queries LocalStack's health endpoint and returns the raw JSON, which maps
        each service (s3, dynamodb, sqs, …) to its state (available, running, …).

        Returns:
            The health JSON, or an error message if LocalStack is unreachable.
        """
        url = self._endpoint_url.rstrip('/') + _HEALTH_PATH
        try:
            async with httpx.AsyncClient(timeout=self._default_timeout) as client:
                response = await client.get(url)
        except httpx.HTTPError as e:
            return f'[error: could not reach LocalStack at {url}: {e}]'
        if response.status_code != 200:
            return f'[error: LocalStack health check returned HTTP {response.status_code}]'
        return response.text
