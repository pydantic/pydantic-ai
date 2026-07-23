"""LocalStack toolset — gives agents access to an emulated AWS environment."""

from __future__ import annotations

import os
import shlex
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit

import anyio
import httpx
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from typing_extensions import Self

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


class LocalStackToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent the ability to drive an emulated AWS environment.

    Wraps the AWS CLI: `aws_cli` runs a command against a running LocalStack
    instance with the endpoint, region, and credentials injected, while
    `localstack_health` reports which emulated services are available.

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
        """Reject model-supplied AWS globals that can override the injected target or credentials."""
        for token in tokens:
            if not token.startswith('--'):
                continue
            name = token.split('=', 1)[0]
            if name in _FORBIDDEN_MODEL_GLOBAL_OPTIONS:
                forbidden = ', '.join(sorted(_FORBIDDEN_MODEL_GLOBAL_OPTIONS))
                raise ModelRetry(
                    f'Do not pass AWS global options that change the LocalStack target or credentials '
                    f'({forbidden}); the capability injects them.'
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

    def _build_env(self) -> dict[str, str]:
        """Inherit non-AWS environment, then set only the intended LocalStack AWS settings."""
        env = {key: value for key, value in os.environ.items() if not key.startswith('AWS_')}
        env['AWS_ACCESS_KEY_ID'] = self._access_key_id
        env['AWS_SECRET_ACCESS_KEY'] = self._secret_access_key
        env['AWS_DEFAULT_REGION'] = self._region
        env['AWS_REGION'] = self._region
        env['AWS_ENDPOINT_URL'] = self._endpoint_url
        return env

    def _truncate(self, text: str) -> str:
        """Truncate output to the configured cap, keeping the tail.

        Errors and the `[stderr]` section land at the end, so the head is
        dropped and the final `max_output_chars` are kept.
        """
        if len(text) <= self._max_output_chars:
            return text
        marker = f'[... output truncated, showing last {self._max_output_chars} chars]\n'
        return marker + text[-self._max_output_chars :]

    async def aws_cli(self, command: str, *, timeout_seconds: float | None = None) -> str:
        """Run an AWS CLI command against the emulated AWS environment.

        Pass the command without the leading `aws` and without `--endpoint-url`;
        the endpoint, region, and credentials are injected automatically. For
        example `s3 mb s3://my-bucket`, `s3 ls`, or `dynamodb list-tables`.

        Args:
            command: The AWS CLI command to run (e.g. `s3 ls`).
            timeout_seconds: Maximum seconds to wait (default: the configured timeout).

        Returns:
            Labelled stdout/stderr output, with an exit code on non-zero exit.
        """
        tokens = self._normalize_command(command)
        self._check_service(tokens)
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout
        argv = [
            self._aws_cli_path,
            '--endpoint-url',
            self._endpoint_url,
            '--region',
            self._region,
            *tokens,
        ]
        return await self._run(argv, timeout)

    async def _run(self, argv: list[str], timeout: float) -> str:
        """Execute the AWS CLI argument vector and format its output.

        Output is captured to temp files rather than pipes: on a timeout the
        process is killed mid-run, and leftover pipe transports would otherwise
        leak (a `ResourceWarning` under cancellation-strict runtimes). Files have
        no transport to leak and the same approach works on every async backend.
        """
        stdout_file = tempfile.NamedTemporaryFile(mode='w+b', prefix='harness_aws_out_', delete=False)
        stderr_file = tempfile.NamedTemporaryFile(mode='w+b', prefix='harness_aws_err_', delete=False)
        try:
            try:
                proc = await anyio.open_process(
                    argv,
                    env=self._build_env(),
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
            except FileNotFoundError:
                return (
                    f'[error: AWS CLI {self._aws_cli_path!r} not found. Install the AWS CLI to use LocalStack tools.]'
                )

            try:
                try:
                    with anyio.fail_after(timeout):
                        await proc.wait()
                except TimeoutError:
                    proc.kill()
                    with anyio.CancelScope(shield=True):
                        await proc.wait()
                    return f'[command timed out after {timeout}s]'
            finally:
                await proc.aclose()

            stdout = Path(stdout_file.name).read_text(encoding='utf-8', errors='replace')
            stderr = Path(stderr_file.name).read_text(encoding='utf-8', errors='replace')

            parts: list[str] = []
            if stdout:
                parts.append(f'[stdout]\n{stdout}')
            if stderr:
                parts.append(f'[stderr]\n{stderr}')
            output = self._truncate('\n'.join(parts) if parts else '(no output)')

            exit_code = proc.returncode
            if exit_code:
                return f'{output}\n[exit code: {exit_code}]'
            return output
        finally:
            stdout_file.close()
            stderr_file.close()
            os.unlink(stdout_file.name)
            os.unlink(stderr_file.name)

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
        return self._truncate(response.text)
