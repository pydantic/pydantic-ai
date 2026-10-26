"""LocalStack toolset that gives agents access to an emulated AWS environment."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import IO, Annotated, Any, Self

import anyio
import anyio.to_thread
import httpx
from pydantic import Field

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool
from pydantic_ai_harness._durable import RetryRequest, raise_retry
from pydantic_ai_harness._output import truncate_tail
from pydantic_ai_harness.localstack._container import (
    LocalStackContainer,
    endpoint_port,
    managed_endpoint_url,
    validate_host_address,
)

_HEALTH_PATH = '/_localstack/health'
_UTF8_MAX_BYTES_PER_CHAR = 4
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
_AWS_GLOBAL_FLAGS = {
    '--cli-auto-prompt',
    '--debug',
    '--no-cli-auto-prompt',
    '--no-cli-pager',
    '--no-paginate',
    '--no-sign-request',
    '--no-verify-ssl',
    '--version',
}
_AWS_GLOBAL_OPTIONS = _AWS_GLOBAL_OPTIONS_WITH_VALUE | _AWS_GLOBAL_FLAGS
_FORBIDDEN_MODEL_GLOBAL_OPTIONS = {
    '--endpoint-url',
    '--no-sign-request',
    '--profile',
    '--region',
}


def validate_settings(
    *, allowed_services: Sequence[str], denied_services: Sequence[str], max_output_chars: int
) -> None:
    """Reject settings that `LocalStack` and `LocalStackToolset` cannot run with."""
    if allowed_services and denied_services:
        raise ValueError('Specify `allowed_services` or `denied_services`, not both.')
    if max_output_chars <= 0:
        raise ValueError('`max_output_chars` must be a positive integer.')


def _resolve_global_option(name: str) -> str | None:
    """Return the AWS CLI global option `name` selects, or None when it selects none.

    Matches the CLI's argparse parsing: an exact name wins, otherwise a prefix
    selects the one global option it starts (`--out` selects `--output`), and an
    ambiguous or unknown prefix selects none.
    """
    if name in _AWS_GLOBAL_OPTIONS:
        return name
    matches = [option for option in _AWS_GLOBAL_OPTIONS if option.startswith(name)]
    return matches[0] if len(matches) == 1 else None


def _read_tail(file: IO[bytes], max_chars: int) -> str:
    """Decode the end of `file`, reading at most the bytes `max_chars` characters can take."""
    size = file.seek(0, os.SEEK_END)
    file.seek(max(0, size - max_chars * _UTF8_MAX_BYTES_PER_CHAR))
    return file.read().decode('utf-8', errors='replace')


@dataclass(frozen=True)
class LocalStackOperations:
    """The durable operations a `LocalStack` capability runs its toolset's commands and health checks through.

    Each takes the endpoint the toolset talks to, which is the managed container's when `manage_container` is set.
    """

    aws_cli: Callable[[str, str, float | None], Awaitable[str | RetryRequest]]
    localstack_health: Callable[[str], Awaitable[str]]


class LocalStackToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent the ability to drive an emulated AWS environment.

    Wraps the AWS CLI: `aws_cli` runs a command against a running LocalStack
    instance with the endpoint, region, and credentials injected, while
    `localstack_health` reports which emulated services are available.

    Commands are executed as an argument vector (no shell), so shell operators
    and redirection in the command string have no effect. Each command runs with
    stdin closed and with its own empty AWS config and credentials files in place
    of `~/.aws/config` and `~/.aws/credentials`.

    `LocalStack` passes `operations` so that each command and health check runs
    as one of its durable operations, whose result durable execution records
    instead of running the command again on recovery.
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
        id: str | None = None,
        operations: LocalStackOperations | None = None,
    ) -> None:
        super().__init__(id=id)
        validate_settings(
            allowed_services=allowed_services, denied_services=denied_services, max_output_chars=max_output_chars
        )
        self._host_port = endpoint_port(endpoint_url) if manage_container else None
        if manage_container:
            validate_host_address(host_address)

        # A managed container's endpoint depends only on where it is published, so every instance
        # sharing these settings, and any tool bound to one of them, names the same endpoint.
        self._endpoint_url = (
            endpoint_url if self._host_port is None else managed_endpoint_url(host_address, self._host_port)
        )
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
        self._entered_count = 0
        self._operations = operations

        self.add_function(self.aws_cli, name='aws_cli')
        self.add_function(self.localstack_health, name='localstack_health')

    @cached_property
    def _enter_lock(self) -> anyio.Lock:
        # Created on first entry so the lock binds to the loop that enters the toolset.
        return anyio.Lock()

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return the instance this run enters: a fresh one when it needs its own container.

        `get_toolset` builds one shared instance at agent construction. A toolset
        that does not manage a container holds no per-run state, so every run
        shares it. A managed toolset gives each run its own instance (and its own
        container) that `__aexit__` can stop, unless the shared instance already
        runs a container, as it does inside `async with agent:`. Runs then share
        that container, which stops when the agent context exits.
        """
        if not self._manage_container or self._container is not None:
            return self
        toolset = LocalStackToolset[AgentDepsT](
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
            id=self.id,
            operations=self._operations,
        )
        # Carry over tools added after construction; the built-in ones stay bound to the new instance.
        for name, tool in self.tools.items():
            toolset.tools.setdefault(name, tool)
        return toolset

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        """Cap every `str` tool result at the dispatch seam, as a backstop.

        `aws_cli` and `localstack_health` already cap their own output so durable
        records stay bounded; this seam also covers tools added after
        construction. A tool returning rich content (e.g. `ToolReturn`) needs
        this seam extended.
        """
        result = await super().call_tool(name, tool_args, ctx, tool)
        if not isinstance(result, str):
            return result
        return truncate_tail(result, self._max_output_chars)

    async def __aenter__(self) -> Self:
        """Start the managed LocalStack container, if configured, before tools run.

        Entering is reentrant: only the first entry starts the container and only
        the matching last exit stops it.
        """
        host_port = self._host_port
        if host_port is None:
            return self
        async with self._enter_lock:
            if self._entered_count == 0:
                if self._container is not None:
                    # The last exit failed to stop it; stop it now rather than start another on its port.
                    await self._container.__aexit__(None, None, None)
                    self._container = None
                container = LocalStackContainer(
                    image=self._image,
                    host_port=host_port,
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
            self._entered_count += 1
        return self

    async def __aexit__(self, *args: object) -> None:
        """Stop the managed LocalStack container once the last entry exits."""
        if self._host_port is None:
            return
        # Shielded so a cancelled run still releases its entry and stops the container.
        with anyio.CancelScope(shield=True):
            async with self._enter_lock:
                self._entered_count -= 1
                if self._entered_count == 0 and self._container is not None:
                    await self._container.__aexit__(*args)
                    self._container = None

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
        """Return the first token that is neither a global option nor its value: the AWS service name.

        A global option is recognized by any prefix the CLI accepts for it, so
        the value of `--out text` or `--quer s3` is skipped like that of
        `--output text` instead of being taken for the service.
        """
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == '--':
                return tokens[index + 1] if index + 1 < len(tokens) else None
            if token.startswith('--'):
                option = _resolve_global_option(token.split('=', 1)[0])
                index += 2 if '=' not in token and option in _AWS_GLOBAL_OPTIONS_WITH_VALUE else 1
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

    def _build_env(self, config_dir: str) -> dict[str, str]:
        """Inherit non-AWS environment, then set only the intended LocalStack AWS settings.

        The config and credentials files point into `config_dir`, so the CLI never
        falls back to `~/.aws/config` and `~/.aws/credentials`, where `aws configure`
        would read or overwrite the real AWS settings of the user running the agent.
        """
        env = {key: value for key, value in os.environ.items() if not key.startswith('AWS_')}
        env['AWS_CONFIG_FILE'] = os.path.join(config_dir, 'config')
        env['AWS_SHARED_CREDENTIALS_FILE'] = os.path.join(config_dir, 'credentials')
        env['AWS_ACCESS_KEY_ID'] = self._access_key_id
        env['AWS_SECRET_ACCESS_KEY'] = self._secret_access_key
        env['AWS_DEFAULT_REGION'] = self._region
        env['AWS_REGION'] = self._region
        env['AWS_ENDPOINT_URL'] = self._endpoint_url
        return env

    async def aws_cli(
        self, command: str, *, timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    ) -> str:
        """Run an AWS CLI command against the emulated AWS environment.

        Pass the command without the leading `aws` and without `--endpoint-url`;
        the endpoint, region, and credentials are injected automatically. For
        example `s3 mb s3://my-bucket`, `s3 ls`, or `dynamodb list-tables`.

        Args:
            command: The AWS CLI command to run (e.g. `s3 ls`).
            timeout_seconds: Maximum seconds to wait, greater than zero (default: the configured timeout).

        Returns:
            Labelled stdout/stderr output, with an exit code on non-zero exit.
        """
        if self._operations is not None:
            return raise_retry(await self._operations.aws_cli(self._endpoint_url, command, timeout_seconds))
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
        """Execute the AWS CLI argument vector and format its output, capped at `max_output_chars`.

        Output is captured to temp files rather than pipes: on a timeout the
        process is killed mid-run, and leftover pipe transports would otherwise
        leak (a `ResourceWarning` under cancellation-strict runtimes). Files have
        no transport to leak and the same approach works on every async backend.
        Only the tail that fits the cap is read back, and the cap applies here
        rather than only at dispatch, so a durable record of the command is
        bounded too.
        """
        with tempfile.TemporaryDirectory(prefix='harness_aws_') as work_dir:
            with (
                Path(work_dir, 'stdout').open('w+b') as stdout_file,
                Path(work_dir, 'stderr').open('w+b') as stderr_file,
            ):
                try:
                    proc = await anyio.open_process(
                        argv,
                        env=self._build_env(work_dir),
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_file,
                        stderr=stderr_file,
                    )
                except FileNotFoundError:
                    return (
                        f'[error: AWS CLI {self._aws_cli_path!r} not found. '
                        'Install the AWS CLI to use LocalStack tools.]'
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

                stdout, stderr = await anyio.to_thread.run_sync(
                    lambda: (
                        _read_tail(stdout_file, self._max_output_chars),
                        _read_tail(stderr_file, self._max_output_chars),
                    )
                )

        parts: list[str] = []
        if stdout:
            parts.append(f'[stdout]\n{stdout}')
        if stderr:
            parts.append(f'[stderr]\n{stderr}')
        output = '\n'.join(parts) if parts else '(no output)'

        exit_code = proc.returncode
        if exit_code:
            output = f'{output}\n[exit code: {exit_code}]'
        return truncate_tail(output, self._max_output_chars)

    async def localstack_health(self) -> str:
        """Report the health and availability of the emulated AWS services.

        Queries LocalStack's health endpoint and returns the raw JSON, which maps
        each service (s3, dynamodb, sqs, etc.) to its state (available, running, etc.).

        Returns:
            The health JSON, or an error message if LocalStack is unreachable.
        """
        if self._operations is not None:
            return await self._operations.localstack_health(self._endpoint_url)
        url = self._endpoint_url.rstrip('/') + _HEALTH_PATH
        try:
            async with httpx.AsyncClient(timeout=self._default_timeout) as client:
                response = await client.get(url)
        except httpx.HTTPError as e:
            return truncate_tail(f'[error: could not reach LocalStack at {url}: {e}]', self._max_output_chars)
        if response.status_code != 200:
            return f'[error: LocalStack health check returned HTTP {response.status_code}]'
        return truncate_tail(response.text, self._max_output_chars)
