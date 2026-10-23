"""Docker-backed lifecycle for a LocalStack container."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit

import anyio
import anyio.to_thread
import httpx

_EDGE_PORT = 4566
_HEALTH_PATH = '/_localstack/health'
_AUTH_TOKEN_ENV = 'LOCALSTACK_AUTH_TOKEN'
_LEGACY_API_KEY_ENV = 'LOCALSTACK_API_KEY'
_DOCKER_SOCKET = '/var/run/docker.sock'
_GATEWAY_LISTEN_ENV = 'GATEWAY_LISTEN'
_GATEWAY_LISTEN = f'0.0.0.0:{_EDGE_PORT}'


class LocalStackError(RuntimeError):
    """Raised when the LocalStack container cannot be started or becomes ready."""


_BIND_ALL_LOOPBACK = {'0.0.0.0': '127.0.0.1', '::': '::1'}
"""Bind-all publish addresses, which are not connectable, mapped to the loopback address that reaches them."""


def _url_host(host: str) -> str:
    """Bracket an IPv6 literal so it can be used in a URL or a Docker port mapping."""
    return f'[{host}]' if ':' in host else host


def _connect_host(host_address: str) -> str:
    """Return the URL host that reaches a container published on `host_address`."""
    return _url_host(_BIND_ALL_LOOPBACK.get(host_address, host_address))


def validate_host_address(host_address: str) -> None:
    """Reject a publish address that the URL and the Docker port mapping cannot both carry."""
    if '%' in host_address:
        raise ValueError(f'`host_address` {host_address!r} is a scoped IPv6 address, which is not supported.')


def managed_endpoint_url(host_address: str, host_port: int) -> str:
    """Return the edge URL of a container published on `host_address:host_port`.

    For the default loopback publish addresses (`127.0.0.1` and the bind-all
    `0.0.0.0`, both reachable at `127.0.0.1`), uses LocalStack's
    `localhost.localstack.cloud` domain, which resolves to `127.0.0.1` and
    supports the subdomain-style hosts some AWS SDKs need. The IPv6 bind-all
    `::` is reached at `::1`. Any other address is used literally, since
    `localhost.localstack.cloud` does not resolve there.
    """
    host = 'localhost.localstack.cloud' if host_address in ('127.0.0.1', '0.0.0.0') else _connect_host(host_address)
    return f'http://{host}:{host_port}'


def endpoint_port(endpoint_url: str) -> int:
    """Return the port of `endpoint_url`, or LocalStack's edge port when it names none or port 0."""
    try:
        port = urlsplit(endpoint_url).port
    except ValueError as e:
        raise ValueError(f'`endpoint_url` {endpoint_url!r} has an invalid port: {e}') from e
    return port or _EDGE_PORT


class LocalStackContainer:
    """Async context manager that starts and stops a LocalStack Docker container.

    Drives the `docker` CLI, so Docker must be installed and running. On enter it
    launches the container, polls the health endpoint until LocalStack is ready,
    and exposes `endpoint_url`. On exit it stops the container, which is started
    with `--rm`, so stopping also removes it.

    ```python
    async with LocalStackContainer() as localstack:
        ...  # talk to localstack.endpoint_url
    ```
    """

    def __init__(
        self,
        *,
        image: str = 'localstack/localstack',
        host_port: int = _EDGE_PORT,
        host_address: str = '127.0.0.1',
        service_port_range: str | None = None,
        mount_docker_socket: bool = False,
        docker_socket_path: str = _DOCKER_SOCKET,
        container_name: str | None = None,
        environment: Mapping[str, str] | None = None,
        docker_path: str = 'docker',
        startup_timeout: float = 120.0,
        poll_interval: float = 1.0,
    ) -> None:
        validate_host_address(host_address)
        self._image = image
        self._host_port = host_port
        self._host_address = host_address
        self._service_port_range = service_port_range
        self._mount_docker_socket = mount_docker_socket
        self._docker_socket_path = docker_socket_path
        self._container_name = container_name
        self._environment = dict(environment or {})
        self._docker_path = docker_path
        self._startup_timeout = startup_timeout
        self._poll_interval = poll_interval
        self._container_id: str | None = None

    @property
    def endpoint_url(self) -> str:
        """URL of the container's edge endpoint.

        For the default loopback publish addresses (`127.0.0.1` and the
        bind-all `0.0.0.0`, both reachable at `127.0.0.1`), uses LocalStack's
        `localhost.localstack.cloud` domain, which resolves to `127.0.0.1` and
        supports the subdomain-style hosts some AWS SDKs need. The IPv6 bind-all
        `::` is reached at `::1`. For any other `host_address`, uses that address
        literally (bracketed when it is an IPv6 literal), since
        `localhost.localstack.cloud` does not resolve there.
        """
        return managed_endpoint_url(self._host_address, self._host_port)

    @property
    def _readiness_url(self) -> str:
        return f'http://{_connect_host(self._host_address)}:{self._host_port}{_HEALTH_PATH}'

    @property
    def container_id(self) -> str | None:
        """The running container's id, or None when it is not running."""
        return self._container_id

    async def __aenter__(self) -> Self:
        """Start the container and wait for it to become ready."""
        self._container_id = await self._start()
        try:
            await self._wait_until_ready()
        except BaseException:
            try:
                await self._stop()
            except LocalStackError:
                pass
            raise
        return self

    async def __aexit__(self, *args: object) -> None:
        """Stop and remove the container."""
        await self._stop()

    def _run_argv(self, environment: Mapping[str, str] | None = None) -> list[str]:
        """Build the `docker run` argument vector."""
        container_environment = self._effective_environment() if environment is None else environment
        host = _url_host(self._host_address)
        argv = [
            self._docker_path,
            'run',
            '-d',
            '--rm',
            '-p',
            f'{host}:{self._host_port}:{_EDGE_PORT}',
        ]
        if self._service_port_range is not None:
            argv += ['-p', f'{host}:{self._service_port_range}:{self._service_port_range}']
        if self._mount_docker_socket:
            if not Path(self._docker_socket_path).exists():
                raise LocalStackError(f'Docker socket {self._docker_socket_path!r} not found.')
            argv += ['-v', f'{self._docker_socket_path}:{_DOCKER_SOCKET}']
        for key in container_environment:
            argv += ['-e', key]
        if self._container_name is not None:
            argv += ['--name', self._container_name]
        argv.append(self._image)
        return argv

    def _effective_environment(self) -> dict[str, str]:
        """Return container env, forwarding LocalStack auth from the process when present."""
        environment = dict(self._environment)
        environment.setdefault(_GATEWAY_LISTEN_ENV, _GATEWAY_LISTEN)
        if _AUTH_TOKEN_ENV in environment or _LEGACY_API_KEY_ENV in environment:
            return environment
        auth_token = os.environ.get(_AUTH_TOKEN_ENV)
        if auth_token:
            environment[_AUTH_TOKEN_ENV] = auth_token
            return environment
        legacy_api_key = os.environ.get(_LEGACY_API_KEY_ENV)
        if legacy_api_key:
            environment[_LEGACY_API_KEY_ENV] = legacy_api_key
        return environment

    def _docker_environment(self, container_environment: Mapping[str, str]) -> dict[str, str]:
        """Return the Docker CLI environment used to forward container env values."""
        environment = dict(os.environ)
        environment.update(container_environment)
        return environment

    async def _run_docker(self, argv: list[str], environment: Mapping[str, str]) -> subprocess.CompletedProcess[bytes]:
        """Run a docker command in a worker thread, bounded by `startup_timeout`.

        A blocking `subprocess.run` with a timeout is used instead of an async
        subprocess because it kills and reaps the child when the timeout fires.
        Cancelling an async subprocess mid-run can leave a dangling process on
        the asyncio backend, so an unresponsive daemon would otherwise leak a
        process (and emit unraisable-exception warnings) instead of failing cleanly.
        """
        docker_env = dict(environment)

        def run() -> subprocess.CompletedProcess[bytes]:
            return subprocess.run(argv, env=docker_env, capture_output=True, timeout=self._startup_timeout)

        return await anyio.to_thread.run_sync(run)

    async def _start(self) -> str:
        """Launch the container detached and return its id."""
        container_environment = self._effective_environment()
        try:
            result = await self._run_docker(
                self._run_argv(container_environment),
                self._docker_environment(container_environment),
            )
        except FileNotFoundError as e:
            raise LocalStackError(
                f'Docker CLI {self._docker_path!r} not found. Install Docker to manage LocalStack.'
            ) from e
        except subprocess.TimeoutExpired as e:
            raise LocalStackError(
                f'Docker did not start the LocalStack container within {self._startup_timeout}s.'
            ) from e
        if result.returncode != 0:
            stderr = result.stderr.decode('utf-8', errors='replace').strip()
            raise LocalStackError(f'Failed to start LocalStack container: {stderr}')
        return result.stdout.decode('utf-8', errors='replace').strip()

    async def _wait_until_ready(self) -> None:
        """Poll the health endpoint until LocalStack responds or `startup_timeout` elapses."""
        url = self._readiness_url
        try:
            with anyio.fail_after(self._startup_timeout):
                async with httpx.AsyncClient() as client:
                    while not await self._is_ready(client, url):
                        await anyio.sleep(self._poll_interval)
        except TimeoutError as e:
            raise LocalStackError(f'LocalStack did not become ready within {self._startup_timeout}s.') from e

    async def _is_ready(self, client: httpx.AsyncClient, url: str) -> bool:
        """Return True once the health endpoint answers with HTTP 200."""
        try:
            response = await client.get(url)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def _stop(self) -> None:
        """Stop the container if one is running, shielded from cancellation.

        Clears `container_id` only after `docker stop` succeeds. A non-zero exit
        code, a missing docker CLI, or a timeout raises `LocalStackError` and
        leaves `container_id` set, so the state reflects that the container (and
        its published ports) may still be running.
        """
        if self._container_id is None:
            return
        container_id = self._container_id
        with anyio.CancelScope(shield=True):
            try:
                result = await self._run_docker([self._docker_path, 'stop', container_id], self._docker_environment({}))
            except FileNotFoundError as e:
                raise LocalStackError(
                    f'Docker CLI {self._docker_path!r} not found while stopping LocalStack container {container_id}.'
                ) from e
            except subprocess.TimeoutExpired as e:
                raise LocalStackError(
                    f'Docker did not stop the LocalStack container {container_id} within {self._startup_timeout}s.'
                ) from e
            if result.returncode != 0:
                stderr = result.stderr.decode('utf-8', errors='replace').strip()
                raise LocalStackError(f'Failed to stop LocalStack container {container_id}: {stderr}')
        self._container_id = None
