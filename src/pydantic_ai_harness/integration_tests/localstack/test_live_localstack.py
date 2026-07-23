"""Live Docker integration tests for the LocalStack capability."""

from __future__ import annotations

import os
import shlex
import shutil
import socket
import subprocess
import uuid
from pathlib import Path

import anyio
import httpx
import pytest
from pydantic import TypeAdapter

from pydantic_ai_harness.localstack import LocalStack, LocalStackContainer, LocalStackToolset

_LOCALSTACK_INFO = TypeAdapter(dict[str, object])
_STARTUP_TIMEOUT = 240.0


@pytest.fixture
def anyio_backend() -> str:
    """Run live Docker tests once under asyncio."""
    return 'asyncio'


def _requires_auth_token() -> bool:
    return os.environ.get('LOCALSTACK_REQUIRE_AUTH_TOKEN', '').lower() in {'1', 'true', 'yes'}


def _auth_environment() -> dict[str, str]:
    auth_token = os.environ.get('LOCALSTACK_AUTH_TOKEN')
    if auth_token:
        return {'LOCALSTACK_AUTH_TOKEN': auth_token}
    legacy_api_key = os.environ.get('LOCALSTACK_API_KEY')
    if legacy_api_key:
        return {'LOCALSTACK_API_KEY': legacy_api_key}
    message = 'Set LOCALSTACK_AUTH_TOKEN to run live LocalStack integration tests.'
    if _requires_auth_token():
        pytest.fail(message)
    return {}


def _docker_path() -> str:
    docker = shutil.which('docker')
    if docker is None:
        pytest.skip('Docker CLI is required for live LocalStack integration tests.')
    result = subprocess.run([docker, 'info'], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        pytest.skip('Docker daemon is required for live LocalStack integration tests.')
    return docker


def _aws_cli_path() -> str:
    configured = os.environ.get('LOCALSTACK_AWS_CLI')
    if configured:
        return configured
    aws_cli = shutil.which('aws') or shutil.which('awslocal')
    if aws_cli is None:
        pytest.skip('AWS CLI or awslocal is required for live LocalStack integration tests.')
    return aws_cli


def _localstack_image() -> str:
    return os.environ.get('LOCALSTACK_IMAGE', 'localstack/localstack')


def _external_endpoint_url() -> str | None:
    return os.environ.get('LOCALSTACK_ENDPOINT_URL')


def _skip_managed_container_tests() -> bool:
    return os.environ.get('LOCALSTACK_SKIP_MANAGED_CONTAINERS', '').lower() in {'1', 'true', 'yes'}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    assert isinstance(port, int)
    return port


def _container_environment(services: str) -> dict[str, str]:
    return {'SERVICES': services, **_auth_environment()}


def _bucket_name() -> str:
    return f'harness-{uuid.uuid4().hex}'


def _queue_name() -> str:
    return f'harness-{uuid.uuid4().hex}'


def test_auth_token_is_not_required_for_default_live_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default live tests should exercise community LocalStack without Pro credentials."""
    monkeypatch.delenv('LOCALSTACK_AUTH_TOKEN', raising=False)
    monkeypatch.delenv('LOCALSTACK_API_KEY', raising=False)
    monkeypatch.delenv('LOCALSTACK_REQUIRE_AUTH_TOKEN', raising=False)

    try:
        environment = _auth_environment()
    except pytest.skip.Exception as exc:  # pragma: no cover - regression path
        pytest.fail(f'default live tests should run against community LocalStack without auth: {exc}')

    assert environment == {}


def test_default_live_image_is_community_localstack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default live tests should not require the Pro image."""
    monkeypatch.delenv('LOCALSTACK_IMAGE', raising=False)

    assert _localstack_image() == 'localstack/localstack'


def test_live_container_environment_scopes_requested_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live tests should keep harness-started LocalStack containers narrow enough for CI startup."""
    monkeypatch.delenv('LOCALSTACK_AUTH_TOKEN', raising=False)
    monkeypatch.delenv('LOCALSTACK_API_KEY', raising=False)
    monkeypatch.delenv('LOCALSTACK_REQUIRE_AUTH_TOKEN', raising=False)

    assert _container_environment('s3') == {'SERVICES': 's3'}


def test_live_tests_run_on_asyncio_backend(anyio_backend: str) -> None:
    """Live Docker tests should not duplicate slow container startup under trio."""
    assert anyio_backend == 'asyncio'


def test_managed_container_round_trips_can_be_skipped_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI can use the setup action's LocalStack service instead of starting nested containers."""
    monkeypatch.setenv('LOCALSTACK_SKIP_MANAGED_CONTAINERS', '1')

    assert _skip_managed_container_tests() is True


def _assert_success(output: str) -> None:
    assert '[exit code:' not in output, output
    assert '[error:' not in output, output


async def _info(endpoint_url: str) -> dict[str, object]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(endpoint_url.rstrip('/') + '/_localstack/info')
    response.raise_for_status()
    return _LOCALSTACK_INFO.validate_json(response.content)


async def _is_reachable(endpoint_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            await client.get(endpoint_url.rstrip('/') + '/_localstack/health')
    except httpx.HTTPError:
        return False
    return True


async def _wait_until_unreachable(endpoint_url: str) -> None:
    with anyio.fail_after(20.0):
        while await _is_reachable(endpoint_url):
            await anyio.sleep(0.25)


def _assert_license_if_auth_required(info: dict[str, object]) -> None:
    if _requires_auth_token():
        assert info.get('is_license_activated') is True


async def _s3_round_trip(endpoint_url: str, aws_cli: str, tmp_path: Path) -> None:
    info = await _info(endpoint_url)
    _assert_license_if_auth_required(info)

    toolset = LocalStack(endpoint_url=endpoint_url, aws_cli_path=aws_cli).get_toolset()
    assert isinstance(toolset, LocalStackToolset)

    bucket = _bucket_name()
    payload = tmp_path / 'payload.txt'
    payload.write_text('hello from pydantic-ai-harness\n')

    create_bucket = await toolset.aws_cli(f's3api create-bucket --bucket {bucket}', timeout_seconds=60.0)
    _assert_success(create_bucket)

    put_object = await toolset.aws_cli(
        f's3api put-object --bucket {bucket} --key payload.txt --body {shlex.quote(str(payload))}',
        timeout_seconds=60.0,
    )
    _assert_success(put_object)

    list_objects = await toolset.aws_cli(f's3api list-objects-v2 --bucket {bucket}', timeout_seconds=60.0)
    _assert_success(list_objects)
    assert 'payload.txt' in list_objects

    health = await toolset.localstack_health()
    assert '[error:' not in health


@pytest.mark.anyio(backends=['asyncio'])
async def test_existing_localstack_s3_round_trip(tmp_path: Path) -> None:
    """Drive S3 through a LocalStack service started by the test environment."""
    endpoint_url = _external_endpoint_url()
    if endpoint_url is None:
        pytest.skip('Set LOCALSTACK_ENDPOINT_URL to test against an existing LocalStack service.')

    await _s3_round_trip(endpoint_url, _aws_cli_path(), tmp_path)


@pytest.mark.anyio(backends=['asyncio'])
async def test_external_container_s3_round_trip(tmp_path: Path) -> None:
    """Drive S3 through an unmanaged capability against a harness-started container."""
    if _skip_managed_container_tests():
        pytest.skip('Managed container round trips are disabled for this environment.')
    docker = _docker_path()
    aws_cli = _aws_cli_path()
    port = _free_port()

    async with LocalStackContainer(
        image=_localstack_image(),
        host_port=port,
        environment=_container_environment('s3'),
        docker_path=docker,
        startup_timeout=_STARTUP_TIMEOUT,
    ) as localstack:
        await _s3_round_trip(localstack.endpoint_url, aws_cli, tmp_path)


@pytest.mark.anyio(backends=['asyncio'])
async def test_managed_container_sqs_round_trip_and_cleanup() -> None:
    """Drive SQS through `manage_container=True` and verify cleanup."""
    if _skip_managed_container_tests():
        pytest.skip('Managed container round trips are disabled for this environment.')
    docker = _docker_path()
    aws_cli = _aws_cli_path()
    port = _free_port()
    endpoint_url = f'http://localhost:{port}'

    toolset = LocalStack(
        endpoint_url=endpoint_url,
        aws_cli_path=aws_cli,
        manage_container=True,
        image=_localstack_image(),
        container_env=_container_environment('sqs'),
        docker_path=docker,
        startup_timeout=_STARTUP_TIMEOUT,
    ).get_toolset()
    assert isinstance(toolset, LocalStackToolset)

    async with toolset:
        info = await _info(endpoint_url)
        _assert_license_if_auth_required(info)

        create_queue = await toolset.aws_cli(f'sqs create-queue --queue-name {_queue_name()}', timeout_seconds=60.0)
        _assert_success(create_queue)
        assert 'QueueUrl' in create_queue

    await _wait_until_unreachable(endpoint_url)
