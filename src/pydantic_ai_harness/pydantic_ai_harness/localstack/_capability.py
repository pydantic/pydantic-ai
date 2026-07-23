"""LocalStack capability that gives agents an emulated AWS environment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.localstack._toolset import LocalStackToolset

_INSTRUCTIONS = (
    'You have access to an emulated AWS environment powered by LocalStack at {endpoint_url}. '
    'Use the `aws_cli` tool to run AWS CLI commands against it — pass the command without the '
    'leading `aws` and without `--endpoint-url`; the endpoint, region, and credentials are '
    'injected automatically. Use `localstack_health` to see which AWS services are available. '
    'This is a local emulator: resources are not real AWS resources and may be reset when '
    'LocalStack restarts.'
)


@dataclass
class LocalStack(AbstractCapability[AgentDepsT]):
    """Access to an emulated AWS environment via LocalStack.

    Gives the agent AWS CLI tooling wired to a running LocalStack instance, so it
    can provision and interact with AWS services (S3, DynamoDB, SQS, Lambda, …)
    without touching real AWS. Start LocalStack separately (`localstack start` or
    its Docker image) before running the agent.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.localstack import LocalStack

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[LocalStack()])
    result = agent.run_sync('Create an S3 bucket called reports and list all buckets.')
    print(result.output)
    ```
    """

    endpoint_url: str = 'http://localhost.localstack.cloud:4566'
    """Base URL of the running LocalStack instance.

    Defaults to LocalStack's `localhost.localstack.cloud` domain (which resolves to
    `127.0.0.1`) for compatibility with AWS SDKs that need subdomain-style hosts.
    """

    region: str = 'us-east-1'
    """AWS region passed to the CLI and exported to the environment."""

    access_key_id: str = 'test'
    """AWS access key id. LocalStack accepts any value; defaults to its `test` convention."""

    secret_access_key: str = 'test'
    """AWS secret access key. LocalStack accepts any value; defaults to its `test` convention."""

    allowed_services: Sequence[str] = field(default_factory=list[str])
    """If non-empty, only these AWS services may be used (allowlist), e.g. `['s3', 'dynamodb']`."""

    denied_services: Sequence[str] = field(default_factory=list[str])
    """These AWS services are always rejected (denylist)."""

    default_timeout: float = 60.0
    """Default timeout in seconds for AWS CLI commands and the health check."""

    max_output_chars: int = 50_000
    """Maximum characters of output returned to the model."""

    aws_cli_path: str = 'aws'
    """Path or name of the AWS CLI executable (e.g. `aws` or `awslocal`)."""

    manage_container: bool = False
    """If True, start a LocalStack Docker container for each run and stop it when the run ends.

    Requires Docker. When False (default), the agent connects to a LocalStack
    instance you started separately at `endpoint_url`.
    """

    image: str = 'localstack/localstack'
    """Docker image to run when `manage_container` is True."""

    host_address: str = '127.0.0.1'
    """Host address Docker publishes the LocalStack edge port on."""

    service_port_range: str | None = None
    """Optional host/container port range for services that expose their own ports, e.g. `4510-4559`."""

    mount_docker_socket: bool = False
    """If True, mount `/var/run/docker.sock` into the managed container for Docker-backed services like Lambda."""

    container_name: str | None = None
    """Optional name for the managed container. Leave None to let Docker assign one."""

    container_env: Mapping[str, str] = field(default_factory=dict[str, str])
    """Environment variables passed to the managed container, e.g. `{'DEBUG': '1'}`."""

    docker_path: str = 'docker'
    """Path or name of the Docker executable used to manage the container."""

    startup_timeout: float = 120.0
    """Seconds to wait for the managed container to become ready before failing."""

    include_instructions: bool = True
    """If True, add instructions telling the model how to use the emulated environment."""

    def get_instructions(self) -> str | None:
        """Explain the emulated environment to the model, unless disabled."""
        if not self.include_instructions:
            return None
        return _INSTRUCTIONS.format(endpoint_url=self.endpoint_url)

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        """Build and return the LocalStack toolset."""
        return LocalStackToolset[AgentDepsT](
            endpoint_url=self.endpoint_url,
            region=self.region,
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
            allowed_services=self.allowed_services,
            denied_services=self.denied_services,
            default_timeout=self.default_timeout,
            max_output_chars=self.max_output_chars,
            aws_cli_path=self.aws_cli_path,
            manage_container=self.manage_container,
            image=self.image,
            host_address=self.host_address,
            service_port_range=self.service_port_range,
            mount_docker_socket=self.mount_docker_socket,
            container_name=self.container_name,
            container_env=self.container_env,
            docker_path=self.docker_path,
            startup_timeout=self.startup_timeout,
        )
