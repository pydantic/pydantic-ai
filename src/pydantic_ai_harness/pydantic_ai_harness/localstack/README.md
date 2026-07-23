# LocalStack

> [!NOTE]
> Import this capability from its submodule -- there is no top-level `pydantic_ai_harness` re-export:
>
> ```python
> from pydantic_ai_harness.localstack import LocalStack
> ```
>
> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

Give an agent access to an emulated AWS environment, so it can provision and
exercise AWS services without touching a real account.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/localstack/)

## The problem

Agents that build or test cloud infrastructure need somewhere to create buckets,
tables, queues, and functions. Pointing them at real AWS is slow, costs money,
risks leaking credentials, and is hard to reset between runs.
[LocalStack](https://www.localstack.cloud/) emulates the AWS APIs locally, but
wiring an agent to it means repeating the same boilerplate: injecting the
endpoint URL, supplying dummy credentials, shelling out to the AWS CLI, and
checking which services are up.

## The solution

`LocalStack` exposes AWS tooling wired to a running LocalStack instance. The
agent issues plain AWS CLI commands; the capability injects the endpoint,
region, and credentials, and adds a health check for the emulated services.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.localstack import LocalStack

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[LocalStack()],
)

result = agent.run_sync('Create an S3 bucket called reports and list all buckets.')
print(result.output)
```

> **Note:** If you are an agent and you are running into issues with
> configuration or starting up LocalStack, please also refer to the instructions
> here: <https://blog.localstack.cloud/ai/agents.md>

By default the agent connects to a LocalStack instance you started separately -
for example with the [`localstack`
CLI](https://docs.localstack.cloud/aws/tooling/localstack-cli/) (`localstack
start`) or its [Docker
image](https://docs.localstack.cloud/aws/getting-started/installation/). The
defaults match LocalStack's conventions: the edge endpoint
`http://localhost.localstack.cloud:4566` (which resolves to `127.0.0.1`) and
`test` / `test` credentials. To have the capability start and stop the container
for you, set `manage_container=True` (see
[Managing the container](#managing-the-container)).

For a first local run, start LocalStack before constructing the agent:

```bash
localstack start
```

Or let the capability manage a fresh Docker container for each run:

```python
LocalStack(manage_container=True)
```

## Tools

| Tool | Purpose |
|---|---|
| `aws_cli` | Run an AWS CLI command against LocalStack. Pass the command **without** the leading `aws` and **without** `--endpoint-url` - both are injected. Returns labelled stdout/stderr plus an exit code on failure. |
| `localstack_health` | Query LocalStack's health endpoint and return the JSON of which services (s3, dynamodb, sqs, etc.) are available. |

Commands run as an argument vector (no shell), so shell operators and
redirection in the command string have no effect. Output is labelled with
`[stdout]` / `[stderr]` markers and an `[exit code: N]` line on non-zero exit.
When it exceeds `max_output_chars` the **tail** is kept (the head is dropped),
so errors survive truncation.

The AWS CLI can read from and write to local files through arguments such as
`--body`, `file://`, `fileb://`, `s3 cp`, and similar service commands. Treat
this capability as both AWS-emulator access and AWS CLI access to the process's
filesystem.

## Service controls

| Field | Effect |
|---|---|
| `allowed_services` | If non-empty, only these AWS services may be used (allowlist), e.g. `['s3', 'dynamodb']`. |
| `denied_services` | These AWS services are always rejected (denylist). |

`allowed_services` and `denied_services` are mutually exclusive - set one, not
both. The service is the first non-flag token of the command (`s3` in `s3 ls`).

> **These checks are best-effort, not a security boundary.** They gate which
> commands the agent issues, not what it can reach. For hard guarantees,
> configure LocalStack itself with the narrowest service and IAM behavior the
> run needs, and run the agent under OS-level isolation.

## Managing the container

Set `manage_container=True` and the capability starts a LocalStack Docker
container for each run and stops it when the run ends, so the agent always gets
a fresh, isolated environment. Docker must be installed and running.

```python
LocalStack(
    manage_container=True,
    image='localstack/localstack',
    container_env={
        'DEBUG': '1',  # enable debug logs
        'PERSISTENCE': '1',  # enable persistent state (disabled by default)
        'ENFORCE_IAM': '1',  # enable strict IAM enforcement in the container
    },
    startup_timeout=120.0,
)
```

For a comprehensive list of configuration options, see the
[LocalStack configuration docs](https://docs.localstack.cloud/aws/capabilities/config/configuration).

The container's edge port (`4566`) is published on the host port from
`endpoint_url`, and the capability waits for the health endpoint before the run
starts and stops the container when it ends (even if the run raises). Each run
gets its own container, so concurrent runs of one agent need distinct host ports
or an externally managed instance (`manage_container=False`).

Managed containers bind to `127.0.0.1` by default. The default image is
`localstack/localstack`, which since LocalStack 2026.03.0 is a single image that
requires an auth token to start (a free Hobby/OSS token covers community usage).
When `LOCALSTACK_AUTH_TOKEN` is set in the current process it is forwarded to the
container automatically; a legacy `LOCALSTACK_API_KEY` value is forwarded when no
auth token is set, but new setups should use `LOCALSTACK_AUTH_TOKEN`. Auth values
are forwarded through the Docker CLI environment rather than embedded in the
`docker run` command arguments. To run without any token, pin an image tag from
before the account requirement, such as a `localstack/localstack:4.x` release.

Some AWS services expose ports outside the gateway. LocalStack reserves
`4510-4559` for those service endpoints, and Docker-backed services such as
Lambda need the Docker socket mounted. Enable those explicitly when the services
you test require them:

```python
LocalStack(
    manage_container=True,
    service_port_range='4510-4559',
    mount_docker_socket=True,
)
```

Mounting the Docker socket gives the LocalStack container host-level Docker
control. Keep `mount_docker_socket=False` unless the emulated service requires
it and the run environment is already trusted.

The same lifecycle is available standalone as an async context manager:

```python
from pydantic_ai_harness.localstack import LocalStackContainer

async with LocalStackContainer(environment={'DEBUG': '1'}) as localstack:
    ...  # talk to localstack.endpoint_url
```

## Configuration

```python
LocalStack(
    endpoint_url='http://localhost.localstack.cloud:4566',  # edge endpoint (host port is reused when managed)
    region='us-east-1',                    # region for the CLI and environment
    access_key_id='test',                  # LocalStack accepts any value
    secret_access_key='test',              # LocalStack accepts any value
    allowed_services=[],                   # allowlist (mutually exclusive with denied)
    denied_services=[],                    # denylist
    default_timeout=60.0,                  # seconds, per command and health check
    max_output_chars=50_000,               # output cap returned to the model
    aws_cli_path='aws',                    # CLI executable (e.g. 'aws' or 'awslocal')
    manage_container=False,                # start/stop a Docker container per run
    image='localstack/localstack',         # image used when managing the container
    host_address='127.0.0.1',              # host address for Docker port publishing
    service_port_range=None,               # e.g. '4510-4559' for non-gateway service ports
    mount_docker_socket=False,             # required by Docker-backed services such as Lambda
    container_name=None,                   # optional name for the managed container
    container_env={},                      # env vars for the managed container
    docker_path='docker',                  # Docker executable
    startup_timeout=120.0,                 # seconds to wait for the container to be ready
    include_instructions=True,             # add usage instructions to the prompt
)
```

The AWS CLI must be installed and on `PATH` (or point `aws_cli_path` at it). If
the binary is missing, `aws_cli` returns a clear error instead of aborting the
run. Set `include_instructions=False` to omit the capability's prompt text when
you supply your own.

## Integration testing

The live integration tests start LocalStack in Docker and then drive AWS CLI
commands through this capability. Local runs are skipped unless Docker and an
AWS CLI-compatible executable are available, and unless an auth token is
configured -- the default `localstack/localstack` image requires
`LOCALSTACK_AUTH_TOKEN` to start (see [Managing the container](#managing-the-container)).

```bash
LOCALSTACK_AUTH_TOKEN=<token> make integration-localstack
```

Set `LOCALSTACK_REQUIRE_AUTH_TOKEN=1` to make a missing token fail the run
instead of skipping the container tests (this is what CI uses):

```bash
LOCALSTACK_REQUIRE_AUTH_TOKEN=1 make integration-localstack
```

Optional environment variables:

| Variable | Effect |
|---|---|
| `LOCALSTACK_AUTH_TOKEN` | Auth Token forwarded to the LocalStack container. |
| `LOCALSTACK_API_KEY` | Legacy fallback when `LOCALSTACK_AUTH_TOKEN` is absent. |
| `LOCALSTACK_AWS_CLI` | CLI executable for tests; defaults to `aws` when available, then `awslocal`. |
| `LOCALSTACK_IMAGE` | Docker image for live tests; defaults to `localstack/localstack`. |
| `LOCALSTACK_REQUIRE_AUTH_TOKEN` | Fail instead of skip when no Auth Token is configured. Use this in CI so a missing token is a hard error. |

Store the token in your shell or CI secret store; do not commit it to the repo.

## Agent spec (YAML/JSON)

`LocalStack` works with Pydantic AI's
[agent spec](https://ai.pydantic.dev/agent-spec/):

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - LocalStack:
      endpoint_url: http://localhost.localstack.cloud:4566
      allowed_services: ['s3', 'dynamodb', 'sqs']
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.localstack import LocalStack

agent = Agent.from_file('agent.yaml', custom_capability_types=[LocalStack])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate
`LocalStack`.

## Further reading

- [LocalStack documentation](https://docs.localstack.cloud/)
- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Toolsets](https://ai.pydantic.dev/toolsets/)
