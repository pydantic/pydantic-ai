# LocalStack

Give an agent access to an emulated AWS environment, so it can provision and
exercise AWS services without touching a real account.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/localstack/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## The problem

Agents that build or test cloud infrastructure need somewhere to create buckets,
tables, queues, and functions. Real AWS costs money, puts real credentials
within the agent's reach, and is hard to reset between runs.
[LocalStack](https://www.localstack.cloud/) emulates AWS locally; this
capability handles the wiring: the endpoint, dummy credentials, the AWS CLI, and
a health check.

## The solution

`LocalStack` exposes AWS tooling wired to a running LocalStack instance. The
agent issues plain AWS CLI commands; the capability injects the endpoint,
region, and credentials, and adds a health check for the emulated services.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import LocalStack

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[LocalStack()],
)

result = agent.run_sync('Create an S3 bucket called reports and list all buckets.')
print(result.output)
```

By default the agent connects to a LocalStack instance you started separately,
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

Or let the capability start and stop a Docker container for you:

```python
from pydantic_ai_harness import LocalStack

LocalStack(manage_container=True)
```

## Tools

| Tool | Purpose |
|---|---|
| `aws_cli` | Run an AWS CLI command against LocalStack. Pass the command **without** the leading `aws` and **without** `--endpoint-url`; both are injected. Returns labelled stdout/stderr plus an exit code on failure. |
| `localstack_health` | Query LocalStack's health endpoint and return the JSON of which services (s3, dynamodb, sqs, etc.) are available. |

Commands run as an argument vector (no shell), so shell operators and
redirection in the command string have no effect. Each command runs with stdin
closed and with its own empty AWS config and credentials files in place of
`~/.aws/config` and `~/.aws/credentials`, so `aws configure` reads and writes
those throwaway files and its settings last for that one command. Output is
labelled with `[stdout]` / `[stderr]` markers and an `[exit code: N]` line on
non-zero exit. When it exceeds `max_output_chars` the **tail** is kept (the head
is dropped), so errors survive truncation. The cap applies to the command's
result itself, so a durable execution record of it is bounded too.

`aws_cli` takes an optional `timeout_seconds` that overrides `default_timeout`
for one command. It must be a finite number greater than zero. To bound how long
the model can make any tool call run, set the agent's `tool_timeout`.

The AWS CLI can read from and write to local files through arguments such as
`--body`, `file://`, `fileb://`, `s3 cp`, and similar service commands. Treat
this capability as both AWS-emulator access and AWS CLI access to the process's
filesystem.

## Service controls

| Field | Effect |
|---|---|
| `allowed_services` | If non-empty, only these AWS services may be used (allowlist), e.g. `['s3', 'dynamodb']`. |
| `denied_services` | These AWS services are always rejected (denylist). |

`allowed_services` and `denied_services` are mutually exclusive: set one, not
both. The service is the first token that is neither a global option nor its
value (`s3` in `--output text s3 ls`). Global options are recognized by any
prefix the AWS CLI accepts, so `--out text s3 ls` names `s3` too. Commands that
pass `--endpoint-url`, `--profile`, `--region`, or `--no-sign-request`, or any
prefix of them, are rejected, since the capability injects the target and
credentials.

> **These checks are best-effort, not a security boundary.** They gate which
> commands the agent issues, not what it can reach. For hard guarantees,
> configure LocalStack itself with the narrowest service and IAM behavior the
> run needs, and run the agent under OS-level isolation.

## Managing the container

Set `manage_container=True` and the capability starts a LocalStack Docker
container for each run and stops it when the run ends, so each run gets a fresh
environment; runs inside `async with agent:` share one instead (see below).
Docker must be installed and running.

```python
from pydantic_ai_harness import LocalStack

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
gets its own container on the same host port, so concurrent runs of one agent
need to share one: inside `async with agent:`, entering the agent starts one
container that the runs inside share, and exiting stops it. Alternatively, point
them at an externally managed instance (`manage_container=False`).

Managed containers bind to `127.0.0.1` by default, reached through
`localhost.localstack.cloud`. Set `host_address` to publish on a different
address; any value other than `127.0.0.1` or `0.0.0.0` is then also the host of
the tool endpoint and of the endpoint the instructions name, except that the
IPv6 bind-all `::` is reached at `::1`. Scoped IPv6 addresses such as
`fe80::1%eth0` are rejected.

Since LocalStack 2026.03.0, `localstack/localstack` needs an auth token to start
(a free Hobby token covers community usage). `LOCALSTACK_AUTH_TOKEN`, or the
legacy `LOCALSTACK_API_KEY` when no token is set, is forwarded from the current
process through the Docker CLI environment, never the `docker run` arguments,
unless `container_env` already sets one of them. To run without a token, pin
`image` to a `localstack/localstack:4.x` tag.

Some AWS services expose ports outside the gateway. LocalStack reserves
`4510-4559` for those service endpoints, and Docker-backed services such as
Lambda need the Docker socket mounted. Enable those explicitly when the services
you test require them:

```python
from pydantic_ai_harness import LocalStack

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
import asyncio

from pydantic_ai_harness.localstack import LocalStackContainer


async def main() -> None:
    async with LocalStackContainer(environment={'DEBUG': '1'}) as localstack:
        ...  # talk to localstack.endpoint_url


asyncio.run(main())
```

## Durable execution

Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/), each AWS CLI command and health check
against an external LocalStack instance is recorded, so a recovered run reuses
its output instead of running the command again. Temporal and Prefect record
each tool call in its own activity or task. DBOS runs function tools in
workflow code, so there the command runs as its own step.

A managed container (`manage_container=True`) starts empty in the process that
recovers a run, so on DBOS `LocalStack` does not record its commands: recovery
runs them again, which rebuilds the container's state. Temporal and Prefect
record every tool call, so there recovery replays the recorded output and the
new container lacks the resources the earlier commands created. Point
`endpoint_url` at an external LocalStack instance for durable runs on those
engines.

Runs that share one container inside `async with agent:` also share its state.
If one of them recovers on DBOS while that container is still up, the commands
it runs again meet the resources it and its sibling runs already created, so
commands that are not idempotent, such as `s3 mb` or `sqs create-queue`, can
fail. For durable runs, use an external instance, or keep commands idempotent.

The records are named after the capability's `id`, which defaults to
`localstack`, so durable execution needs no configuration.

## Telemetry

`LocalStack` emits no spans of its own. Core's
[instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) already records each
`aws_cli` and `localstack_health` call as a tool span. Starting and stopping a
managed container makes no decision worth a span. When Docker is missing or
fails, or LocalStack never becomes ready, the start raises `LocalStackError`.

## Configuration

```python
from pydantic_ai_harness import LocalStack

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
    manage_container=False,                # start/stop a Docker container per run or agent context
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

## Agent spec (YAML/JSON)

`LocalStack` works with Pydantic AI's
[agent spec](https://pydantic.dev/docs/ai/core-concepts/agent-spec/):

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
from pydantic_ai_harness import LocalStack

agent = Agent.from_file('agent.yaml', custom_capability_types=[LocalStack])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate
`LocalStack`.

## Further reading

- [LocalStack documentation](https://docs.localstack.cloud/)
- [Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/)
- [Toolsets](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/)
