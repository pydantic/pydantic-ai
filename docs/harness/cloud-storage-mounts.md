---
title: Mounting Cloud Storage
description: "Show an S3 or S3-compatible bucket as a folder in a Pydantic AI agent's sandbox, so its commands and file tools work on the bucket's files."
---

# Mounting Cloud Storage

Show an S3 bucket, or a bucket on an S3-compatible store such as Cloudflare R2 or MinIO, as a folder in the agent's sandbox. The agent's shell commands and file tools then read and write the bucket's files like any others, with no storage SDK or keys in your tools.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Mount a bucket

Pass the sandbox capability `mounts=`, a list of `S3Mount`s:

```python {names="defined"}
import os
from datetime import UTC, datetime, timedelta

from pydantic_ai import Agent
from pydantic_ai_harness import S3Mount
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.docker_sandbox import DockerSandbox

FUSE = ['--device', '/dev/fuse', '--cap-add', 'SYS_ADMIN', '--security-opt', 'apparmor:unconfined']

reports = S3Mount(
    path='reports',
    url='s3://acme-reports/2026/',
    access_key_id=os.environ['REPORTS_ACCESS_KEY_ID'],
    secret_access_key=os.environ['REPORTS_SECRET_ACCESS_KEY'],
    session_token=os.environ['REPORTS_SESSION_TOKEN'],
    expires_at=datetime.now(UTC) + timedelta(hours=1),
    region='eu-west-1',
)
agent = Agent(
    '{{ anthropic:large }}',
    capabilities=[DockerSandbox(image='python:3.13-slim', docker_args=FUSE, mounts=[reports]), Coder()],
)
result = agent.run_sync('Summarize the CSV files in reports/ into reports/summary.md.')
```

`reports/` in the sandbox's working directory now shows the `2026/` prefix of the `acme-reports` bucket; `url='s3://acme-reports/'` shows the whole bucket. The folder is created if it's missing. Each mount needs a folder of its own: two mounts at the same folder, or one inside another, are refused with `MountError`.

`expires_at` is required, and is when the keys stop working. A plain list's keys are used until then; an operation after that raises `MountError`. For runs that may outlast the keys, pass a function instead.

## Fresh keys for each run

`mounts=` can be a function of the run, returning the mounts with fresh keys. This one gives each customer temporary keys from [AWS STS](https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html) for their own prefix only:

```python {names="defined"}
import json
from collections.abc import Sequence
from dataclasses import dataclass

import boto3

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness import S3Mount
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.docker_sandbox import DockerSandbox

BUCKET = 'acme-customer-data'
ROLE_ARN = 'arn:aws:iam::123456789012:role/agent-bucket-access'
FUSE = ['--device', '/dev/fuse', '--cap-add', 'SYS_ADMIN', '--security-opt', 'apparmor:unconfined']


@dataclass
class Customer:
    id: str


def customer_files(ctx: RunContext[Customer]) -> Sequence[S3Mount]:
    prefix = f'customers/{ctx.deps.id}/'
    # The role's own policy allows the bucket; this one narrows the keys to the customer's prefix.
    policy = {
        'Version': '2012-10-17',
        'Statement': [
            {
                'Effect': 'Allow',
                'Action': ['s3:GetObject', 's3:PutObject', 's3:DeleteObject'],
                'Resource': f'arn:aws:s3:::{BUCKET}/{prefix}*',
            },
            {
                'Effect': 'Allow',
                'Action': 's3:ListBucket',
                'Resource': f'arn:aws:s3:::{BUCKET}',
                'Condition': {'StringLike': {'s3:prefix': f'{prefix}*'}},
            },
        ],
    }
    credentials = boto3.client('sts').assume_role(
        RoleArn=ROLE_ARN,
        RoleSessionName=f'agent-{ctx.deps.id}',
        Policy=json.dumps(policy),
        DurationSeconds=3600,
    )['Credentials']
    return [
        S3Mount(
            path='files',
            url=f's3://{BUCKET}/{prefix}',
            access_key_id=credentials['AccessKeyId'],
            secret_access_key=credentials['SecretAccessKey'],
            session_token=credentials['SessionToken'],
            expires_at=credentials['Expiration'],
        )
    ]


agent = Agent(
    '{{ anthropic:large }}',
    deps_type=Customer,
    capabilities=[DockerSandbox(image='python:3.13-slim', docker_args=FUSE, mounts=customer_files), Coder()],
)
result = agent.run_sync('Tidy up the files in files/.', deps=Customer(id='c-1042'))
```

The function is called from a worker thread each time the sandbox needs keys: before the run's first command or file operation, and before the first operation in the five minutes before the keys expire. Building the run's workspace doesn't call it, so a slow call to a key service delays only the first operation. Each call must return the same mounts, by path, bucket, endpoint, region and `read_only`, with only the keys changing; otherwise the operation raises `MountError`.

## Cloudflare R2, MinIO and other S3-compatible stores

Set `endpoint` to the store's S3 API, and `region` if it needs one:

```python {names="defined"}
import os
from datetime import UTC, datetime, timedelta

from pydantic_ai_harness import S3Mount

r2 = S3Mount(
    path='assets',
    url='s3://assets/',
    access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    expires_at=datetime.now(UTC) + timedelta(hours=1),
    endpoint=f'https://{os.environ["R2_ACCOUNT_ID"]}.r2.cloudflarestorage.com',
    region='auto',
)
```

Without `endpoint`, the mount is on AWS S3.

## Read-only mounts

Pass `read_only=True` to refuse writes in the folder, whatever the keys allow:

```python {names="defined"}
import os
from datetime import UTC, datetime, timedelta

from pydantic_ai_harness import S3Mount

datasets = S3Mount(
    path='datasets',
    url='s3://acme-datasets/',
    access_key_id=os.environ['DATASETS_ACCESS_KEY_ID'],
    secret_access_key=os.environ['DATASETS_SECRET_ACCESS_KEY'],
    session_token=os.environ['DATASETS_SESSION_TOKEN'],
    expires_at=datetime.now(UTC) + timedelta(hours=1),
    read_only=True,
)
```

A command that writes there fails with `Read-only file system`. Keys that can only read are the stronger guard, since code in the sandbox can read the keys and use them directly.

## Lifecycle

A mount belongs to the sandbox, not the run. The run's first command or file operation mounts it, and a later run on the same sandbox with the same mount reuses it and gives it the new keys. The mount stays after the run with the keys it last received, and stops working when they expire, until a run gives it fresh ones.

A different bucket at the same folder is refused while the keys of the one there are valid; after they expire, the new one replaces it. Files written into the folder while nothing was mounted there are moved to `.pydantic-ai-recovered/` next to it, with a warning.

To clear mounts, destroy the sandbox. In a sandbox you keep, run `fusermount3 -u <folder>` as root.

## How it works

Before the run's first command or file operation, the sandbox runs a helper script as root. It installs FUSE and [rclone](https://rclone.org/) if the image lacks them, starts an rclone daemon, and mounts each bucket at its folder. Later calls only send the daemon new keys. The keys reach the helper on its standard input, never on a command line or in a log.

The image needs `python3`. It needs `fusermount3`, or `apt-get` to install `fuse3`, and rclone v1.75.2 or later, or access to `downloads.rclone.org` to download that version (checked against its published checksum). Install both in your image to skip the first run's install.

Files belong to the owner of the working directory. Writes are cached on the sandbox's disk and uploaded a few seconds after the file is closed, so appends, in-place edits and renames work; a change made to the bucket from outside shows in the folder within about ten seconds.

## Sandboxes

| Sandbox | What it needs |
| --- | --- |
| [Docker Sandbox](docker-sandbox.md) | The FUSE device, added when the container is created: `docker_args=['--device', '/dev/fuse', '--cap-add', 'SYS_ADMIN', '--security-opt', 'apparmor:unconfined']`. A container without it is refused with `MountError` naming these flags. The container must reach the store, so leave `allow_internet_access` on; to limit where it connects, pass a `--network` of your own in `docker_args`. |

The other sandboxes don't support `mounts=` yet.

## Security

Code in the sandbox can read the keys, so pass only short-lived keys limited to that bucket, or to the prefix you mount, as in the STS example above. `S3Mount` keeps the keys out of its `repr` and out of the sandbox's durable policy, and `MountError` messages leave them out.

The keys keep working after the run, through the mount and for anything else code in the sandbox does with them, until they expire. `read_only=True` only refuses writes through the folder.

In Docker, `--cap-add SYS_ADMIN` and `apparmor:unconfined` give every command in the container more access to the kernel it shares with your machine, which makes it a weaker boundary, so add these flags only to containers that use mounts.

The helper runs as root and makes the mount folder inside the working directory, which the agent's user owns. It doesn't follow links that user puts in the folder, but a command running at the same moment as a mount can still swap the folder for a link to elsewhere before rclone mounts on it. That matters only when commands run as a user that can't already become root, as in a Docker image with a non-root `USER`: there, treat the agent as able to reach root.

## Telemetry

Mounting emits no spans or attributes of its own.

## API reference

::: pydantic_ai_harness.S3Mount

::: pydantic_ai_harness.MountError
