---
title: Cloud Storage
description: "Give a Pydantic AI agent a cloud storage bucket (S3, R2, Google Cloud Storage, Azure Blob) as its workspace, or as a folder in one."
---

# Cloud Storage

Give your agent a cloud storage bucket (Amazon S3 and S3-compatible services such as Cloudflare R2 and MinIO, Google Cloud Storage, or Azure Blob Storage) as its workspace, or as a folder in a sandbox.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/cloud_storage/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
pip/uv-add "pydantic-ai-harness[cloud-storage,anthropic]"
```

The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra.

## Quick start

A bucket is a filesystem: pass it as the run's workspace, and the agent's file tools work in it. There is no shell, so shell tools aren't offered.

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness import FileSystem
from pydantic_ai_harness.cloud_storage import S3Filesystem

agent = Agent('{{ anthropic:large }}', capabilities=[FileSystem()])
exports = S3Filesystem('s3://acme-data/exports/')


async def main():
    result = await agent.run('Summarise the latest export', workspace=exports)
    print(result.output)
```

The file tools read `2026-10-01.csv` from `s3://acme-data/exports/2026-10-01.csv`. Pass `workspace=ReadOnlyWorkspace(exports)` (from `pydantic_ai.workspaces`) to refuse changes.

## As a folder

A [`Mount`](https://pydantic.dev/docs/ai/core-concepts/workspace/#mounting-filesystems) shows the bucket as a folder of another workspace. `LocalWorkspace` and every sandbox capability take `mounts=`, a list or a function of the run, for example one folder per customer:

```python {names="defined"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.workspaces import Mount
from pydantic_ai_harness.cloud_storage import S3Filesystem
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox


@dataclass
class Customer:
    id: str


def customer_files(ctx: RunContext[Customer]) -> list[Mount]:
    files = S3Filesystem(f's3://acme-customers/{ctx.deps.id}/')
    return [Mount('files', files, read_only=True, description="The customer's files")]


agent = Agent('{{ anthropic:large }}', deps_type=Customer, capabilities=[E2BSandbox(mounts=customer_files), Coder()])
```

File tools reach `files/`; commands don't, and the model is told so.

## Commands

With `Mount('files', files, commands=True)`, the sandbox mounts the bucket with [rclone](https://rclone.org), so commands see the same files as an ordinary folder: `grep -r`, `pandas`, `git` and editors all work there.

That needs:

- a sandbox that allows FUSE: [E2B](e2b-sandbox.md), [Sprites](sprites-sandbox.md), [Modal](modal-sandbox.md) with `runtime='vm'`, and [Docker](docker-sandbox.md) when its `docker_args` give the container `/dev/fuse`;
- Python 3 in the image; `fuse3` (with apt) and rclone are installed on first use if the image lacks them, which needs internet access;
- an `S3Filesystem` or `AzureFilesystem` with `credentials=` that expire. Commands in the sandbox can read the keys a mount uses, so only keys with an `expires_at` are handed over, given directly or returned by a function, which is called again before they expire. Make them keys limited to the folder, so a command that reads them can't reach anything else: temporary AWS keys with a session policy, or an Azure SAS token for the container (`{'sas_token': ..., 'expires_at': ...}`). An Azure mount is mounted again to take a new SAS token, so a command still using the folder at that moment fails.

When the sandbox or the keys can't work, the run fails before it starts and says why; a missing Python 3, or no internet to install rclone, shows at the first command. With AWS, a function like this one hands out keys that last 15 minutes and reach only `exports/`:

```python {names="defined"}
import json

import boto3

from pydantic_ai_harness.cloud_storage import S3Filesystem

sts = boto3.client('sts')


def exports_keys():
    folder = {'Effect': 'Allow', 'Action': ['s3:GetObject', 's3:PutObject', 's3:DeleteObject'], 'Resource': 'arn:aws:s3:::acme-data/exports/*'}
    listing = {
        'Effect': 'Allow',
        'Action': 's3:ListBucket',
        'Resource': 'arn:aws:s3:::acme-data',
        'Condition': {'StringLike': {'s3:prefix': ['exports', 'exports/*']}},
    }
    policy = json.dumps({'Version': '2012-10-17', 'Statement': [folder, listing]})
    keys = sts.assume_role(
        RoleArn='arn:aws:iam::123456789012:role/agent-buckets', RoleSessionName='agent', Policy=policy, DurationSeconds=900
    )['Credentials']
    return {
        'access_key_id': keys['AccessKeyId'],
        'secret_access_key': keys['SecretAccessKey'],
        'token': keys['SessionToken'],
        'expires_at': keys['Expiration'],
    }


exports = S3Filesystem('s3://acme-data/exports/', credentials=exports_keys)
```

A function whose keys turn out not to expire is only found when the first command runs. What commands write reaches the bucket a few seconds after the file is closed, so a sandbox destroyed right after a run can lose its last writes. If the sandbox restarts while files are still being sent, they are sent when the bucket is mounted again. Files written into the folder while it wasn't mounted, such as after a restart, are moved to `.pydantic-ai-recovered/` next to it when it is mounted again. A folder that already holds files in the sandbox is refused. A sandbox has one set of mounts at a time: a later run with other mounts replaces them, and a run at the same time with other mounts is refused, so give concurrent runs their own sandboxes.

## Choosing the class

| Storage | Filesystem |
|---|---|
| Amazon S3 | `S3Filesystem('s3://<bucket>/<folder>/')` |
| Cloudflare R2, MinIO and other S3-compatible services | `S3Filesystem('s3://<bucket>/', endpoint='<service URL>', region='auto')` |
| Google Cloud Storage | `GCSFilesystem('gs://<bucket>/<folder>/')` |
| Azure Blob Storage | `AzureFilesystem('https://<account>.blob.core.windows.net/<container>/<folder>/')` |

The folder is optional; without it the agent sees the whole bucket. Google Cloud Storage can't be mounted for commands.

## Credentials

By default, a filesystem finds keys the way the cloud's own tools do: environment variables, a shared config file, or the role of the machine it runs on. To pass keys yourself, give `credentials=` the keys, or a function that returns them, which is called again before they expire:

```python {names="defined"}
import os

from pydantic_ai_harness.cloud_storage import S3Filesystem

data = S3Filesystem(
    's3://acme-data/',
    credentials={
        'access_key_id': os.environ['ACME_KEY_ID'],
        'secret_access_key': os.environ['ACME_SECRET'],
        'token': None,
        'expires_at': None,
    },
)
```

The keys have the shape [obstore](https://developmentseed.org/obstore/latest/authentication/) uses for each cloud.

## Limits

- A bucket is not a disk: through file tools there are no symlinks or permissions, and when two agents write the same file, the last write wins.
- A folder exists while it holds a file. An empty folder made by the agent holds a hidden `.pydantic-ai-dir` file.
- Every file operation is a request to the storage service, so searching a large folder through file tools is slow.
- Under [durable execution](https://pydantic.dev/docs/ai/core-concepts/workspace/#durable-execution), a run rebuilds its workspace from the agent's capabilities, so mount the bucket on a sandbox or `LocalWorkspace` there rather than passing it as `workspace=`.
