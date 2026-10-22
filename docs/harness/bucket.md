---
title: Bucket
description: "Give a Pydantic AI agent a cloud storage bucket (S3, R2, Google Cloud Storage, Azure Blob) as a folder in its workspace, or as the whole workspace."
---

# Bucket

Give your agent a cloud storage bucket (Amazon S3 and S3-compatible services such as Cloudflare R2 and MinIO, Google Cloud Storage, or Azure Blob Storage) as a folder in its workspace, or as the whole workspace.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/bucket/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
pip/uv-add "pydantic-ai-harness[bucket,anthropic]"
```

The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra.

## Quick start

Mount a bucket as a folder of the workspace:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.workspaces import Mount
from pydantic_ai_harness import FileSystem
from pydantic_ai_harness.bucket import Bucket

exports = Bucket('s3://acme-data/exports/')
agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[
        LocalWorkspace('.', mounts=[Mount('exports', exports, read_only=True, description='Daily CSV exports')]),
        FileSystem(),
    ],
)
```

The agent's file tools read `exports/2026-10-01.csv` from `s3://acme-data/exports/2026-10-01.csv`. Commands on your machine don't see the folder.

## In a sandbox

`LocalWorkspace` and every sandbox capability take `mounts=`. Pass a function of the run instead of a list to pick the mounts per run, for example one folder per customer:

```python {names="defined"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.workspaces import Mount
from pydantic_ai_harness.bucket import Bucket
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox


@dataclass
class Customer:
    id: str


def customer_files(ctx: RunContext[Customer]) -> list[Mount]:
    return [Mount('files', Bucket(f's3://acme-customers/{ctx.deps.id}/'), description="The customer's files")]


agent = Agent('anthropic:claude-opus-5-5', deps_type=Customer, capabilities=[E2BSandbox(mounts=customer_files), Coder()])
```

## Shell commands

Commands in a sandbox see a mounted bucket too, as an ordinary folder, when the sandbox can mount it: `grep -r`, `pandas`, `git` and editors all work there. The model is told which folders commands see.

A sandbox mounts an S3 or Azure bucket with [rclone](https://rclone.org) when:

- its image has Python 3; `fuse3` (with apt) and rclone are installed on first use if the image lacks them, which needs internet access;
- the sandbox allows FUSE: [E2B](e2b-sandbox.md), [Sprites](sprites-sandbox.md), [Modal](modal-sandbox.md) with `runtime='vm'`, and [Docker](docker-sandbox.md) when its `docker_args` give the container `/dev/fuse`;
- the bucket's keys expire: commands in the sandbox can read the keys a mount uses, so a bucket hands over only keys with an `expires_at`, given directly or returned by a function, which is called again before they expire. Make them keys limited to the bucket's folder, so a command that reads them can't reach anything else. Azure mounts use the storage account key, so they also need `share_long_lived_credentials=True` on the bucket.

Otherwise, and for Google Cloud Storage buckets, the folder is reachable through file tools only. With AWS, a function like this one hands out keys that last 15 minutes and reach only `exports/`:

```python {names="defined"}
import json

import boto3

from pydantic_ai_harness.bucket import Bucket

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


exports = Bucket('s3://acme-data/exports/', credentials=exports_keys)
```

What commands write reaches the bucket a few seconds after the file is closed, so a sandbox destroyed right after a run can lose its last writes. If the sandbox restarts while files are still being sent, they are sent when the bucket is mounted again. Files written into the folder while it wasn't mounted, such as after a restart, are moved to `.pydantic-ai-recovered/` next to it when it is mounted again. A sandbox has one set of mounts at a time: a run with other mounts replaces them.

## Choosing the bucket

| Storage | URL |
|---|---|
| Amazon S3 | `Bucket('s3://<bucket>/<folder>/')` |
| Cloudflare R2, MinIO and other S3-compatible services | `Bucket('s3://<bucket>/', endpoint='<service URL>', region='auto')` |
| Google Cloud Storage | `Bucket('gs://<bucket>/<folder>/')` |
| Azure Blob Storage | `Bucket('https://<account>.blob.core.windows.net/<container>/<folder>/')` |

The folder is optional; without it the agent sees the whole bucket.

## Credentials

By default, `Bucket` finds keys the way the cloud's own tools do: environment variables, a shared config file, or the role of the machine it runs on. To pass keys yourself, give `credentials=` the keys, or a function that returns them, which is called again before they expire:

```python {names="defined"}
import os

from pydantic_ai_harness.bucket import Bucket

bucket = Bucket(
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

## The bucket as the workspace

`BucketWorkspace` makes the bucket the whole workspace. File tools work in it; there are no commands:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness import FileSystem
from pydantic_ai_harness.bucket import Bucket, BucketWorkspace

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[BucketWorkspace(Bucket('gs://acme-docs/handbook/'), read_only=True), FileSystem()],
)
```

## Limits

- A bucket is not a filesystem: through file tools there are no symlinks or permissions, and when two agents write the same file, the last write wins.
- A folder exists while it holds a file. An empty folder made by the agent holds a hidden `.pydantic-ai-dir` file.
- Every file operation is a request to the storage service, so searching a large folder through file tools is slow.
