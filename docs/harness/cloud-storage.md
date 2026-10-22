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

The file tools read `2026-10-01.csv` from `s3://acme-data/exports/2026-10-01.csv`. Pass `workspace=ReadOnlyWorkspace(Workspace(exports))` (both from `pydantic_ai.workspaces`) to refuse changes.

## As a folder

A [`Mount`](https://pydantic.dev/docs/ai/core-concepts/workspace/#mounting-filesystems) shows the bucket as a folder of another workspace, such as `LocalWorkspace`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.workspaces import Mount
from pydantic_ai_harness import FileSystem
from pydantic_ai_harness.cloud_storage import S3Filesystem

exports = S3Filesystem('s3://acme-data/exports/')
agent = Agent(
    '{{ anthropic:large }}',
    capabilities=[LocalWorkspace('.', mounts=[Mount('exports', exports, read_only=True)]), FileSystem()],
)
```

File tools reach `exports/`; commands on your machine don't.

## Choosing the class

| Storage | Filesystem |
|---|---|
| Amazon S3 | `S3Filesystem('s3://<bucket>/<folder>/')` |
| Cloudflare R2, MinIO and other S3-compatible services | `S3Filesystem('s3://<bucket>/', endpoint='<service URL>', region='auto')` |
| Google Cloud Storage | `GCSFilesystem('gs://<bucket>/<folder>/')` |
| Azure Blob Storage | `AzureFilesystem('https://<account>.blob.core.windows.net/<container>/<folder>/')` |

The folder is optional; without it the agent sees the whole bucket.

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
