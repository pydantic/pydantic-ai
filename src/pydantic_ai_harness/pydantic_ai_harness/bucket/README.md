# Bucket

Give your agent a cloud storage bucket (Amazon S3 and S3-compatible services such as Cloudflare R2 and MinIO, Google Cloud Storage, or Azure Blob Storage) as a folder in its workspace, or as the whole workspace.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/bucket/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[bucket,anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[bucket,anthropic]"
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

- A bucket is not a filesystem: there are no symlinks or permissions, and when two agents write the same file, the last write wins.
- A folder exists while it holds a file. An empty folder made by the agent holds a hidden `.pydantic-ai-dir` file.
- Every file operation is a request to the storage service, so searching a large folder through file tools is slow.
