"""Cloud storage buckets (S3 and S3-compatible services, Google Cloud Storage, Azure Blob) for workspaces.

Mount a `Bucket` as a folder with `pydantic_ai.workspaces.Mount`, or make it the whole workspace with
`BucketWorkspace`. Needs the `bucket` extra.
"""

from pydantic_ai_harness.bucket._bucket import Bucket
from pydantic_ai_harness.bucket._capability import BucketWorkspace, BucketWorkspaceBackend

__all__ = [
    'Bucket',
    'BucketWorkspace',
    'BucketWorkspaceBackend',
]
