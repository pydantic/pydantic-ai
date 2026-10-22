"""S3, Google Cloud Storage and Azure Blob Storage as filesystems for workspaces.

Pass one as a run's `workspace=`, or show it as a folder with `pydantic_ai.workspaces.Mount`. Needs the
`cloud-storage` extra.
"""

from pydantic_ai_harness.cloud_storage._filesystem import AzureFilesystem, GCSFilesystem, S3Filesystem

__all__ = [
    'AzureFilesystem',
    'GCSFilesystem',
    'S3Filesystem',
]
