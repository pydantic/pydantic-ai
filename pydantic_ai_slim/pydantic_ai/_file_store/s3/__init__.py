"""S3-compatible file store implementation."""

from .s3_file_store import DEFAULT_TTL, S3FileStore
from .utils import S3Error

__all__ = ('S3FileStore', 'S3Error', 'DEFAULT_TTL')
