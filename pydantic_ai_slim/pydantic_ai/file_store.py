"""FileStore abstraction for binary data storage in multi-modal LLM interactions.

This module provides:
- [`FileStore`][pydantic_ai.file_store.FileStore]: Protocol for file storage backends
- [`S3FileStore`][pydantic_ai.file_store.S3FileStore]: S3-compatible file store implementation
- [`file_store_processor`][pydantic_ai.file_store.file_store_processor]: History processor factory
- [`generate_file_key`][pydantic_ai.file_store.generate_file_key]: Utility to generate storage keys

Example:
    ```python
    from pydantic_ai.file_store import S3FileStore, file_store_processor

    # Using environment variables (S3_ENDPOINT, S3_ACCESS_KEY_ID,
    # S3_SECRET_ACCESS_KEY, S3_REGION)
    store = S3FileStore(bucket='my-bucket')

    # Create a history processor for automatic file upload
    processor = file_store_processor(store)
    ```
"""

from ._file_store import (
    FileStore,
    S3Error,
    S3FileStore,
    file_store_processor,
    generate_file_key,
)

__all__ = (
    'FileStore',
    'file_store_processor',
    'generate_file_key',
    'S3FileStore',
    'S3Error',
)
