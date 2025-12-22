"""FileStore abstraction for binary data storage in multi-modal LLM interactions.

This module provides:
- `FileStore`: Protocol for file storage backends
- `S3FileStore`: S3-compatible file store implementation
- `generate_file_key`: Utility to generate storage keys from BinaryContent
- `file_store_processor`: History processor factory for file storage

Example:
    ```python
    from pydantic_ai.file_store import S3FileStore, file_store_processor

    # Using environment variables (S3_ENDPOINT, S3_ACCESS_KEY_ID,
    # S3_SECRET_ACCESS_KEY, S3_REGION)
    store = S3FileStore(bucket='my-bucket')

    # Create a history processor
    processor = file_store_processor(store)

    # Use with Agent
    agent = Agent('openai:gpt-4o', history_processors=[processor])
    ```
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .._run_context import RunContext
from ..messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    UserContent,
    UserPromptPart,
    VideoUrl,
)
from .s3.s3_file_store import S3FileStore
from .s3.utils import S3Error

if TYPE_CHECKING:
    from ..messages import ModelRequestPart
    from ..models import Model

__all__ = (
    'FileStore',
    'file_store_processor',
    'generate_file_key',
    'S3FileStore',
    'S3Error',
)


@runtime_checkable
class FileStore(Protocol):
    """Protocol for file storage backends that can store/retrieve binary data.

    Implementations must provide async methods for storing and retrieving data,
    as well as a sync method for generating download URIs.
    """

    async def store(self, key: str, data: bytes) -> str:
        """Store data in the file store.

        Args:
            key: Object key (path within the store)
            data: Raw bytes to store

        Returns:
            The object key that was stored
        """
        ...

    async def retrieve(self, key: str) -> bytes:
        """Retrieve data from the file store.

        Args:
            key: Object key to retrieve

        Returns:
            Raw bytes of the object
        """
        ...

    def get_download_uri(self, key: str) -> str:
        """Get a download URI for the stored object.

        Args:
            key: Object key

        Returns:
            URL for downloading the object
        """
        ...

    async def exists(self, key: str) -> bool:
        """Check if an object exists in the store.

        Args:
            key: Object key to check

        Returns:
            True if the object exists, False otherwise
        """
        ...


def generate_file_key(content: BinaryContent) -> str:
    """Generate a storage key from BinaryContent using identifier.format pattern.

    Args:
        content: The binary content to generate a key for

    Returns:
        A key in the format '{identifier}.{format}' (e.g., 'a1b2c3.png')
    """
    return f'{content.identifier}.{content.format}'


_URL_TYPE_MAP: dict[str, type[ImageUrl] | type[AudioUrl] | type[VideoUrl] | type[DocumentUrl]] = {
    'image': ImageUrl,
    'audio': AudioUrl,
    'video': VideoUrl,
    'document': DocumentUrl,
}


def _get_media_category(media_type: str) -> str | None:
    """Get the category (image, audio, video, document) from a media type string."""
    if media_type.startswith('image/'):
        return 'image'
    elif media_type.startswith('audio/'):
        return 'audio'
    elif media_type.startswith('video/'):
        return 'video'
    elif media_type.startswith('application/') or media_type.startswith('text/'):
        return 'document'
    return None


def _get_url_support(model: Model) -> dict[str, bool]:
    """Determine which URL types the model supports based on provider.

    Returns a dict like {'image': True, 'audio': False, 'video': False, 'document': False}
    indicating whether the model can handle URLs for each media category.
    """
    system = model.system.lower()

    # OpenAI: images via URL ok (unless force_download), audio/docs need bytes
    if system == 'openai':
        return {'image': True, 'audio': False, 'video': False, 'document': False}

    # Anthropic: images via URL ok
    if system == 'anthropic':
        return {'image': True, 'audio': False, 'video': False, 'document': True}

    # Google Vertex: most URLs ok
    if system == 'google-vertex':
        return {'image': True, 'audio': True, 'video': True, 'document': True}

    # Google AI (gemini/gla): generally needs bytes for external URLs
    if system in ('google-gla', 'gemini'):
        return {'image': False, 'audio': False, 'video': False, 'document': False}

    # Bedrock: depends on underlying model, default conservative
    if system == 'bedrock':
        return {'image': True, 'audio': False, 'video': False, 'document': False}

    # Default: conservative - assume bytes needed for most media types
    return {'image': False, 'audio': False, 'video': False, 'document': False}


def file_store_processor(
    store: FileStore,
) -> Callable[[RunContext[Any], list[ModelMessage]], Awaitable[list[ModelMessage]]]:
    """Create a model-aware history processor for file storage.

    This processor handles bidirectional conversion:

    **Store direction** (for serialization/DB storage):
    - Uploads BinaryContent/FilePart to the store
    - Converts to FileUrl types (ImageUrl, AudioUrl, etc.) if model supports URLs

    **Load direction** (when replaying to a different model):
    - Converts FileUrl back to FilePart/BinaryContent if model needs bytes
    - Refreshes URLs for already-uploaded files (e.g., presigned URLs that expire)

    The processor tracks uploaded files by their storage key (stored in FileUrl.identifier),
    enabling deduplication and URL-independent file tracking.

    Args:
        store: The file store to use for uploading/retrieving files

    Returns:
        A history processor function compatible with Agent's history_processors parameter

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.file_store import S3FileStore, file_store_processor

        store = S3FileStore(bucket='my-bucket')
        agent = Agent(
            'openai:gpt-4o',
            history_processors=[file_store_processor(store)]
        )
        ```
    """
    uploaded_keys: set[str] = set()

    async def processor(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        url_support = _get_url_support(ctx.model)
        result: list[ModelMessage] = []

        for message in messages:
            if isinstance(message, ModelResponse):
                processed = await _process_response(message, store, url_support, uploaded_keys)
                result.append(processed)
            elif isinstance(message, ModelRequest):
                processed = await _process_request(message, store, url_support, uploaded_keys)
                result.append(processed)
            else:
                result.append(message)

        return result

    return processor


async def _process_response(
    response: ModelResponse,
    store: FileStore,
    url_support: dict[str, bool],
    uploaded_keys: set[str],
) -> ModelResponse:
    """Process a ModelResponse, uploading FileParts and converting to/from URL types."""
    new_parts: list[ModelResponsePart] = []
    modified = False

    for part in response.parts:
        if isinstance(part, FilePart) and part.content.data:
            new_part = await _process_file_part(part, store, url_support, uploaded_keys)
            new_parts.append(new_part)
            if new_part is not part:
                modified = True
        elif isinstance(part, (ImageUrl, AudioUrl, VideoUrl, DocumentUrl)):
            new_part = await _process_file_url(part, store, url_support, uploaded_keys)
            new_parts.append(new_part)
            if new_part is not part:
                modified = True
        else:
            new_parts.append(part)

    if modified:
        return replace(response, parts=tuple(new_parts))
    return response


async def _process_request(
    request: ModelRequest,
    store: FileStore,
    url_support: dict[str, bool],
    uploaded_keys: set[str],
) -> ModelRequest:
    """Process a ModelRequest, handling BinaryContent in UserPromptPart."""
    new_parts: list[ModelRequestPart] = []
    modified = False

    for part in request.parts:
        if isinstance(part, UserPromptPart):
            new_content = await _process_user_content(part.content, store, url_support, uploaded_keys)
            if new_content is not part.content:
                new_parts.append(replace(part, content=new_content))
                modified = True
            else:
                new_parts.append(part)
        else:
            new_parts.append(part)

    if modified:
        return replace(request, parts=tuple(new_parts))
    return request


async def _process_user_content(
    content: str | Sequence[UserContent],
    store: FileStore,
    url_support: dict[str, bool],
    uploaded_keys: set[str],
) -> str | Sequence[UserContent]:
    """Process user content, storing BinaryContent and handling FileUrls."""
    if isinstance(content, str):
        return content

    new_items: list[UserContent] = []
    modified = False

    for item in content:
        if isinstance(item, BinaryContent) and item.data:
            new_item = await _process_binary_content(item, store, url_support, uploaded_keys)
            new_items.append(new_item)
            if new_item is not item:
                modified = True
        elif isinstance(item, (ImageUrl, AudioUrl, VideoUrl, DocumentUrl)):
            new_item = await _process_file_url_for_user(item, store, url_support, uploaded_keys)
            new_items.append(new_item)
            if new_item is not item:
                modified = True
        else:
            new_items.append(item)

    return new_items if modified else content


async def _process_file_part(
    part: FilePart,
    store: FileStore,
    url_support: dict[str, bool],
    uploaded_keys: set[str],
) -> ModelResponsePart:
    """Process a FilePart: upload to store and optionally convert to URL type."""
    content = part.content
    key = generate_file_key(content)

    # Upload if not already uploaded (deduplication)
    if key not in uploaded_keys:
        await store.store(key, content.data)
        uploaded_keys.add(key)

    media_type = content.media_type
    category = _get_media_category(media_type)

    # Convert to URL type if model supports it
    if category and url_support.get(category, False):
        url = store.get_download_uri(key)
        url_class = _URL_TYPE_MAP[category]
        return url_class(url=url, media_type=media_type, identifier=key)

    return part


async def _process_binary_content(
    content: BinaryContent,
    store: FileStore,
    url_support: dict[str, bool],
    uploaded_keys: set[str],
) -> UserContent:
    """Process BinaryContent: upload to store and optionally convert to URL type."""
    key = generate_file_key(content)

    # Upload if not already uploaded (deduplication)
    if key not in uploaded_keys:
        await store.store(key, content.data)
        uploaded_keys.add(key)

    media_type = content.media_type
    category = _get_media_category(media_type)

    # Convert to URL type if model supports it
    if category and url_support.get(category, False):
        url = store.get_download_uri(key)
        url_class = _URL_TYPE_MAP[category]
        return url_class(url=url, media_type=media_type, identifier=key)

    return content


async def _process_file_url(
    part: ImageUrl | AudioUrl | VideoUrl | DocumentUrl,
    store: FileStore,
    url_support: dict[str, bool],
    uploaded_keys: set[str],
) -> ModelResponsePart:
    """Process a FileUrl: load back to FilePart if model needs bytes, or refresh URL."""
    key = part.identifier
    category = _get_media_category(part.media_type)

    # Check if this is a file we uploaded
    if key in uploaded_keys:
        # If model doesn't support this URL type, load content back
        if category and not url_support.get(category, False):
            data = await store.retrieve(key)
            content = BinaryContent(data=data, media_type=part.media_type)
            return FilePart(content=content)

        # Model supports URLs - refresh the URL if needed
        new_url = store.get_download_uri(key)
        if new_url != part.url:
            return type(part)(url=new_url, media_type=part.media_type, identifier=key)

    # External URL or unchanged - pass through
    return part


async def _process_file_url_for_user(
    part: ImageUrl | AudioUrl | VideoUrl | DocumentUrl,
    store: FileStore,
    url_support: dict[str, bool],
    uploaded_keys: set[str],
) -> UserContent:
    """Process a FileUrl in user content: load back to BinaryContent if model needs bytes."""
    key = part.identifier
    category = _get_media_category(part.media_type)

    # Check if this is a file we uploaded
    if key in uploaded_keys:
        # If model doesn't support this URL type, load content back
        if category and not url_support.get(category, False):
            data = await store.retrieve(key)
            return BinaryContent(data=data, media_type=part.media_type)

        # Model supports URLs - refresh the URL if needed
        new_url = store.get_download_uri(key)
        if new_url != part.url:
            return type(part)(url=new_url, media_type=part.media_type, identifier=key)

    # External URL or unchanged - pass through
    return part
