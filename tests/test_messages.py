import pytest

from pydantic_ai.messages import BinaryContent, DocumentUrl, ImageUrl


def test_image_url():
    image_url = ImageUrl(url='https://example.com/image.jpg')
    assert image_url.media_type == 'image/jpeg'
    assert image_url.format == 'jpeg'


def test_document_url():
    with pytest.raises(RuntimeError, match='Unknown document file extension: https://example.com/document.potato'):
        document_url = DocumentUrl(url='https://example.com/document.potato')
        document_url.media_type

    document_url = DocumentUrl(url='https://example.com/document.pdf')
    assert document_url.media_type == 'application/pdf'
    assert document_url.format == 'pdf'


@pytest.mark.parametrize(
    'media_type, format',
    [
        ('audio/wav', 'wav'),
        ('audio/mpeg', 'mp3'),
    ],
)
def test_binary_content_audio(media_type: str, format: str):
    binary_content = BinaryContent(data=b'Hello, world!', media_type=media_type)
    assert binary_content.is_audio
    assert binary_content.format == format


@pytest.mark.parametrize(
    'media_type, format',
    [
        ('image/jpeg', 'jpeg'),
        ('image/png', 'png'),
        ('image/gif', 'gif'),
        ('image/webp', 'webp'),
    ],
)
def test_binary_content_image(media_type: str, format: str):
    binary_content = BinaryContent(data=b'Hello, world!', media_type=media_type)
    assert binary_content.is_image
    assert binary_content.format == format


@pytest.mark.parametrize(
    'media_type, format',
    [
        ('application/pdf', 'pdf'),
        ('text/plain', 'txt'),
        ('text/csv', 'csv'),
        ('application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'docx'),
        ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'xlsx'),
        ('text/html', 'html'),
        ('text/markdown', 'md'),
        ('application/vnd.ms-excel', 'xls'),
    ],
)
def test_binary_content_document(media_type: str, format: str):
    binary_content = BinaryContent(data=b'Hello, world!', media_type=media_type)
    assert binary_content.is_document
    assert binary_content.format == format
