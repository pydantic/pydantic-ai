"""Tests for redirect URL resolution."""

from pydantic_ai._ssrf import resolve_redirect_url


def test_query_only_redirect_preserves_the_current_resource() -> None:
    assert resolve_redirect_url('https://example.com/a/file?old=1', '?new=2') == 'https://example.com/a/file?new=2'


def test_fragment_only_redirect_preserves_the_current_resource_and_query() -> None:
    assert (
        resolve_redirect_url('https://example.com/a/file?old=1', '#section')
        == 'https://example.com/a/file?old=1#section'
    )


def test_protocol_relative_redirect_preserves_path_parameters() -> None:
    assert (
        resolve_redirect_url('https://example.com/a/file', '//other.example/path;param?x=1')
        == 'https://other.example/path;param?x=1'
    )
