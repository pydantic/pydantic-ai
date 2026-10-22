"""Cloudflare keeps native decision imports independent from the optional chat SDK."""

from unittest.mock import Mock

import pytest

from pydantic_clai2.models.model_catalog import check_installed


def test_chat_checks_optional_sdk(monkeypatch: pytest.MonkeyPatch):
    def missing_sdk(module: str):
        assert module == 'pydantic_ai.models.cloudflare'
        raise ImportError('Please install the OpenAI SDK')

    monkeypatch.setattr('pydantic_clai2.models.model_catalog.import_module', missing_sdk)
    with pytest.raises(ValueError, match='OpenAI SDK'):
        check_installed('cloudflare:@cf/meta/llama-4-scout-17b-16e-instruct')


def test_clef_does_not_require_chat_sdk(monkeypatch: pytest.MonkeyPatch):
    no_chat_import = Mock()
    monkeypatch.setattr('pydantic_clai2.models.model_catalog.import_module', no_chat_import)
    check_installed('cloudflare:@cf/cloudflare/clef-flash')
    no_chat_import.assert_not_called()
