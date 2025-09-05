import pytest

from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.outlines import OutlinesProvider


def test_outlines_provider() -> None:
    provider = OutlinesProvider()
    assert provider.name == 'outlines'

    with pytest.raises(
        NotImplementedError,
        match=(
            'The Outlines provider does not have a set base URL as it functions '
            + 'with a set of different underlying models.'
        ),
    ):
        provider.base_url

    with pytest.raises(
        NotImplementedError,
        match=(
            'The Outlines provider does not have a set client as it functions '
            + 'with a set of different underlying models.'
        ),
    ):
        provider.client

    assert provider.model_profile('outlines-model') == ModelProfile(
        supports_tools=False,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        default_structured_output_mode='native',
        thinking_tags=('<think>', '</think>'),
        ignore_streamed_leading_whitespace=False,
    )
