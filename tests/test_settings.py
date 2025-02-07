from typing import cast

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.settings import ModelSettings, validate_model_settings


def test_valid_model_settings():
    # Test with valid parameters
    settings: ModelSettings = {'temperature': 0.7, 'max_tokens': 100, 'top_p': 0.9}
    validate_model_settings(settings)  # Should not raise any exception


def test_invalid_model_settings():
    # Test with an invalid parameter
    settings = cast(
        ModelSettings,
        {
            'temperature': 0.7,
            'tempural': 45,  # Invalid parameter
        },
    )
    with pytest.raises(UserError) as exc_info:
        validate_model_settings(settings)

    assert 'Invalid model setting parameter(s): tempural' in str(exc_info.value)
    assert 'Valid parameters are:' in str(exc_info.value)


def test_multiple_invalid_model_settings():
    # Test with multiple invalid parameters
    settings = cast(ModelSettings, {'temperature': 0.7, 'tempural': 45, 'invalid_param': 'value'})
    with pytest.raises(UserError) as exc_info:
        validate_model_settings(settings)

    assert 'Invalid model setting parameter(s): invalid_param, tempural' in str(exc_info.value)
    assert 'Valid parameters are:' in str(exc_info.value)


