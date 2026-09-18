from pydantic_ai.profiles import DEFAULT_PROFILE
from pydantic_ai.profiles.typesafe import typesafe_model_profile


def test_supports_text_output():
    assert DEFAULT_PROFILE['supports_text_output'] is True
    profile = typesafe_model_profile('jev-latest')
    assert profile is not None
    assert profile['supports_text_output'] is False
