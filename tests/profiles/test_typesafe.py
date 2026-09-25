from pydantic_ai.profiles import DEFAULT_PROFILE
from pydantic_ai.profiles.typesafe import typesafe_model_profile


def test_supports_text_output():
    assert DEFAULT_PROFILE.get('supports_text_output') is True
    profile = typesafe_model_profile('jev-latest')
    assert profile is not None
    assert profile.get('supports_text_output') is False


def test_context_window_is_the_binding_limit():
    # Jev refuses a request whose state plus longest question passes ~32k tokens, well before its 64k combined
    # budget, so compaction keyed on `context_window_used` has to measure against 32k to fire in time.
    profile = typesafe_model_profile('jev-latest')
    assert profile is not None
    assert profile.get('context_window') == 32_000
