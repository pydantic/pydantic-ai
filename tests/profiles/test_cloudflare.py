import pytest

from pydantic_ai.models import infer_model_profile
from pydantic_ai.models.cloudflare import CloudflareClefModel
from pydantic_ai.models.cloudflare_clairic import CloudflareClairicModel
from pydantic_ai.profiles.cloudflare import cloudflare_model_profile
from pydantic_ai.profiles.decision import DecisionModelProfile
from pydantic_ai.providers.cloudflare import CloudflareProvider


@pytest.mark.parametrize(
    ('model_name', 'context_window'),
    [('@cf/cloudflare/clef', 65_536), ('@cf/cloudflare/clef-flash', 24_576)],
)
def test_model_profile(model_name: str, context_window: int):
    profile = cloudflare_model_profile(model_name)
    assert profile is not None
    assert profile.get('context_window') == context_window
    assert profile.get('decision_max_choice_options') == 255
    assert profile.get('decision_max_score_levels') == 10
    assert profile.get('supports_tools') is True
    assert profile.get('supports_text_output') is False
    assert profile.get('supports_audio_input') is False
    assert CloudflareProvider.model_profile(model_name) == profile
    assert infer_model_profile(f'cloudflare:{model_name}') == profile


def test_unknown_model():
    assert cloudflare_model_profile('@cf/unknown/model') is None


def test_clairic_profile():
    profile = cloudflare_model_profile('@cf/cloudflare/clairic')
    assert profile is not None
    assert profile.get('context_window') == 65_536
    assert profile.get('default_structured_output_mode') == 'native'
    assert profile.get('supports_json_schema_output') is True
    assert profile.get('supports_tools') is False
    assert profile.get('supports_text_output') is False
    assert CloudflareProvider.model_profile('@cf/cloudflare/clairic') == profile
    assert infer_model_profile('cloudflare:@cf/cloudflare/clairic') == profile


async def test_clairic_profile_overrides():
    async with CloudflareProvider(api_key='token', account_id='account') as provider:
        model = CloudflareClairicModel(provider=provider, profile={'context_window': 4096})
        assert model.profile.get('context_window') == 4096
        assert model.profile.get('supports_json_schema_output') is True
        assert model.profile.get('supports_tools') is False


async def test_profile_overrides():
    async with CloudflareProvider(api_key='token', account_id='account') as provider:
        profile = DecisionModelProfile(context_window=4096, decision_max_choice_options=4)
        model = CloudflareClefModel('@cf/cloudflare/clef-flash', provider=provider, profile=profile)
        assert model.profile.get('context_window') == 4096
        assert model.profile.get('decision_max_choice_options') == 4
        assert model.profile.get('decision_max_score_levels') == 10
        assert model.profile.get('supports_text_output') is False
        assert model.profile.get('supports_inline_system_prompts') is True
        assert model.supports_image_input is True
        assert model.max_images == 4
        assert model.max_questions == 64
