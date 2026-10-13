from __future__ import annotations as _annotations

from pydantic_ai.profiles import InlineDefsJsonSchemaTransformer
from pydantic_ai.profiles.nvidia import nvidia_model_profile


def test_llama_nemotron_models_use_meta_profile():
    profile = nvidia_model_profile('llama-3.1-nemotron-70b-instruct')
    assert profile is not None
    assert profile.get('json_schema_transformer') == InlineDefsJsonSchemaTransformer


def test_other_nvidia_models_have_no_family_profile():
    assert nvidia_model_profile('nemotron-3-super-120b-a12b') is None
