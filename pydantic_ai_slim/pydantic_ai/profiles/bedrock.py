from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_ai import ModelProfile
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile


@dataclass(kw_only=True)
class BedrockModelProfile(ModelProfile):
    """Profile for models used with BedrockModel.

    ALL FIELDS MUST BE `bedrock_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    bedrock_supports_tool_choice: bool = False
    bedrock_tool_result_format: Literal['text', 'json'] = 'text'
    bedrock_send_back_thinking_parts: bool = False


def bedrock_amazon_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Amazon model used via Bedrock."""
    profile = amazon_model_profile(model_name)
    if 'nova' in model_name:
        return BedrockModelProfile(bedrock_supports_tool_choice=True).update(profile)
    return profile


def bedrock_deepseek_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a DeepSeek model used via Bedrock."""
    profile = deepseek_model_profile(model_name)
    if 'r1' in model_name:
        return BedrockModelProfile(bedrock_send_back_thinking_parts=True).update(profile)
    return profile  # pragma: no cover
