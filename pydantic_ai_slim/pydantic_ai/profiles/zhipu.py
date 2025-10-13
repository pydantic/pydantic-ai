from __future__ import annotations as _annotations

from .openai import OpenAIModelProfile


def zhipu_model_profile(model_name: str) -> OpenAIModelProfile:
    """Get the model profile for a Zhipu AI model.

    Zhipu AI provides OpenAI-compatible API, so we use OpenAIModelProfile.

    Args:
        model_name: The Zhipu model name (e.g., 'glm-4.6', 'glm-4.5', 'glm-4.5-air', 'glm-4.5v').

    Returns:
        Model profile with Zhipu-specific configurations.
    """
    # Vision models — docs show vision variants with a trailing `v` like `glm-4.5v`
    # Ref: https://docs.bigmodel.cn/cn/guide/develop/openai/introduction
    is_vision_model = model_name.startswith(('glm-4.5v', 'glm-4v')) or (
        'v' in model_name and ('glm-4.5v' in model_name or 'glm-4v' in model_name)
    )

    # Zhipu AI models support JSON schema and object output
    # All GLM-4 series models support function calling
    supports_tools = model_name.startswith(('glm-4', 'codegeex-4'))

    # Zhipu AI doesn't support temperature=0 (must be in range (0, 1))
    # This is a known difference from OpenAI
    openai_unsupported_model_settings = ()

    return OpenAIModelProfile(
        supports_json_schema_output=supports_tools,
        supports_json_object_output=supports_tools,
        supports_image_output=is_vision_model,
        openai_supports_strict_tool_definition=False,  # Zhipu doesn't support strict mode
        openai_supports_tool_choice_required=True,
        openai_unsupported_model_settings=openai_unsupported_model_settings,
        openai_system_prompt_role=None,  # Use default 'system' role
        openai_chat_supports_web_search=False,
        openai_supports_encrypted_reasoning_content=False,
    )
