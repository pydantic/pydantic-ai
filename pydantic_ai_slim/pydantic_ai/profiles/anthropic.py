from __future__ import annotations as _annotations

from dataclasses import dataclass

from anthropic.types.beta import (
    BetaToolTextEditor20241022Param,
    BetaToolTextEditor20250124Param,
    BetaToolTextEditor20250728Param,
)

from . import ModelProfile


@dataclass
class AnthropicModelProfile(ModelProfile):
    """Profile for models used with AnthropicModel.

    ALL FIELDS MUST BE `anthropic_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    anthropic_text_editor_tool: (
        BetaToolTextEditor20241022Param | BetaToolTextEditor20250124Param | BetaToolTextEditor20250728Param | None
    ) = None
    """The text editor tool to use for the model.

    See <http://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool> for more details.
    """


def anthropic_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Anthropic model."""
    if model_name.startswith('claude-3-5-sonnet'):
        text_editor_tool = BetaToolTextEditor20241022Param(name='str_replace_editor', type='text_editor_20241022')
    elif model_name.startswith('claude-3-7-sonnet'):
        text_editor_tool = BetaToolTextEditor20250124Param(name='str_replace_editor', type='text_editor_20250124')
    elif model_name.startswith('claude-4'):
        text_editor_tool = BetaToolTextEditor20250728Param(
            name='str_replace_based_edit_tool', type='text_editor_20250728'
        )
    else:
        text_editor_tool = None

    return AnthropicModelProfile(
        thinking_tags=('<thinking>', '</thinking>'),
        anthropic_text_editor_tool=text_editor_tool,
    )
