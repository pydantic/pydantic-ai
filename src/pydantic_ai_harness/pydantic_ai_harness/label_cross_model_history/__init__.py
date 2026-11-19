"""Tell the model when earlier assistant turns came from another model."""

from pydantic_ai_harness.label_cross_model_history._capability import DEFAULT_CROSS_MODEL_NOTICE, LabelCrossModelHistory

__all__ = [
    'DEFAULT_CROSS_MODEL_NOTICE',
    'LabelCrossModelHistory',
]
