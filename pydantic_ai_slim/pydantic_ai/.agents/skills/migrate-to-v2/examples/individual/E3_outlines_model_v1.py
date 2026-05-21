"""v1: OutlinesModel instantiation emits PydanticAIDeprecationWarning.

The class is decorated with `@deprecated`, so the warning fires on construction,
not on bare import.
"""
from pydantic_ai.models.outlines import OutlinesModel


def trigger():
    # DEPRECATION: E3_outlines_model
    try:
        OutlinesModel(model=None)  # type: ignore[arg-type]
    except Exception:
        # Model arg is invalid but the @deprecated warning already fired.
        pass


EXPECT = '`OutlinesModel` is deprecated and will be removed in v2'

if __name__ == '__main__':
    trigger()
