import importlib

# Forward imports from the slim package implementation
_real = importlib.import_module('pydantic_ai_slim.pydantic_ai._spec')
# Export everything
globals().update(_real.__dict__)
# Preserve __all__ if defined
if hasattr(_real, '__all__'):
    __all__ = list(_real.__all__)
