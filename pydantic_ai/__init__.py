import importlib
import sys
import pkgutil

# Pre‑load the instrumentation submodule that the slim package expects under the legacy name
try:
    _instrumentation = importlib.import_module('pydantic_ai_slim.pydantic_ai._instrumentation')
    sys.modules['pydantic_ai._instrumentation'] = _instrumentation
except ImportError:
    pass

# Import the actual implementation package
_real_pkg = importlib.import_module('pydantic_ai_slim.pydantic_ai')

# Expose all attributes of the real package at the top level of the shim
globals().update(_real_pkg.__dict__)

# Ensure that submodules of the real package are importable via the old package name
if hasattr(_real_pkg, '__path__'):
    for finder, name, ispkg in pkgutil.iter_modules(_real_pkg.__path__, _real_pkg.__name__ + '.'):
        submod = importlib.import_module(name)
        alias_name = name.replace('pydantic_ai_slim.', 'pydantic_ai.')
        sys.modules[alias_name] = submod
        if alias_name == 'pydantic_ai.__init__':
            sys.modules['pydantic_ai'] = submod

# Finally, register the shim itself as the package

