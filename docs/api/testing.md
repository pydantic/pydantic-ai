# `pydantic_ai.testing`

Utilities for integration testing Pydantic AI applications with VCR cassettes.

This module provides a custom YAML serializer for [VCR.py](https://vcrpy.readthedocs.io/)
that handles LLM API responses with:

- JSON body parsing and pretty-printing
- Gzip/Brotli decompression
- Smart quote normalization (Unicode to ASCII)
- Sensitive header filtering
- Access token scrubbing

Here's a minimal example using pytest-recording:

```py {title="conftest.py"}
from vcr import VCR

from pydantic_ai.testing import json_body_serializer


def pytest_recording_configure(config, vcr: VCR):
    vcr.register_serializer('yaml', json_body_serializer)
```

See [Integration Testing with VCR](../testing.md#integration-testing-with-vcr)
for detailed documentation.

::: pydantic_ai.testing
