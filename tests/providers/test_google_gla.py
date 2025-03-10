import os
import re
from unittest.mock import patch

import pytest

from pydantic_ai.providers.google_gla import GoogleGLAProvider


def test_google_gla_provider_need_api_key() -> None:
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(
            ValueError,
            match=re.escape(
                'Set the `GEMINI_API_KEY` environment variable or pass it via `GoogleGLAProvider(api_key=...)`'
                'to use the Google GLA provider.'
            ),
        ):
            GoogleGLAProvider()
