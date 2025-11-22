from typing import TYPE_CHECKING, Any, Union

from outlines.models.base import AsyncModel, Model

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

class SGLang(Model): ...
class AsyncSGLang(AsyncModel): ...

def from_sglang(client: OpenAI | AsyncOpenAI, *args: Any, **kwargs: Any) -> SGLang | AsyncSGLang: ...
