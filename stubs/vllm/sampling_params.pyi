from typing import Any


class SamplingParams:
    max_tokens: int | None
    temperature: float | None
    top_p: float | None
    seed: int | None
    presence_penalty: float | None
    frequency_penalty: float | None
    logit_bias: dict[int, float] | None
    extra_body: dict[str, Any] | None

    def __init__(
        self,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[int, float] | None = None,
        extra_body: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
