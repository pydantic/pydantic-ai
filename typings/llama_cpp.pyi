from collections.abc import Sequence
from os import PathLike
from typing import Any, Literal

from typing_extensions import Self

class Llama:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: str | None = None,
        additional_files: Sequence[str] | None = None,
        local_dir: str | PathLike[str] | None = None,
        local_dir_use_symlinks: bool | Literal['auto'] = 'auto',
        cache_dir: str | PathLike[str] | None = None,
        **kwargs: Any,
    ) -> Self: ...
