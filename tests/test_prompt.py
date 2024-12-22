import json
from typing import final, override

from pydantic_ai.prompt import TagBuilder


@final
class DummyTagBuilder(TagBuilder):
    @override
    def build(self) -> str:
        return json.dumps(self._content)


class TestContentFormatting:
    def test_str(self) -> None:
        builder = DummyTagBuilder('test', 'Hello, world!')
        content = json.loads(builder.build())

        assert content['test'] == 'Hello, world!'


# class TestContentFormatting:
#     def test_dict(self) -> None:
#         raise NotImplementedError
#
#     def test_base_model(self) -> None:
#         raise NotImplementedError
#
#     def test_dataclass(self) -> None:
#         raise NotImplementedError
#
#     def test_iterable(self) -> None:
#         raise NotImplementedError
#
#     def test_unsupported_content(self) -> None:
#         raise NotImplementedError
#
#
# class TestXMLEncoder:
#     pass
