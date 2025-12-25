from typing import Any

from pydantic_core import core_schema


class SpanTreeRecordingError(Exception):
    """An exception that is used to provide the reason why a SpanTree was not recorded by `context_subtree`.

    This will either be due to missing dependencies or because a tracer provider had not been set.
    """

    message: str

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> core_schema.CoreSchema:
        """Pydantic core schema to allow `SpanTreeRecordingError` to be (de)serialized."""
        schema = core_schema.typed_dict_schema(
            {
                'message': core_schema.typed_dict_field(core_schema.str_schema()),
                'kind': core_schema.typed_dict_field(core_schema.literal_schema(['span-tree-recording-error'])),
            }
        )
        return core_schema.no_info_after_validator_function(
            lambda dct: SpanTreeRecordingError(dct['message']),
            schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: {'message': x.message, 'kind': 'span-tree-recording-error'},
                return_schema=schema,
            ),
        )
