from functools import partial
from typing import Any
from unittest.mock import Mock

from pydantic_ai import RunContext
from pydantic_ai._function_schema import takes_ctx as _takes_ctx


def test_regular_function_with_ctx():
    """Test regular function that takes RunContext as first parameter."""

    def func_with_ctx(ctx: RunContext[Any], x: int) -> str: ...  # pragma: no cover

    assert _takes_ctx(func_with_ctx) is True


def test_regular_function_without_ctx():
    """Test regular function that doesn't take RunContext as first parameter."""

    def func_without_ctx(x: int, y: str) -> str: ...  # pragma: no cover

    assert _takes_ctx(func_without_ctx) is False


def test_regular_function_no_params():
    """Test regular function with no parameters."""

    def func_no_params() -> str: ...  # pragma: no cover

    assert _takes_ctx(func_no_params) is False


def test_regular_function_ctx_not_first():
    """Test regular function where RunContext is not the first parameter."""

    def func_ctx_not_first(x: int, ctx: RunContext[Any]) -> str: ...  # pragma: no cover

    assert _takes_ctx(func_ctx_not_first) is False


def test_partial_function_with_ctx():
    """Test partial function where original function takes RunContext as first parameter."""

    def original_func(ctx: RunContext[Any], x: int, y: str) -> str: ...  # pragma: no cover

    # Create partial with y bound
    partial_func = partial(original_func, y='bound')

    assert _takes_ctx(partial_func) is True


def test_partial_function_without_ctx():
    """Test partial function where original function doesn't take RunContext."""

    def original_func(x: int, y: str, z: float) -> str: ...  # pragma: no cover

    # Create partial with z bound
    partial_func = partial(original_func, z=3.14)

    assert _takes_ctx(partial_func) is False


def test_partial_function_ctx_bound():
    """Test partial function where RunContext parameter is bound."""

    def original_func(ctx: RunContext[Any], x: int, y: str) -> str: ...  # pragma: no cover

    mock_ctx = Mock(spec=RunContext[Any])
    partial_func = partial(original_func, ctx=mock_ctx)

    assert _takes_ctx(partial_func) is True


def test_callable_class_with_ctx():
    """Test callable class where __call__ takes RunContext as first parameter."""

    class CallableWithCtx:
        def __call__(self, ctx: RunContext[Any], x: int) -> str: ...  # pragma: no cover

    callable_obj = CallableWithCtx()

    assert _takes_ctx(callable_obj) is True


def test_callable_class_without_ctx():
    """Test callable class where __call__ doesn't take RunContext."""

    class CallableWithoutCtx:
        def __call__(self, x: int, y: str) -> str: ...  # pragma: no cover

    callable_obj = CallableWithoutCtx()

    assert _takes_ctx(callable_obj) is False


def test_callable_class_ctx_not_first():
    """Test callable class where RunContext is not the first parameter."""

    class CallableCtxNotFirst:
        def __call__(self, x: int, ctx: RunContext[Any]) -> str: ...  # pragma: no cover

    callable_obj = CallableCtxNotFirst()

    assert _takes_ctx(callable_obj) is False


def test_method_with_ctx():
    """Test bound method that takes RunContext as first parameter (after )."""

    class TestClass:
        def method_with_ctx(self, ctx: RunContext[Any], x: int) -> str: ...  # pragma: no cover

    obj = TestClass()
    bound_method = obj.method_with_ctx

    assert _takes_ctx(bound_method) is True


def test_method_without_ctx():
    """Test bound method that doesn't take RunContext."""

    class TestClass:
        def method_without_ctx(self, x: int, y: str) -> str: ...  # pragma: no cover

    obj = TestClass()
    bound_method = obj.method_without_ctx

    assert _takes_ctx(bound_method) is False


def test_static_method_with_ctx():
    """Test static method that takes RunContext as first parameter."""

    class TestClass:
        @staticmethod
        def static_method_with_ctx(ctx: RunContext[Any], x: int) -> str: ...  # pragma: no cover

    assert _takes_ctx(TestClass.static_method_with_ctx) is True


def test_static_method_without_ctx():
    """Test static method that doesn't take RunContext."""

    class TestClass:
        @staticmethod
        def static_method_without_ctx(x: int, y: str) -> str: ...  # pragma: no cover

    assert _takes_ctx(TestClass.static_method_without_ctx) is False


def test_class_method_with_ctx():
    """Test class method that takes RunContext as first parameter (after cls)."""

    class TestClass:
        @classmethod
        def class_method_with_ctx(cls, ctx: RunContext[Any], x: int) -> str: ...  # pragma: no cover

    assert _takes_ctx(TestClass.class_method_with_ctx) is True


def test_class_method_without_ctx():
    """Test class method that doesn't take RunContext."""

    class TestClass:
        @classmethod
        def class_method_without_ctx(cls, x: int, y: str) -> str: ...  # pragma: no cover

    assert _takes_ctx(TestClass.class_method_without_ctx) is False


def test_function_no_annotations():
    """Test function with no type annotations."""

    def func_no_annotations(ctx, x):  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        ...  # pragma: no cover

    # Without annotations, _takes_ctx should return False
    assert _takes_ctx(func_no_annotations) is False  # pyright: ignore[reportUnknownArgumentType]


def test_function_wrong_annotation_type():
    """Test function with wrong annotation type for first parameter."""

    def func_wrong_annotation(ctx: str, x: int) -> str: ...  # pragma: no cover

    assert _takes_ctx(func_wrong_annotation) is False


def test_lambda_with_ctx():
    """Test lambda function that takes RunContext as first parameter."""
    lambda_with_ctx = lambda ctx, x: f'{ctx.deps} {x}'  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType, reportUnknownVariableType]  # noqa: E731

    # Lambda without annotations should return False
    assert _takes_ctx(lambda_with_ctx) is False  # pyright: ignore[reportUnknownArgumentType]


def test_builtin_function():
    """Test builtin function."""
    assert _takes_ctx(len) is False
    assert _takes_ctx(str) is False
    assert _takes_ctx(int) is False


def test_file_url_tool_arguments_strip_force_download():
    """FileUrl tool parameters must omit force_download from schema and ignore model-supplied values."""
    import json

    from pydantic import TypeAdapter, create_model
    from pydantic.json_schema import GenerateJsonSchema

    from pydantic_ai._function_schema import function_schema
    from pydantic_ai.messages import AudioUrl, DocumentUrl, FileUrl, ImageUrl, VideoUrl

    file_url_types: list[type[FileUrl]] = [ImageUrl, DocumentUrl, AudioUrl, VideoUrl]

    for url_cls in file_url_types:
        # Single model-like parameter
        def single_tool(file: Any) -> str:  # pragma: no cover
            return str(file)

        single_tool.__annotations__['file'] = url_cls
        fs_single = function_schema(single_tool, GenerateJsonSchema)
        assert 'force_download' not in json.dumps(fs_single.json_schema)

        for bad_mode in (True, 'allow-local'):
            validated = fs_single.validator.validate_python(
                {'url': 'https://example.com/file', 'force_download': bad_mode}
            )
            assert isinstance(validated['file'], url_cls)
            assert validated['file'].force_download is False

            validated_json = fs_single.validator.validate_json(
                json.dumps({'url': 'https://example.com/file', 'force_download': bad_mode})
            )
            assert isinstance(validated_json['file'], url_cls)
            assert validated_json['file'].force_download is False

        # Multi-argument and nested model/list parameters
        Wrapper = create_model('Wrapper', primary=(url_cls, ...), extras=(list[url_cls], []))

        def nested_tool(wrapper: Any, direct: Any) -> str:  # pragma: no cover
            return f'{wrapper}{direct}'

        nested_tool.__annotations__['wrapper'] = Wrapper
        nested_tool.__annotations__['direct'] = url_cls
        fs_nested = function_schema(nested_tool, GenerateJsonSchema)
        assert 'force_download' not in json.dumps(fs_nested.json_schema)

        validated_nested = fs_nested.validator.validate_python(
            {
                'wrapper': {
                    'primary': {'url': 'https://example.com/a', 'force_download': 'allow-local'},
                    'extras': [{'url': 'https://example.com/b', 'force_download': True}],
                },
                'direct': {'url': 'https://example.com/c', 'force_download': 'allow-local'},
            }
        )
        assert validated_nested['wrapper'].primary.force_download is False
        assert validated_nested['wrapper'].extras[0].force_download is False
        assert validated_nested['direct'].force_download is False

        # Non-tool TypeAdapter validation must remain unaffected
        direct_adapter = TypeAdapter(url_cls)
        assert 'force_download' in json.dumps(direct_adapter.json_schema())
        direct_obj = direct_adapter.validate_python(
            {'url': 'https://example.com/d', 'force_download': 'allow-local'}
        )
        assert direct_obj.force_download == 'allow-local'
