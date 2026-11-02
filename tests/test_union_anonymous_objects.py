"""Union members that are anonymous objects must not collapse into one TypedDict."""

from pydantic_ai.function_signature import FunctionSignature


def test_anonymous_objects_in_union_get_distinct_names():
    union = {
        'anyOf': [
            {'type': 'object', 'properties': {'url': {'type': 'string'}}, 'required': ['url']},
            {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']},
        ]
    }
    sig = FunctionSignature.from_schema(
        name='search',
        parameters_schema={'type': 'object', 'properties': {'payload': union}, 'required': ['payload']},
    )
    assert [(t.name, sorted(t.fields)) for t in sig.referenced_types] == [
        ('SearchPayload', ['url']),
        ('SearchPayload1', ['query']),
    ]
    rendered = sig.render("...", name="search")
    assert rendered == """\
def search(*, payload: SearchPayload | SearchPayload1) -> Any:
    ...\
"""
    definitions = FunctionSignature.render_type_definitions([sig], frozenset())
    compile(chr(10).join([rendered, *definitions]), "<test>", "exec")


def test_union_members_after_null_still_get_distinct_names():
    sig = FunctionSignature.from_schema(
        name="fetch",
        parameters_schema={
            'type': 'object',
            'properties': {
                'body': {
                    'oneOf': [
                        {'type': 'null'},
                        {'type': 'object', 'properties': {'url': {'type': 'string'}}, 'required': ['url']},
                        {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']},
                    ]
                }
            },
            'required': ['body'],
        },
    )
    assert [(t.name, sorted(t.fields)) for t in sig.referenced_types] == [
        ('FetchBody', ['url']),
        ('FetchBody1', ['query']),
    ]
