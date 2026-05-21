"""v2: pydantic_ai.ext.aci is removed. Wrap ACI tools manually with Tool.from_schema."""
from pydantic_ai.tools import Tool


def trigger():
    # Demonstrate the v2 replacement shape without actually contacting ACI.
    def _handler(**kwargs):
        return {'ok': True}

    Tool.from_schema(
        function=_handler,
        name='GITHUB__CREATE_ISSUE',
        description='Stub for documentation purposes',
        json_schema={'type': 'object', 'properties': {}},
    )


if __name__ == '__main__':
    trigger()
