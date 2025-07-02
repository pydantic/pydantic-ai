from aci import ACI
from pydantic_ai import Tool
from typing import Any

def _clean_schema(schema):
    if isinstance(schema, dict):
        # Remove non-standard keys (e.g., 'visible')
        return {k: clean_schema(v) for k, v in schema.items() if k not in {'visible'}}
    elif isinstance(schema, list):
        return [clean_schema(item) for item in schema]
    else:
        return schema

def tool_from_aci(
    aci_function: str,
    linked_account_owner_id: str
) -> Tool:
    """Creates a PydanticAI tool from an ACI function.

    Args:
        aci_function: The ACI function to use.
        linked_account_owner_id: The ACI user ID to execute the function on behalf of.

    Returns:
        A PydanticAI tool that can be used in an Agent.

    Example:
    ```python
    tavily_search = tool_from_aci(
        "TAVILY__SEARCH",
        linked_account_owner_id=LINKED_ACCOUNT_OWNER_ID
    )

    agent = Agent(
        "openai:gpt-4.1",
        tools=[tavily_search]
    )
    
    result = agent.run_sync("Search the web and tell me the next match Chelsea will play and when the match is")
    print(result.output)
    ```
    """
    aci = ACI()
    function_definition = aci.functions.get_definition(aci_function)
    function_name = function_definition["function"]['name']
    function_description = function_definition["function"]['description']
    inputs = function_definition["function"]['parameters']

    json_schema = {
        'additionalProperties': inputs.get('additionalProperties', False),
        'properties': inputs.get('properties', {}),
        'required': inputs.get('required', []),
        'type': inputs.get('type', 'object'), # Default to 'object' if not specified
    }

    # Clean the schema
    json_schema = _clean_schema(json_schema)

    def implementation(*args: Any, **kwargs: Any) -> str:
        if args:
            raise TypeError("Positional arguments are not allowed")
        return aci.handle_function_call(
            function_name,
            kwargs,
            linked_account_owner_id=linked_account_owner_id,
            allowed_apps_only=True,
        )

    return Tool.from_schema(
        function=implementation,
        name=function_name,
        description=function_description,
        json_schema=json_schema,
    )
