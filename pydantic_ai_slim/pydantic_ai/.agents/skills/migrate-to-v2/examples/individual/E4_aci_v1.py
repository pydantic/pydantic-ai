"""v1: pydantic_ai.ext.aci.ACIToolset emits PydanticAIDeprecationWarning on construction."""
from pydantic_ai.ext.aci import ACIToolset


def trigger():
    # DEPRECATION: E4_aci — fires on instantiation (not bare import)
    ACIToolset(aci_functions=[], linked_account_owner_id='owner')


EXPECT = '`pydantic_ai.ext.aci` is deprecated and will be removed in 2.0'

if __name__ == '__main__':
    trigger()
