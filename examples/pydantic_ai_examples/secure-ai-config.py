# type: ignore
"""Permit.io configuration for Financial Advisor security perimeters.

Sets up the complete ABAC (Attribute-Based Access Control) model including:
- Resources and their attributes
- Roles and their base permissions
- Condition sets for fine-grained access control
- User sets with their attributes
- Resource sets based on classification levels.
"""

from __future__ import annotations as _annotations

import asyncio
import os

try:
    from permit import Permit

    PERMIT_KEY = os.environ['PERMIT_KEY']
    permit = Permit(
        token=PERMIT_KEY,
    )
except ImportError:
    pass


# API keys


# Initialize Permit.io SDK


# Define resources for Financial Advisor security
resources = [
    {
        'key': 'financial_advice',
        'name': 'Financial Advice',
        'description': 'AI-generated financial advice',
        'actions': {
            'receive': {},
        },
        'attributes': {
            'is_ai_generated': {
                'type': 'bool',
                'description': 'Whether the advice is AI-generated',
            },
            'risk_level': {
                'type': 'string',
                'description': 'Risk level of the advice (low, medium, high)',
            },
        },
    },
    {
        'key': 'financial_document',
        'name': 'Financial Document',
        'description': 'Financial knowledge documents',
        'actions': {
            'read': {},
        },
        'attributes': {
            'doc_type': {
                'type': 'string',
                'description': 'Type of financial document',
            },
            'classification': {
                'type': 'string',
                'description': 'Document classification level',
            },
            'clearance_required': {
                'type': 'string',
                'description': 'Required clearance level to access',
            },
        },
    },
    {
        'key': 'financial_response',
        'name': 'Financial Response',
        'description': 'AI-generated response content',
        'actions': {
            'requires_disclaimer': {},
        },
        'attributes': {
            'contains_advice': {
                'type': 'bool',
                'description': 'Whether the response contains financial advice',
            },
            'risk_level': {
                'type': 'string',
                'description': 'Risk level of the response',
            },
        },
    },
    {
        'key': 'portfolio',
        'name': 'Investment Portfolio',
        'description': 'User investment portfolio',
        'actions': {
            'update': {},
            'read': {},
            'analyze': {},
        },
        'attributes': {
            'owner_id': {
                'type': 'string',
                'description': 'Portfolio owner ID',
            },
            'value_tier': {
                'type': 'string',
                'description': 'Portfolio value classification',
            },
        },
    },
]

# Define user attributes
user_attributes = [
    {
        'key': 'clearance_level',
        'type': 'string',
        'description': "User's security clearance level (low, high)",
    },
    {
        'key': 'ai_advice_opted_in',
        'type': 'bool',
        'description': 'Whether user has opted in to receive AI-generated advice',
    },
]

# Define user sets with their attributes
user_sets = [
    {
        'key': 'opted_in_users',
        'name': 'AI Advice Opted-in Users',
        'description': 'Users who have consented to AI-generated advice',
        'type': 'userset',
        'conditions': {'allOf': [{'user.ai_advice_opted_in': {'equals': True}}]},
    },
    {
        'key': 'high_clearance_users',
        'name': 'High Clearance Users',
        'description': 'Users with high-level document access',
        'type': 'userset',
        'conditions': {'allOf': [{'user.clearance_level': {'equals': 'high'}}]},
    },
]

# Define resource sets based on classification
resource_sets = [
    {
        'key': 'confidential_docs',
        'type': 'resourceset',
        'resource_id': 'financial_document',
        'name': 'Confidential Documents',
        'description': 'Documents with confidential classification',
        'conditions': {
            'allOf': [{'resource.classification': {'equals': 'confidential'}}]
        },
    },
    {
        'key': 'finance_advice',
        'type': 'resourceset',
        'resource_id': 'financial_advice',
        'name': 'Financial Advice',
        'description': 'Financial advice with ai content',
        'conditions': {'allOf': [{'resource.is_ai_generated': {'equals': True}}]},
    },
]

# Define condition set rules to link user sets with resource sets
condition_set_rules = [
    {
        'user_set': 'opted_in_users',
        'permission': 'financial_advice:receive',
        'resource_set': 'finance_advice',
    },
    {
        'user_set': 'high_clearance_users',
        'permission': 'financial_document:read',
        'resource_set': 'confidential_docs',
    },
]

# Define roles with ABAC rules
roles = [
    {'name': 'restricted_user'},
    {
        'name': 'premium_user',
        'permissions': [
            {
                'resource': 'financial_advice',
                'actions': ['receive'],
                'attributes': {'is_ai_generated': ['true', 'false']},
                'condition_sets': ['opt_in_check', 'risk_level_check'],
            },
            {
                'resource': 'financial_document',
                'actions': ['read'],
                'condition_sets': ['document_clearance'],
            },
            {
                'resource': 'portfolio',
                'actions': ['update', 'read', 'analyze'],
                'attributes': {'value_tier': ['premium', 'standard']},
            },
        ],
    },
]

# Define example users with their attributes
example_users = [
    {
        'key': 'user@example.com',
        'email': 'user@example.com',
        'first_name': 'Example',
        'last_name': 'User',
        'attributes': {
            'clearance_level': 'high',
            'ai_advice_opted_in': True,
        },
        'role': 'premium_user',
    },
    {
        'key': 'restricted@example.com',
        'email': 'restricted@example.com',
        'first_name': 'Restricted',
        'last_name': 'User',
        'attributes': {
            'clearance_level': 'low',
            'ai_advice_opted_in': False,
        },
        'role': 'restricted_user',
    },
]


async def create_permit_config():  # noqa: C901
    """Create all required configurations in Permit.io.

    This includes resources, roles, permissions, and user assignments.
    """
    try:
        print('\n=== Starting Permit.io Configuration ===\n')

        # Create resources first
        print('\nCreating resources...')
        for resource in resources:
            try:
                print(f'\nAttempting to create resource: {resource["name"]}')
                print(f'Resource config: {resource}')
                await permit.api.resources.create(resource)
                print(f'✓ Successfully created resource: {resource["name"]}')
            except Exception as e:
                print(f'✗ Failed to create resource {resource["name"]}')
                print(f'Error details: {str(e)}')
                raise

        # Create user attributes
        print('\nCreating user attributes...')
        for attr in user_attributes:
            try:
                print(f'\nAttempting to create user attribute: {attr["key"]}')
                print(f'Attribute config: {attr}')
                await permit.api.resource_attributes.create('__user', attr)

                print(f'✓ Successfully created user attribute: {attr["key"]}')
            except Exception as e:
                print(f'✗ Failed to create user attribute {attr["key"]}')
                print(f'Error details: {str(e)}')
                raise

        # Create roles with simpler permission format
        print('\nCreating roles...')
        for role in roles:
            try:
                print(f'\nAttempting to create role: {role["name"]}')
                # Convert permissions to string format
                permissions = []
                for permission in role.get('permissions', []):
                    for action in permission['actions']:
                        permissions.append(f'{permission["resource"]}:{action}')

                # Create role object with string permissions
                role_obj = {
                    'name': role['name'],
                    'key': role['name'].lower().replace(' ', '_'),
                    'permissions': permissions,
                    'description': f'Role for {role["name"]} with ABAC rules',
                }
                print(f'Role configuration: {role_obj}')
                await permit.api.roles.create(role_obj)
                print(f'✓ Successfully created role: {role["name"]}')
            except Exception as e:
                print(f'✗ Failed to create role {role["name"]}')
                print(f'Error details: {str(e)}')
                raise

        # # Create user sets
        print('\nCreating user sets...')
        for user_set in user_sets:
            try:
                print(f'\nAttempting to create user set: {user_set["name"]}')
                print(f'User set config: {user_set}')
                await permit.api.condition_sets.create(user_set)
                print(f'✓ Successfully created user set: {user_set["name"]}')
            except Exception as e:
                print(f'✗ Failed to create user set {user_set["name"]}')
                print(f'Error details: {str(e)}')
                raise

        # Create resource sets
        print('\nCreating resource sets...')
        for resource_set in resource_sets:
            try:
                print(f'\nAttempting to create resource set: {resource_set["name"]}')
                print(f'Resource set config: {resource_set}')
                await permit.api.condition_sets.create(resource_set)
                print(f'✓ Successfully created resource set: {resource_set["name"]}')
            except Exception as e:
                print(f'✗ Failed to create resource set {resource_set["name"]}')
                print(f'Error details: {str(e)}')
                raise

        # Create condition set rules
        print('\nCreating condition set rules...')
        for rule in condition_set_rules:
            try:
                print('\nAttempting to create condition set rule')
                print(f'Rule config: {rule}')
                await permit.api.condition_set_rules.create(rule)
                print('✓ Successfully created condition set rule')
            except Exception as e:
                print('✗ Failed to create condition set rule')
                print(f'Error details: {str(e)}')
                raise

        # Create users and assign roles last
        print('\nCreating users and assigning roles...')
        for user in example_users:
            try:
                print(f'\nAttempting to create user: {user["email"]}')
                print(f'User config: {user}')
                # Create the user with correct sync format
                await permit.api.users.sync(
                    {
                        'key': user['key'],
                        'email': user['email'],
                        'first_name': user['first_name'],
                        'last_name': user['last_name'],
                        'attributes': user['attributes'],
                    }
                )
                # Assign role to the user
                await permit.api.users.assign_role(
                    {
                        'user': user['key'],
                        'role': user['role'],
                        'tenant': 'default',
                    }
                )
                print(
                    f'✓ Successfully created user: {user["email"]} with role: {user["role"]}'
                )
            except Exception as e:
                print(f'✗ Failed to create/assign role to user {user["email"]}')
                print(f'Error details: {str(e)}')
                raise

        print('\n=== Configuration completed successfully ===\n')

    except Exception as e:
        print('\n✗ Configuration failed')
        print(f'Final error: {str(e)}')
        raise


if __name__ == '__main__':
    asyncio.run(create_permit_config())
