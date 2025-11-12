"""Type guards for OpenAI Responses protocol types."""

from typing import Any, TypeGuard


def _is_input_text_item(item: Any) -> TypeGuard[dict[str, Any]]:
    """Check if item is an input_text type."""
    return isinstance(item, dict) and item.get('type') == 'input_text'


def _is_input_image_item(item: Any) -> TypeGuard[dict[str, Any]]:
    """Check if item is an input_image type."""
    return isinstance(item, dict) and item.get('type') == 'input_image'


def _is_input_file_item(item: Any) -> TypeGuard[dict[str, Any]]:
    """Check if item is an input_file type."""
    return isinstance(item, dict) and item.get('type') == 'input_file'


def _is_message_item(item: Any) -> TypeGuard[dict[str, Any]]:
    """Check if item is a message type (or implicit message with role/content)."""
    if not isinstance(item, dict):
        return False
    item_type = item.get('type')
    # Explicit message type or implicit (has role and content)
    return item_type == 'message' or (item_type is None and 'role' in item and 'content' in item)


def _is_function_call_output_item(item: Any) -> TypeGuard[dict[str, Any]]:
    """Check if item is a function_call_output type."""
    return isinstance(item, dict) and item.get('type') == 'function_call_output'


def _is_system_role(message: dict[str, Any]) -> TypeGuard[dict[str, Any]]:
    """Check if message has system or developer role."""
    role = message.get('role')
    return role in ('system', 'developer')


def _is_user_role(message: dict[str, Any]) -> TypeGuard[dict[str, Any]]:
    """Check if message has user role."""
    return message.get('role') == 'user'


def _is_assistant_role(message: dict[str, Any]) -> TypeGuard[dict[str, Any]]:
    """Check if message has assistant role."""
    return message.get('role') == 'assistant'


def _is_text_content_part(part: Any) -> TypeGuard[dict[str, Any]]:
    """Check if content part is text type."""
    if not isinstance(part, dict):
        return False
    ptype = part.get('type')
    return ptype in ('input_text', 'text', 'output_text')
