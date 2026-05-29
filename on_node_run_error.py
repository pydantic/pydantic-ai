# on_node_run_error.py
from typing import Callable, Any

async def on_node_run_error(error: Exception, node_name: str) -> None:
    """
    处理节点运行错误。

    参数:
        error (Exception): 发生的异常。注意：类型为 Exception，而非 BaseException，
                           以确保正确处理 asyncio.CancelledError 等异常。
        node_name (str): 发生错误的节点名称。

    返回:
        None
    """
    # 实际处理逻辑，例如记录日志
    print(f"Error in node '{node_name}': {error}")
    # 可以添加更多处理，如重试逻辑
    # 注意：不要捕获 BaseException，以免干扰 asyncio.CancelledError

def get_error_handler() -> Callable[[Exception, str], Any]:
    """
    返回 on_node_run_error 函数，用于错误处理。
    """
    return on_node_run_error