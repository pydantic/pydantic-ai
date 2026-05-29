# test_on_node_run_error.py
import asyncio
from on_node_run_error import on_node_run_error

def test_on_node_run_error():
    """测试 on_node_run_error 函数"""
    # 测试普通异常
    try:
        raise ValueError("测试错误")
    except ValueError as e:
        asyncio.run(on_node_run_error(e, "test_node"))
        print("普通异常处理成功")

    # 测试 asyncio.CancelledError
    async def test_cancelled():
        try:
            raise asyncio.CancelledError()
        except asyncio.CancelledError as e:
            await on_node_run_error(e, "cancelled_node")
            print("CancelledError 处理成功")

    asyncio.run(test_cancelled())

if __name__ == "__main__":
    test_on_node_run_error()