from __future__ import annotations

import contextvars
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

try:
    from opentelemetry import context as otel_context
    from opentelemetry.trace import INVALID_SPAN, Span, get_current_span, set_span_in_context

    T = TypeVar('T')

    @dataclass
    class WithContext(Generic[T]):
        data: T
        span: Span
        token: contextvars.Token[otel_context.Context] | None = None

        @classmethod
        def create(cls, data: T):
            return cls(data, get_current_span())

        def attach(self):
            if self.span is INVALID_SPAN:
                return

            self.token = otel_context.attach(set_span_in_context(self.span))  # type: ignore

        def detach(self):
            if self.token:
                otel_context.detach(self.token)

    @dataclass
    class InstrumentedSendStream(Generic[T]):
        stream: MemoryObjectSendStream[WithContext[T]]

        def send(self, item: T):
            return self.stream.send(WithContext[T].create(item))

        def send_nowait(self, item: T):
            return self.stream.send_nowait(WithContext[T].create(item))

        def __getattr__(self, item: str):
            return getattr(self.stream, item)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_: object):
            await self.aclose()

        def __enter__(self):
            return self

        def __exit__(self, *_: object):
            self.close()

    @dataclass
    class InstrumentedReceiveStream(AsyncIterable[T]):
        stream: MemoryObjectReceiveStream[WithContext[T]]
        current: WithContext[T] | None = None

        async def receive(self):
            return (await self.stream.receive()).data

        def receive_nowait(self):
            return self.stream.receive_nowait().data

        def __aiter__(self):
            return self

        async def __anext__(self) -> T:
            self._detach()
            self.current = await self.stream.__anext__()
            self.current.attach()
            return self.current.data

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_: object):
            await self.aclose()

        def __enter__(self):
            return self

        def __exit__(self, *_: object):
            self.close()

        def close(self):
            self._detach()
            self.stream.close()

        async def aclose(self):
            self.close()

        def _detach(self):
            if self.current:
                self.current.detach()
                self.current = None

        def __getattr__(self, item: str):
            return getattr(self.stream, item)

    def instrument_create_memory_object_stream():
        original = anyio.create_memory_object_stream

        class wrapper(original):  # type: ignore
            def __new__(cls, *args: Any, **kwargs: Any):  # type: ignore
                send, recv = super().__new__(cls, *args, **kwargs)  # type: ignore
                return InstrumentedSendStream(send), InstrumentedReceiveStream(recv)  # type: ignore

        anyio.create_memory_object_stream = wrapper
except ImportError:

    def instrument_create_memory_object_stream(): ...
