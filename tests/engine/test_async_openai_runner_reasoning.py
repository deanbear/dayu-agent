"""AsyncOpenAIRunner reasoning 协议归一化的单元测试。

覆盖三类不变量：
- ``_resolve_reasoning_tag`` 直接委托 ``resolve_reasoning_protocol``，行为等价。
- ``_yield_non_stream_content`` 已收紧为"只发剥离后正文"，不再产 reasoning。
- non-stream 与 SSE 路径在 ``(content_complete.data, metadata.reasoning_content)``
  与事件序列上完全等价（核心协议归一化承诺）。
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List

import pytest

from dayu.engine.async_openai_runner import (
    AsyncOpenAIRunner,
    AsyncOpenAIRunnerRunningConfig,
)
from dayu.engine.events import EventType, StreamEvent
from dayu.engine.sse_parser import SSEStreamParser


class _MockRunner(AsyncOpenAIRunner):
    """绕过 aiohttp 初始化的轻量 runner，用于隔离测试受保护方法。"""

    def __init__(self, running_config: AsyncOpenAIRunnerRunningConfig) -> None:
        self.running_config = running_config
        self.default_extra_payloads: Dict[str, Any] = {}
        self.name = "mock_runner"


@pytest.fixture
def runner() -> _MockRunner:
    """构造无网络依赖的 runner 桩。"""

    return _MockRunner(AsyncOpenAIRunnerRunningConfig())


def test_resolve_reasoning_tag_google_intent(runner: _MockRunner) -> None:
    """``_resolve_reasoning_tag`` 必须严格委托 reasoning_protocol。"""

    assert runner._resolve_reasoning_tag({
        "extra_body": {"google": {"thinking_config": {"include_thoughts": True}}},
    }) == "thought"

    assert runner._resolve_reasoning_tag({
        "extra_body": {"google": {"thinking_config": {"thinking_budget": 5000}}},
    }) is None

    assert runner._resolve_reasoning_tag({
        "extra_body": {"google": {"thinking_config": {"include_thoughts": False}}},
    }) is None

    assert runner._resolve_reasoning_tag({"extra_body": {"google": {}}}) is None


@pytest.mark.asyncio
async def test_yield_non_stream_content_emits_only_stripped_content(
    runner: _MockRunner,
) -> None:
    """``_yield_non_stream_content`` 已收紧为"只发剥离后正文"。

    协议归一化承诺：剥离与 reasoning 合并都在 ``_process_non_stream`` 里完成；
    本方法不再处理 ``<thought>``，对它而言 ``content`` 就是最终正文。
    """

    events: List[StreamEvent] = []
    async for event in runner._yield_non_stream_content("After"):
        events.append(event)

    assert len(events) == 1
    assert events[0].type == EventType.CONTENT_DELTA
    assert events[0].data == "After"


@pytest.mark.asyncio
async def test_yield_non_stream_content_skips_empty(runner: _MockRunner) -> None:
    """空 content 不应产出任何事件（与 SSE 路径"无 delta 不发"对齐）。"""

    events: List[StreamEvent] = []
    async for event in runner._yield_non_stream_content(""):
        events.append(event)
    assert events == []


# --- SSE↔non-stream 协议等价性测试 ----------------------------------


class _RunningConfigStub(AsyncOpenAIRunnerRunningConfig):
    """SSEStreamParser 测试用的最小 running_config 桩，继承真实配置以满足类型约束。"""

    pass


class _ResponseStub:
    """模拟 aiohttp 流式响应的最小桩，按 chunk 列表逐块吐出。"""

    def __init__(self, chunks: List[bytes]) -> None:
        self._chunks = list(chunks)
        self.content = self

    async def iter_chunked(self, _size: int) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


async def _collect(stream: AsyncIterator[StreamEvent]) -> List[StreamEvent]:
    """把异步迭代器收集成 list，方便断言。"""

    out: List[StreamEvent] = []
    async for ev in stream:
        out.append(ev)
    return out


@pytest.mark.asyncio
async def test_sse_and_non_stream_produce_equivalent_reasoning() -> None:
    """同一份 ``<thought>X</thought>Y`` 在 SSE 与 non-stream 上产出等价。

    等价定义：
    - ``CONTENT_DELTA`` 累计文本相同；
    - ``REASONING_DELTA`` 累计文本相同；
    - 最终 ``(content, reasoning_content)`` 元组相同。
    """

    raw_content = "<thought>analyzing</thought>final answer"

    # --- SSE 路径 ---
    parser = SSEStreamParser(
        name="gemini-stream",
        request_id="req_eq_sse",
        running_config=_RunningConfigStub(),
        content_reasoning_tag="thought",
    )
    chunk = json.dumps({"choices": [{"delta": {"content": raw_content}, "finish_reason": "stop"}]})
    response = _ResponseStub([
        f"data: {chunk}\n\n".encode("utf-8"),
        b"data: [DONE]\n\n",
    ])

    sse_events = await _collect(parser.parse_stream(response))  # type: ignore[arg-type]
    sse_result = parser.get_result()

    sse_content = "".join(e.data for e in sse_events if e.type == EventType.CONTENT_DELTA)
    sse_reasoning = "".join(e.data for e in sse_events if e.type == EventType.REASONING_DELTA)

    # --- non-stream 路径 ---
    runner = _MockRunner(AsyncOpenAIRunnerRunningConfig())
    non_stream_payload: Dict[str, Any] = {
        "choices": [{"message": {"content": raw_content}, "finish_reason": "stop"}],
    }
    ns_events = await _collect(runner._process_non_stream(
        non_stream_payload, request_id="req_eq_ns", trace_meta={},
        content_reasoning_tag="thought",
    ))

    ns_content_complete = next(e for e in ns_events if e.type == EventType.CONTENT_COMPLETE)
    ns_reasoning_evt = [e for e in ns_events if e.type == EventType.REASONING_DELTA]
    ns_content_evt = [e for e in ns_events if e.type == EventType.CONTENT_DELTA]

    ns_content = "".join(e.data for e in ns_content_evt)
    ns_reasoning = "".join(e.data for e in ns_reasoning_evt)

    # 等价性核心断言
    assert sse_content == ns_content == "final answer"
    assert sse_reasoning == ns_reasoning == "analyzing"
    assert sse_result.content == ns_content_complete.data == "final answer"
    assert sse_result.reasoning_content == ns_content_complete.metadata["reasoning_content"] == "analyzing"


@pytest.mark.asyncio
async def test_non_stream_merges_native_reasoning_with_extracted() -> None:
    """non-stream 路径必须把从 ``<thought>`` 中剥离出的内容与原生
    ``message.reasoning_content`` 合并到唯一的 reasoning 真源。

    合并顺序与 SSE 路径对齐：``<thought>`` 抽出的片段先入，
    ``reasoning_content`` 字段后追加（``SSEStreamParser._handle_payload``
    的 delta 处理顺序）。同等输入下两条路径必须产出相同的 reasoning 串。
    """

    runner = _MockRunner(AsyncOpenAIRunnerRunningConfig())
    payload: Dict[str, Any] = {
        "choices": [{
            "message": {
                "reasoning_content": "A",
                "content": "<thought>B</thought>C",
            },
            "finish_reason": "stop",
        }],
    }
    events = await _collect(runner._process_non_stream(
        payload, request_id="req_merge", trace_meta={},
        content_reasoning_tag="thought",
    ))

    reasoning_acc = "".join(e.data for e in events if e.type == EventType.REASONING_DELTA)
    content_acc = "".join(e.data for e in events if e.type == EventType.CONTENT_DELTA)
    cc = next(e for e in events if e.type == EventType.CONTENT_COMPLETE)

    assert reasoning_acc == "BA"
    assert content_acc == "C"
    assert cc.data == "C"
    assert cc.metadata["reasoning_content"] == "BA"


@pytest.mark.asyncio
async def test_sse_and_non_stream_merge_order_are_equivalent() -> None:
    """SSE↔non-stream 在 ``<thought>`` 抽出片段 + 原生 reasoning_content
    的合并顺序上必须完全一致。

    构造同一逻辑输入 ``reasoning_content="A"`` + ``content="<thought>B</thought>C"``，
    SSE 路径上同一条 delta 内先消费 ``content`` 抽出 ``B``、再追加
    ``reasoning_content`` 的 ``A``，最终累计为 ``"BA"``；non-stream 必须等价。
    """

    # --- SSE 路径 ---
    parser = SSEStreamParser(
        name="merge-stream",
        request_id="req_merge_sse",
        running_config=_RunningConfigStub(),
        content_reasoning_tag="thought",
    )
    chunk = json.dumps({
        "choices": [{
            "delta": {
                "content": "<thought>B</thought>C",
                "reasoning_content": "A",
            },
            "finish_reason": "stop",
        }],
    })
    response = _ResponseStub([
        f"data: {chunk}\n\n".encode("utf-8"),
        b"data: [DONE]\n\n",
    ])
    sse_events = await _collect(parser.parse_stream(response))  # type: ignore[arg-type]
    sse_result = parser.get_result()
    sse_reasoning = "".join(e.data for e in sse_events if e.type == EventType.REASONING_DELTA)
    sse_content = "".join(e.data for e in sse_events if e.type == EventType.CONTENT_DELTA)

    # --- non-stream 路径 ---
    runner = _MockRunner(AsyncOpenAIRunnerRunningConfig())
    payload: Dict[str, Any] = {
        "choices": [{
            "message": {
                "reasoning_content": "A",
                "content": "<thought>B</thought>C",
            },
            "finish_reason": "stop",
        }],
    }
    ns_events = await _collect(runner._process_non_stream(
        payload, request_id="req_merge_ns", trace_meta={},
        content_reasoning_tag="thought",
    ))
    ns_reasoning = "".join(e.data for e in ns_events if e.type == EventType.REASONING_DELTA)
    ns_content = "".join(e.data for e in ns_events if e.type == EventType.CONTENT_DELTA)
    ns_cc = next(e for e in ns_events if e.type == EventType.CONTENT_COMPLETE)

    # 等价性核心断言：合并顺序、累计文本、(content, reasoning_content) 元组完全一致
    assert sse_reasoning == ns_reasoning == "BA"
    assert sse_content == ns_content == "C"
    assert sse_result.content == ns_cc.data == "C"
    assert sse_result.reasoning_content == ns_cc.metadata["reasoning_content"] == "BA"
