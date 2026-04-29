"""``AsyncAgent`` 对 vendor 私有 reasoning 协议无感的端到端不变量测试。

PR 91 的回归说明：``<thought>`` 等 vendor 私有 reasoning 表达必须完全封装在
Runner 协议适配层内（``SSEStreamParser`` + ``reasoning_protocol`` +
``xml_extractor`` + ``AsyncOpenAIRunner._process_non_stream``），跨过 Runner
边界后，``CONTENT_*`` / ``FINAL_ANSWER`` / 下一轮 assistant message / Host
持久化 / 压缩链路统一只看到剥离后的正文。

本文件锁定如下三条不变量：

- ``test_final_answer_strips_reasoning_tag``：流式 Runner 已剥离的
  ``CONTENT_DELTA`` 累计 → ``FINAL_ANSWER.data["content"]`` 不含 ``<thought>``。
- ``test_assistant_message_for_next_turn_has_no_thought``：tool-call 路径下
  下一轮 assistant message 的 ``content`` 字段必须是剥离版。
- ``test_async_agent_source_is_thought_agnostic``：grep 守护测试，确保
  ``async_agent.py`` 源码不出现 ``<thought>`` / ``thought`` 字面量
  （即生产代码不依赖 vendor 协议字面量）。
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, AsyncIterator, Dict, Iterable, List

import pytest

from dayu.contracts.agent_types import AgentMessage
from dayu.engine import (
    AsyncAgent,
    EventType,
    content_complete,
    content_delta,
    done_event,
    tool_call_dispatched,
    tool_call_result,
    tool_calls_batch_done,
)
from dayu.engine.events import StreamEvent


class _StripDummyRunner:
    """模拟"已经在 Runner 边界内完成 ``<thought>`` 剥离"的 runner 桩。

    与生产语义一致：``CONTENT_DELTA`` 只承载剥离后的正文；
    ``CONTENT_COMPLETE.data`` 同样是剥离后的正文，``metadata['reasoning_content']``
    才是合并后的推理。
    """

    def __init__(self, batches: Iterable[List[StreamEvent]], supports_tools: bool = False) -> None:
        self.batches: List[List[StreamEvent]] = [list(b) for b in batches]
        self.calls: List[Dict[str, Any]] = []
        self._supports_tools = supports_tools

    def is_supports_tool_calling(self) -> bool:
        """是否支持 tool calling，由构造参数控制。"""

        return self._supports_tools

    def set_tools(self, *_args: Any, **_kwargs: Any) -> None:
        """no-op，保持 Runner 协议一致。"""

        return None

    async def close(self) -> None:
        """no-op，保持 Runner 协议一致。"""

        return None

    async def call(
        self, messages: List[AgentMessage], *, stream: bool = True, **extra_payloads: Any
    ) -> AsyncIterator[StreamEvent]:
        """按调用顺序吐出预设事件批次。"""

        self.calls.append({"messages": list(messages), "stream": stream, "extra_payloads": extra_payloads})
        batch = self.batches.pop(0)
        for event in batch:
            yield event


@pytest.mark.asyncio
async def test_final_answer_strips_reasoning_tag() -> None:
    """Runner 已剥离的 stream → ``FINAL_ANSWER.data['content']`` 不含 ``<thought>``。

    模拟 Runner 协议适配层完成 ``<thought>X</thought>Y`` 的剥离后，对外只发
    ``CONTENT_DELTA('Y')`` + ``CONTENT_COMPLETE('Y', reasoning_content='X')``，
    ``AsyncAgent`` 据此装配的 ``FINAL_ANSWER.content`` 必须是 ``"Y"``。
    """

    runner = _StripDummyRunner([
        [
            content_delta("Y"),
            content_complete("Y", reasoning_content="X"),
            done_event(),
        ],
    ])
    agent = AsyncAgent(runner)

    events: List[StreamEvent] = []
    async for event in agent.run("anything"):
        events.append(event)

    final = next(e for e in events if e.type == EventType.FINAL_ANSWER)
    assert isinstance(final.data, dict)
    assert final.data["content"] == "Y"
    assert "<thought>" not in final.data["content"]
    assert "</thought>" not in final.data["content"]


@pytest.mark.asyncio
async def test_assistant_message_for_next_turn_has_no_thought() -> None:
    """tool-call 路径下，下一轮 assistant message 的 ``content`` 必须剥离干净。"""

    tool_args = {"path": "test.txt"}
    runner = _StripDummyRunner([
        [
            content_delta("Y"),
            tool_call_dispatched("call_1", "tool", tool_args, index_in_iteration=0),
            tool_call_result(
                "call_1",
                {"ok": True, "value": "ok"},
                name="tool",
                arguments=tool_args,
                index_in_iteration=0,
            ),
            tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
            content_complete("Y", reasoning_content="X"),
            done_event(),
        ],
        [content_delta("done"), content_complete("done"), done_event()],
    ])
    agent = AsyncAgent(runner)

    async for _event in agent.run("anything"):
        pass

    assert len(runner.calls) == 2
    second_messages = runner.calls[1]["messages"]
    assistant_message = next(m for m in second_messages if m.get("role") == "assistant")

    content_field = assistant_message.get("content", "")
    assert "<thought>" not in content_field
    assert "</thought>" not in content_field
    # 同时核对正向期望：assistant.content 就是剥离版正文
    assert content_field == "Y"


def test_async_agent_source_is_thought_agnostic() -> None:
    """守护测试：``dayu/engine/async_agent.py`` 源码不应出现 vendor 协议字面量。

    这里禁止的是明确的 vendor 协议 token，而不是任意包含 ``thought``
    子串的英文注释/标识符，避免把无关文字误报成协议泄漏。
    """

    project_root = Path(__file__).resolve().parents[2]
    source_path = project_root / "dayu" / "engine" / "async_agent.py"
    source = source_path.read_text(encoding="utf-8")
    assert re.search(r"<\s*/?\s*thought\b", source, re.IGNORECASE) is None, (
        "AsyncAgent 不应出现 <thought> vendor 标签字面量"
    )
    assert "thought_signature" not in source, (
        "AsyncAgent 不应出现 thought_signature vendor 协议字段"
    )
