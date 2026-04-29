"""
Reasoning 协议适配层。

职责：
- 探测请求 payload 中 vendor 私有的"推理表达"协议（例如 Google Gemini
  在 `extra_body.google.thinking_config.include_thoughts=True` 时把推理内容
  以 ``<thought>...</thought>`` 嵌入正文流），并把它归一化为 Engine 内部统一的
  reasoning 抽象（`REASONING_DELTA` 事件 + `reasoning_content`）。
- 协议探测以注册表形式承载，未来新增 provider 时只追加 detector，不修改
  Runner 主路径。
- 跨过 Runner 边界后，vendor 私有 reasoning 表达不再存在；上层
  (`AsyncAgent` / `Host` / `Service` / `UI`) 只看到 `CONTENT_*`（剥离后的正文）
  与 `REASONING_*` / `reasoning_content`（合并后的推理）。

模块边界：
- 本模块只解析"请求 payload"，不感知响应；响应侧的标签剥离由
  `dayu.engine.xml_extractor` 完成，本模块只负责告诉调用方"用什么标签名"。
- 不向上层暴露 `Optional`：`resolve_reasoning_protocol` 总是返回一个具体的
  `ReasoningProtocolHook`，未命中时返回 `EMPTY_REASONING_HOOK`（`tag_name is None`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping


@dataclass(frozen=True)
class ReasoningProtocolHook:
    """单次请求的 reasoning 协议归一化配置。

    Attributes:
        tag_name: 需要在响应正文流中拦截剥离的 vendor 私有 XML 标签名；
            None 表示当前请求未启用任何 vendor 私有 reasoning 协议，
            响应正文不需要做标签剥离。
    """

    tag_name: str | None = None

    @property
    def is_enabled(self) -> bool:
        """是否启用了 vendor 私有 reasoning 协议剥离。"""

        return self.tag_name is not None


EMPTY_REASONING_HOOK: ReasoningProtocolHook = ReasoningProtocolHook(tag_name=None)
"""未命中任何 vendor 私有 reasoning 协议时的占位 hook。"""


_PayloadDetector = Callable[[Mapping[str, Any]], ReasoningProtocolHook | None]


def _detect_google_thinking(payload: Mapping[str, Any]) -> ReasoningProtocolHook | None:
    """探测 Google Gemini 的 ``thinking_config.include_thoughts`` 协议。

    Gemini 在 OpenAI 兼容模式下，若请求显式声明
    ``extra_body.google.thinking_config.include_thoughts=True``，
    会在响应正文流中以 ``<thought>...</thought>`` 形式嵌入推理内容。

    Args:
        payload: Runner 即将发出的完整请求 payload。

    Returns:
        命中时返回 `ReasoningProtocolHook(tag_name="thought")`；
        未命中返回 ``None``。

    Raises:
        无。
    """

    extra_body = payload.get("extra_body")
    if not isinstance(extra_body, Mapping):
        return None
    google_block = extra_body.get("google")
    if not isinstance(google_block, Mapping):
        return None
    thinking_config = google_block.get("thinking_config")
    if not isinstance(thinking_config, Mapping):
        return None
    if thinking_config.get("include_thoughts") is True:
        return ReasoningProtocolHook(tag_name="thought")
    return None


_PROTOCOL_DETECTORS: List[_PayloadDetector] = [
    _detect_google_thinking,
]
"""注册表：按顺序尝试每个 detector，第一个命中即返回。

新增 provider 时在此追加 detector，不要在 Runner 主路径加分支。
"""


def resolve_reasoning_protocol(payload: Mapping[str, Any]) -> ReasoningProtocolHook:
    """根据请求 payload 解析 reasoning 协议归一化配置。

    Args:
        payload: Runner 即将发出的完整请求 payload。

    Returns:
        命中任一 vendor 私有协议时返回对应 `ReasoningProtocolHook`；
        未命中返回 `EMPTY_REASONING_HOOK`（`tag_name is None`）。

    Raises:
        无。
    """

    for detector in _PROTOCOL_DETECTORS:
        hook = detector(payload)
        if hook is not None:
            return hook
    return EMPTY_REASONING_HOOK


__all__ = [
    "EMPTY_REASONING_HOOK",
    "ReasoningProtocolHook",
    "resolve_reasoning_protocol",
]
