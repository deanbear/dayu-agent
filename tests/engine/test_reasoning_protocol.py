"""测试 dayu.engine.reasoning_protocol 协议适配层。

覆盖三类边界：
- Google Gemini ``thinking_config.include_thoughts=True`` 命中。
- 各种未命中情形（无 extra_body / 无 google / include_thoughts=False / 类型异常）。
- ``ReasoningProtocolHook.is_enabled`` 派生属性的语义。
"""

from __future__ import annotations

from dayu.engine.reasoning_protocol import (
    EMPTY_REASONING_HOOK,
    ReasoningProtocolHook,
    resolve_reasoning_protocol,
)


def test_resolve_returns_thought_hook_when_google_thinking_enabled() -> None:
    """命中 Google thinking_config 协议时，返回 tag_name='thought'。"""
    hook = resolve_reasoning_protocol({
        "extra_body": {"google": {"thinking_config": {"include_thoughts": True}}},
    })
    assert hook.tag_name == "thought"
    assert hook.is_enabled is True


def test_resolve_returns_empty_hook_when_no_extra_body() -> None:
    """payload 完全没有 extra_body 时，返回空 hook。"""
    hook = resolve_reasoning_protocol({"model": "gpt-4o"})
    assert hook is EMPTY_REASONING_HOOK
    assert hook.tag_name is None
    assert hook.is_enabled is False


def test_resolve_returns_empty_hook_when_include_thoughts_false() -> None:
    """include_thoughts 显式为 False 时，不应激活协议。"""
    hook = resolve_reasoning_protocol({
        "extra_body": {"google": {"thinking_config": {"include_thoughts": False}}},
    })
    assert hook.tag_name is None


def test_resolve_returns_empty_hook_when_include_thoughts_truthy_but_not_true() -> None:
    """include_thoughts 必须严格是布尔 True；非 True 真值不激活。

    防御 vendor payload 把 ``"true"`` 字符串误填进来——detector 只认布尔。
    """
    hook = resolve_reasoning_protocol({
        "extra_body": {"google": {"thinking_config": {"include_thoughts": "true"}}},
    })
    assert hook.tag_name is None


def test_resolve_returns_empty_hook_when_extra_body_not_mapping() -> None:
    """extra_body 不是 mapping 时安全降级，不抛异常。"""
    hook = resolve_reasoning_protocol({"extra_body": "not-a-dict"})
    assert hook.tag_name is None


def test_resolve_returns_empty_hook_when_thinking_config_missing() -> None:
    """google 块存在但缺 thinking_config 时返回空 hook。"""
    hook = resolve_reasoning_protocol({
        "extra_body": {"google": {"safety_settings": []}},
    })
    assert hook.tag_name is None


def test_empty_reasoning_hook_singleton_is_disabled() -> None:
    """模块级 EMPTY_REASONING_HOOK 必须是不可变、未启用状态。"""
    assert EMPTY_REASONING_HOOK.tag_name is None
    assert EMPTY_REASONING_HOOK.is_enabled is False


def test_reasoning_protocol_hook_is_frozen() -> None:
    """ReasoningProtocolHook 是 frozen dataclass，禁止运行时改字段。"""
    hook = ReasoningProtocolHook(tag_name="thought")
    try:
        hook.tag_name = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ReasoningProtocolHook 必须是 frozen 的")
