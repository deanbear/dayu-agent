import pytest
from dayu.engine.async_openai_runner import AsyncOpenAIRunner, AsyncOpenAIRunnerRunningConfig
from dayu.engine.events import EventType

class _MockRunner(AsyncOpenAIRunner):
    """用于测试受保护方法的 Mock Runner"""
    def __init__(self, running_config: AsyncOpenAIRunnerRunningConfig):
        # 绕过复杂的 aiohttp 初始化
        self.running_config = running_config
        self.default_extra_payloads = {}
        self.name = "mock_runner"

@pytest.fixture
def runner():
    config = AsyncOpenAIRunnerRunningConfig()
    return _MockRunner(config)

def test_resolve_reasoning_tag_google_intent(runner):
    """测试 Google 思考配置的自动探测"""
    # 场景 1: 开启 include_thoughts
    payloads = {
        "extra_body": {
            "google": {
                "thinking_config": {
                    "include_thoughts": True,
                    "thinking_budget": 1024
                }
            }
        }
    }
    assert runner._resolve_reasoning_tag(payloads) == "thought"

    # 场景 2: 仅开启 thinking_budget (不应触发提取)
    payloads = {
        "extra_body": {
            "google": {
                "thinking_config": {
                    "thinking_budget": 5000
                }
            }
        }
    }
    assert runner._resolve_reasoning_tag(payloads) is None

    # 场景 3: include_thoughts 为 False，但有 budget (不应触发提取)
    payloads = {
        "extra_body": {
            "google": {
                "thinking_config": {
                    "include_thoughts": False,
                    "thinking_budget": 5000
                }
            }
        }
    }
    assert runner._resolve_reasoning_tag(payloads) is None

    # 场景 4: 未开启任何相关配置
    payloads = {"extra_body": {"google": {}}}
    assert runner._resolve_reasoning_tag(payloads) is None

@pytest.mark.asyncio
async def test_yield_non_stream_content_extraction(runner):
    """测试非流式路径下的内容拆分与元数据注入"""
    content = "<thought>Thinking process</thought>After"
    trace_meta = {"run_id": "test_run_123"}

    # 场景 1: 开启提取并注入元数据
    events = []
    async for event in runner._yield_non_stream_content(content, tag_name="thought"):
        # 模拟 Runner 外部调用的标注过程
        events.append(runner._annotate_event(event, trace_meta))

    assert len(events) == 2
    assert events[0].type == EventType.REASONING_DELTA
    assert events[0].data == "Thinking process"
    assert events[0].metadata["run_id"] == "test_run_123"

    assert events[1].type == EventType.CONTENT_DELTA
    assert events[1].data == "After"
    assert events[1].metadata["run_id"] == "test_run_123"

    # 场景 2: 关闭提取
    events = []
    async for event in runner._yield_non_stream_content(content, tag_name=None):
        events.append(event)

    assert len(events) == 1
    assert events[0].type == EventType.CONTENT_DELTA
    assert events[0].data == content
