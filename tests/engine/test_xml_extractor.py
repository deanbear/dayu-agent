
from dayu.engine.xml_extractor import StreamingXMLTagExtractor


def test_extractor_basic():
    """基础测试：单次喂入带有完整标签的字符串。"""
    extractor = StreamingXMLTagExtractor("thought", start_only=False)
    text = "Hello <thought>thinking process</thought> World"

    res = extractor.process(text) + extractor.flush()
    assert res == [
        ("Hello ", False),
        ("thinking process", True),
        (" World", False),
    ]


def test_extractor_no_tags():
    """测试：没有任何标签。"""
    extractor = StreamingXMLTagExtractor("thought", start_only=False)
    text = "Just a normal string without any tags."

    res = extractor.process(text) + extractor.flush()
    assert res == [(text, False)]


def test_extractor_only_open_tag():
    """测试：只有开标签，未闭合。"""
    extractor = StreamingXMLTagExtractor("thought", start_only=False)
    text = "Start <thought>thinking..."

    res = extractor.process(text) + extractor.flush()
    assert res == [
        ("Start ", False),
        ("thinking...", True),
    ]


def test_extractor_streaming_fragmented():
    """测试：标签被极度碎片化截断，跨多个 chunk 到达。"""
    extractor = StreamingXMLTagExtractor("thought", start_only=False)
    chunks = [
        "Hel",
        "lo <",
        "thou",
        "ght>think",
        "ing</",
        "t",
        "hought> Wo",
        "rld",
    ]

    all_res = []
    for chunk in chunks:
        all_res.extend(extractor.process(chunk))
    all_res.extend(extractor.flush())

    # 手动整理合并连续的相同状态文本以方便断言（实际提取器不保证合并，它按 chunk 输出）
    def merge_results(results):
        merged = []
        for text, is_tag in results:
            if not text:
                continue
            if merged and merged[-1][1] == is_tag:
                merged[-1] = (merged[-1][0] + text, is_tag)
            else:
                merged.append((text, is_tag))
        return merged

    merged_res = merge_results(all_res)
    assert merged_res == [
        ("Hello ", False),
        ("thinking", True),
        (" World", False),
    ]


def test_extractor_partial_prefix_flushed():
    """测试：流突然结束，而 buffer 里残留的是可能为标签前缀的内容。"""
    extractor = StreamingXMLTagExtractor("thought", start_only=False)
    text = "Hello <thou"

    # 处理时由于可能存在开标签，"<thou" 会被卡在 buffer 里
    res1 = extractor.process(text)
    assert res1 == [("Hello ", False)]

    # 强制 flush，剩余前缀被安全释放为 False
    res2 = extractor.flush()
    assert res2 == [("<thou", False)]


def test_extractor_consecutive_tags():
    """测试：连续多个相同的标签。"""
    extractor = StreamingXMLTagExtractor("thought", start_only=False)
    text = "A<thought>B</thought>C<thought>D</thought>E"

    res = extractor.process(text) + extractor.flush()
    # 过滤掉空字符串输出
    res = [(t, is_tag) for t, is_tag in res if t]
    assert res == [
        ("A", False),
        ("B", True),
        ("C", False),
        ("D", True),
        ("E", False),
    ]


def test_extractor_start_only_deactivates_on_normal_text():
    """测试：如果是 start_only 模式，开头的正常文本会让提取器永久失活。"""
    extractor = StreamingXMLTagExtractor("thought", start_only=True)
    text = "Here is an example: <thought>this is not a real thought</thought>"

    res = extractor.process(text) + extractor.flush()
    # 因为前面有 "Here is an example: "，提取器失活，后续标签作为普通文本透传
    assert res == [(text, False)]

def test_extractor_start_only_with_leading_whitespace():
    """测试：如果是 start_only 模式，前导空白不会让提取器失活。"""
    extractor = StreamingXMLTagExtractor("thought", start_only=True)
    text = "   \n  <thought>real thought</thought>"

    res = extractor.process(text) + extractor.flush()
    assert res == [
        ("   \n  ", False),
        ("real thought", True),
    ]

def test_extractor_start_only_stuttering_tag():
    """测试：流式输出中，提前遇到一部分非正文，然后遇到标签。"""
    extractor = StreamingXMLTagExtractor("thought", start_only=True)
    res = []
    res.extend(extractor.process("<t"))
    res.extend(extractor.process("<thought>real thought</thought>"))
    res.extend(extractor.flush())

    # 手动合并相同类型的连续结果
    def merge(results):
        merged = []
        for text, is_tag in results:
            if not text:
                continue
            if merged and merged[-1][1] == is_tag:
                merged[-1] = (merged[-1][0] + text, is_tag)
            else:
                merged.append((text, is_tag))
        return merged

    assert merge(res) == [
        ("<t", False),
        ("real thought", True),
    ]
