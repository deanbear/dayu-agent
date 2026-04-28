"""
轻量级 XML 标签流式提取器

用于在模型增量输出中拦截特定的 XML 标签（如 <thought>），
将其内容分离并安全提取，而无须等待整个数据流结束。
"""

from typing import List, Tuple


class StreamingXMLTagExtractor:
    """
    轻量级的流式 XML 标签提取器状态机。

    用于在增量字符流中拦截特定的 XML 标签（如 <thought>），
    并将其内容与普通正文分离开来。这是一个无副作用的纯函数引擎。
    """

    def __init__(self, tag_name: str, start_only: bool = True, enabled: bool = True):
        """
        Args:
            tag_name: 需要提取的 XML 标签名，例如 "thought"。
            start_only: 安全锁开关，默认为 True。
                        如果为 True，则只提取出现在回复绝对开头（允许有前置空格/换行）的标签。
                        一旦在标签外部出现了任何实质性正文（非空白字符），提取器将永久失活并放行后续所有内容。
            enabled: 是否启用提取器。如果为 False，则 process() 将直接原样返回输入，不做任何处理。
        """
        self.open_tag = f"<{tag_name}>"
        self.close_tag = f"</{tag_name}>"
        self._buffer = ""
        self._in_tag = False

        self.enabled = enabled
        self.start_only = start_only
        self._is_active = enabled  # 如果未启用，则初始化即失活
        self._has_seen_non_whitespace = False

    def process(self, chunk: str) -> List[Tuple[str, bool]]:
        """处理一段文本块，返回解析后的 (text, is_thought) 元组列表。

        Args:
            chunk: 输入的增量文本片段。

        Returns:
            List[Tuple[str, bool]]: 解析后的元组列表。
                                    每个元组为 (内容文本, 是否属于目标标签内部)。

        Raises:
            无。
        """
        if not self.enabled:
            return [(chunk, False)]

        if not self._is_active:
            return [(chunk, False)] if chunk else []

        self._buffer += chunk
        results: List[Tuple[str, bool]] = []

        while self._buffer:
            # 1. 检查提取器失活 (安全锁)
            if self.start_only and not self._in_tag and self._is_active:
                first_lt = self._buffer.find("<")
                current_pre_text_has_content = False
                if first_lt == -1:
                    current_pre_text_has_content = bool(self._buffer.strip())
                elif first_lt > 0:
                    current_pre_text_has_content = bool(self._buffer[:first_lt].strip())

                if self._has_seen_non_whitespace or current_pre_text_has_content:
                    self._is_active = False
                    results.append((self._buffer, False))
                    self._buffer = ""
                    break

            # 2. 正常提取逻辑
            target = self.close_tag if self._in_tag else self.open_tag
            idx = self._buffer.find(target)

            if idx != -1:
                # 找到了完整的开/闭标签
                if idx > 0:
                    out_text = self._buffer[:idx]
                    results.append((out_text, self._in_tag))
                    if not self._in_tag and out_text.strip():
                        self._has_seen_non_whitespace = True
                # 越过标签本身
                self._buffer = self._buffer[idx + len(target) :]
                self._in_tag = not self._in_tag
            else:
                # 没找到完整标签，检查是否存在可能的跨 chunk 标签前缀被截断
                # 例如 target 是 "<thought>"，当前 buffer 结尾恰好是 "<thou"
                partial_match_len = 0
                for i in range(len(target) - 1, 0, -1):
                    if self._buffer.endswith(target[:i]):
                        partial_match_len = i
                        break

                if partial_match_len > 0:
                    # 发现存在部分前缀，将前缀之前的安全部分切出返回，保留前缀在 buffer 中等待下文
                    safe_part = self._buffer[:-partial_match_len]
                    if safe_part:
                        results.append((safe_part, self._in_tag))
                        if not self._in_tag and safe_part.strip():
                            self._has_seen_non_whitespace = True
                    self._buffer = self._buffer[-partial_match_len:]
                    break  # 停止循环，等待下一个 chunk
                else:
                    # 末尾没有任何标签的可能前缀，全部安全输出
                    results.append((self._buffer, self._in_tag))
                    if not self._in_tag and self._buffer.strip():
                        self._has_seen_non_whitespace = True
                    self._buffer = ""

        return results

    def flush(self) -> List[Tuple[str, bool]]:
        """
        在流结束（或接收完毕）时调用，强制输出缓冲中残留的片段。

        Returns:
            List[Tuple[str, bool]]: 残留的提取结果。
        """
        if not self._buffer:
            return []
        res = [(self._buffer, self._in_tag)]
        self._buffer = ""
        # 流结束时重置状态
        self._in_tag = False
        return res
