"""LLM 集成模块 - 摘要生成、标签提取、向量嵌入"""

from __future__ import annotations

import json
import logging
import time

from openai import OpenAI

from .config import Settings

logger = logging.getLogger(__name__)

# ── Prompt 模板 ──────────────────────────────────────────

SUMMARY_PROMPT = """你是一个专业的文档分析助手。请为以下文档生成简洁的中文摘要。

要求：
1. 摘要长度控制在 100-200 字
2. 准确概括文档的核心内容和关键信息
3. 使用客观、简洁的语言
4. 如果文档内容不足，返回"内容不足，无法生成摘要"

文档标题：{title}

文档内容：
{content}

请直接输出摘要，不要添加额外说明。"""

TAGS_PROMPT = """你是一个专业的文档分类助手。请为以下文档生成 3-8 个标签。

要求：
1. 标签应反映文档的主题、领域、关键概念
2. 每个标签 2-6 个字
3. 标签应具有区分度，避免过于宽泛（如"文档"、"笔记"）
4. 使用 JSON 数组格式输出，例如：["机器学习", "深度学习", "PyTorch"]

文档标题：{title}

文档内容：
{content}

请直接输出 JSON 数组，不要添加额外说明。"""


class LLMClient:
    """OpenAI 兼容 API 客户端"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_request_timeout,
        )

    def generate_summary(self, title: str, content: str) -> str:
        """
        为文档生成摘要。

        Args:
            title: 文档标题
            content: 文档内容（已截断）

        Returns:
            生成的摘要文本
        """
        prompt = SUMMARY_PROMPT.format(title=title, content=content)
        return self._chat_completion(prompt)

    def generate_tags(self, title: str, content: str) -> list[str]:
        """
        为文档生成标签。

        Args:
            title: 文档标题
            content: 文档内容（已截断）

        Returns:
            标签列表
        """
        prompt = TAGS_PROMPT.format(title=title, content=content)
        response = self._chat_completion(prompt)

        # 解析 JSON 数组
        try:
            # 尝试直接解析
            tags = json.loads(response)
            if isinstance(tags, list):
                return [str(t).strip() for t in tags if str(t).strip()]
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块中提取
        if "```" in response:
            code_block = response.split("```")[1]
            # 移除可能的 json 标记
            for line in code_block.split("\n"):
                line = line.strip()
                if not line or line.startswith("json"):
                    continue
                try:
                    tags = json.loads(line if line == code_block.strip() else code_block.strip())
                    if isinstance(tags, list):
                        return [str(t).strip() for t in tags if str(t).strip()]
                except json.JSONDecodeError:
                    continue

        # 降级：按逗号/换行分割
        tags = [t.strip().lstrip("- ").strip("'\"") for t in response.replace("\n", ",").split(",")]
        return [t for t in tags if t and len(t) <= 20]

    def embed_text(self, text: str) -> list[float]:
        """
        生成文本的向量嵌入。

        Args:
            text: 输入文本

        Returns:
            向量嵌入列表
        """
        for attempt in range(self._settings.max_retries):
            try:
                response = self._client.embeddings.create(
                    model=self._settings.llm_embedding_model,
                    input=text,
                )
                return response.data[0].embedding

            except Exception as e:
                if attempt < self._settings.max_retries - 1:
                    delay = self._settings.retry_delay * (2**attempt)
                    logger.warning("嵌入生成失败，%0.1f 秒后重试: %s", delay, e)
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"嵌入生成失败: {e}") from e

        raise RuntimeError("嵌入生成失败：未知错误")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        批量生成向量嵌入。

        Args:
            texts: 文本列表

        Returns:
            向量嵌入列表
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        batch_size = 20  # OpenAI 批量嵌入限制

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(self._settings.max_retries):
                try:
                    response = self._client.embeddings.create(
                        model=self._settings.llm_embedding_model,
                        input=batch,
                    )
                    # 按索引排序确保顺序正确
                    sorted_data = sorted(response.data, key=lambda x: x.index)
                    all_embeddings.extend([d.embedding for d in sorted_data])
                    break
                except Exception as e:
                    if attempt < self._settings.max_retries - 1:
                        delay = self._settings.retry_delay * (2**attempt)
                        logger.warning("批量嵌入失败，%0.1f 秒后重试: %s", delay, e)
                        time.sleep(delay)
                    else:
                        raise RuntimeError(f"批量嵌入生成失败: {e}") from e

        return all_embeddings

    def _chat_completion(self, prompt: str) -> str:
        """调用聊天补全 API"""
        for attempt in range(self._settings.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._settings.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是一个专业的文档分析助手，擅长提取关键信息"
                                "和生成标签。请严格按照要求格式输出。"
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self._settings.llm_max_tokens,
                    temperature=self._settings.llm_temperature,
                )
                content = response.choices[0].message.content
                return (content or "").strip()

            except Exception as e:
                if attempt < self._settings.max_retries - 1:
                    delay = self._settings.retry_delay * (2**attempt)
                    logger.warning("LLM 调用失败，%0.1f 秒后重试: %s", delay, e)
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"LLM 调用失败: {e}") from e

        raise RuntimeError("LLM 调用失败：未知错误")
