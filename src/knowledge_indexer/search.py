"""搜索引擎 - 支持自然语言查询"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np

from .config import Settings
from .llm import LLMClient
from .models import DocumentMetadata, SearchResult

logger = logging.getLogger(__name__)


class SearchEngine:
    """基于 FAISS 向量索引的搜索引擎"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm: LLMClient | None = None
        self._index = None
        self._id_map: dict[int, str] = {}
        self._metadata_map: dict[str, DocumentMetadata] = {}
        self._load_index()

    @property
    def _llm_client(self) -> LLMClient:
        """延迟初始化 LLM 客户端，避免在仅需加载索引时连接 API"""
        if self._llm is None:
            self._llm = LLMClient(self._settings)
        return self._llm

    def _load_index(self) -> None:
        """加载 FAISS 索引和元数据"""
        import faiss

        index_path = self._settings.index_path

        if not index_path.exists():
            logger.warning("索引文件不存在: %s", index_path)
            return

        # 加载 FAISS 索引
        self._index = faiss.read_index(str(index_path))
        # 维度一致性检查
        if self._index.d != self._settings.embedding_dim:
            logger.warning(
                "FAISS 索引维度 (%d) 与配置维度 (%d) 不一致，"
                "索引不可用。请运行 scan --force 重建索引。",
                self._index.d,
                self._settings.embedding_dim,
            )
            self._index = None
            return
        logger.info("FAISS 索引已加载，共 %d 个向量", self._index.ntotal)

        # 加载 ID 映射
        id_map_path = self._settings.data_dir / "id_map.json"
        if id_map_path.exists():
            try:
                data = json.loads(id_map_path.read_text(encoding="utf-8"))
                self._id_map = {int(k): v for k, v in data.items()}
            except Exception as e:
                logger.error("加载 ID 映射失败: %s", e)

        # 加载元数据
        metadata_path = self._settings.metadata_path
        if metadata_path.exists():
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
                self._metadata_map = {
                    k: DocumentMetadata.model_validate(v) for k, v in data.items()
                }
            except Exception as e:
                logger.error("加载元数据失败: %s", e)

    @property
    def is_ready(self) -> bool:
        """索引是否已加载"""
        return self._index is not None and self._index.ntotal > 0

    @property
    def total_documents(self) -> int:
        """已索引文档总数"""
        return len(self._metadata_map)

    def search(
        self,
        query: str,
        top_k: int = 10,
        tag_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        执行自然语言搜索。

        Args:
            query: 自然语言查询
            top_k: 返回结果数量
            tag_filter: 可选标签过滤（逗号分隔多个标签）

        Returns:
            按相似度排序的搜索结果列表
        """
        if not self.is_ready:
            logger.warning("索引未就绪，请先执行 scan 命令构建索引")
            return []

        # 生成查询向量
        query_embedding = self._llm_client.embed_text(query)
        query_vec = np.array([query_embedding], dtype=np.float32)
        import faiss

        faiss.normalize_L2(query_vec)

        # 搜索
        actual_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, actual_k)

        # 构建结果
        results: list[SearchResult] = []
        filter_tags: set[str] = set()
        if tag_filter:
            filter_tags = {t.strip() for t in tag_filter.split(",") if t.strip()}

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            obj_token = self._id_map.get(int(idx))
            if not obj_token:
                continue

            metadata = self._metadata_map.get(obj_token)
            if not metadata:
                continue

            # 标签过滤
            if filter_tags and not filter_tags.intersection(set(metadata.tags)):
                continue

            results.append(
                SearchResult(
                    obj_token=metadata.obj_token,
                    title=metadata.title,
                    summary=metadata.summary,
                    tags=metadata.tags,
                    score=float(score),
                    source_url=metadata.source_url,
                    content_preview=metadata.content_preview,
                    doc_type=metadata.doc_type,
                    indexed_at=metadata.indexed_at,
                )
            )

        return results

    def get_document(self, obj_token: str) -> Optional[DocumentMetadata]:
        """获取指定文档的元数据"""
        return self._metadata_map.get(obj_token)

    def list_all_tags(self) -> dict[str, int]:
        """统计所有标签及其出现次数"""
        tag_counts: dict[str, int] = {}
        for metadata in self._metadata_map.values():
            for tag in metadata.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))
