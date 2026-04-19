"""索引构建器 - 扫描飞书知识库、生成摘要标签、构建 FAISS 向量索引"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import numpy as np

from .config import Settings
from .feishu import FeishuClient
from .llm import LLMClient
from .models import (
    DocType,
    DocumentMetadata,
    IndexState,
    ScanStats,
    WikiNode,
)

logger = logging.getLogger(__name__)


class IndexBuilder:
    """索引构建器：扫描知识库 → 生成摘要/标签 → 构建向量索引"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._feishu = FeishuClient(settings)
        self._llm = LLMClient(settings)
        self._state = self._load_state()
        self._metadata_map: dict[str, DocumentMetadata] = self._load_metadata()

        # FAISS 索引和 ID 映射（在内存中维护，避免频繁磁盘 I/O）
        self._faiss_index = None
        self._id_map: dict[int, str] = {}
        self._load_faiss_index()

    def scan_and_index(
        self,
        space_id: Optional[str] = None,
        force: bool = False,
    ) -> ScanStats:
        """
        执行完整的扫描和索引流程。

        Args:
            space_id: 知识库空间 ID，为空则使用配置中的默认值
            force: 是否强制重新索引所有文档

        Returns:
            ScanStats: 扫描统计信息
        """
        space_id = space_id or self._settings.wiki_space_id
        stats = ScanStats()

        logger.info("开始扫描知识库空间: %s", space_id)

        # 1. 遍历所有节点
        nodes_to_process: list[WikiNode] = []
        for node in self._feishu.list_all_nodes(space_id):
            stats.total_nodes += 1

            # 跳过非文档类型
            if node.obj_type not in (DocType.DOCX, DocType.DOC):
                logger.debug(
                    "跳过非文档节点: %s (类型: %s)", node.title, node.obj_type
                )
                continue

            # 增量更新检查
            if not force and self._is_up_to_date(node):
                stats.skipped_docs += 1
                continue

            nodes_to_process.append(node)

        logger.info(
            "共 %d 个节点，%d 个需要处理，%d 个跳过",
            stats.total_nodes,
            len(nodes_to_process),
            stats.skipped_docs,
        )

        # 2. 批量处理文档
        batch_size = self._settings.batch_size
        for i in range(0, len(nodes_to_process), batch_size):
            batch = nodes_to_process[i : i + batch_size]
            logger.info(
                "处理批次 %d/%d（共 %d 个文档）",
                i // batch_size + 1,
                (len(nodes_to_process) + batch_size - 1) // batch_size,
                len(batch),
            )

            for node in batch:
                try:
                    self._process_document(node, stats, force)
                except Exception as e:
                    stats.failed_docs += 1
                    error_msg = f"处理文档失败 [{node.title}]: {e}"
                    logger.error(error_msg, exc_info=True)
                    stats.errors.append(error_msg)

        # 3. 一次性保存索引和状态
        if nodes_to_process:
            self._save_all()

        logger.info(
            "扫描完成: 新增 %d, 更新 %d, 跳过 %d, 失败 %d",
            stats.new_docs,
            stats.updated_docs,
            stats.skipped_docs,
            stats.failed_docs,
        )

        return stats

    def _process_document(
        self, node: WikiNode, stats: ScanStats, force: bool
    ) -> None:
        """处理单个文档：获取内容 → 生成摘要/标签 → 添加到索引"""
        logger.info("处理文档: %s (%s)", node.title, node.obj_token)

        # 获取文档内容
        content = self._feishu.get_document_content(node.obj_token, node.obj_type)
        if not content or len(content.strip()) < 10:
            logger.warning("文档内容为空或过短，跳过: %s", node.title)
            stats.skipped_docs += 1
            return

        # 截断内容
        max_len = self._settings.max_content_length
        truncated_content = content[:max_len]

        # 生成摘要和标签
        summary = self._llm.generate_summary(node.title, truncated_content)
        tags = self._llm.generate_tags(node.title, truncated_content)

        # 生成嵌入向量
        embed_text = f"{node.title}\n{summary}\n{' '.join(tags)}"
        embedding = self._llm.embed_text(embed_text)

        # 维度校验
        if len(embedding) != self._settings.embedding_dim:
            raise ValueError(
                f"嵌入维度不匹配: 期望 {self._settings.embedding_dim}, "
                f"实际 {len(embedding)}。请检查 KI_EMBEDDING_DIM 配置 "
                f"或清空索引数据后重新扫描。"
            )

        # 构建元数据
        is_new = node.obj_token not in self._metadata_map
        if is_new:
            stats.new_docs += 1
        else:
            stats.updated_docs += 1

        metadata = DocumentMetadata(
            obj_token=node.obj_token,
            node_token=node.node_token,
            space_id=node.space_id,
            title=node.title,
            doc_type=node.obj_type,
            summary=summary,
            tags=tags,
            content_preview=content[:500],
            source_url=self._feishu.build_doc_url(node.obj_token, node.obj_type),
            indexed_at=datetime.now().isoformat(),
            doc_edited_at=node.obj_edit_time,
        )

        # 更新元数据和状态
        self._metadata_map[node.obj_token] = metadata
        self._state.processed_docs[node.obj_token] = node.obj_edit_time

        # 添加到内存中的 FAISS 索引
        self._add_to_index(embedding, node.obj_token)

    def _is_up_to_date(self, node: WikiNode) -> bool:
        """检查文档是否需要更新"""
        last_edit = self._state.processed_docs.get(node.obj_token, "")
        return last_edit == node.obj_edit_time and node.obj_edit_time != ""

    # ── FAISS 索引管理 ────────────────────────────────────

    def _load_faiss_index(self) -> None:
        """加载 FAISS 索引和 ID 映射到内存"""
        import faiss

        index_path = self._settings.index_path
        id_map_path = self._settings.data_dir / "id_map.json"

        if index_path.exists():
            try:
                self._faiss_index = faiss.read_index(str(index_path))
                # 维度一致性检查
                if self._faiss_index.d != self._settings.embedding_dim:
                    logger.warning(
                        "FAISS 索引维度 (%d) 与配置维度 (%d) 不一致，"
                        "将重建索引。请确认 KI_EMBEDDING_DIM 配置正确。",
                        self._faiss_index.d,
                        self._settings.embedding_dim,
                    )
                    self._faiss_index = faiss.IndexFlatIP(
                        self._settings.embedding_dim
                    )
                    self._id_map = {}
                    # 立即保存正确的空索引，覆盖错误的文件
                    faiss.write_index(
                        self._faiss_index, str(index_path)
                    )
                else:
                    logger.info(
                        "FAISS 索引已加载，共 %d 个向量",
                        self._faiss_index.ntotal,
                    )
            except Exception as e:
                logger.warning("加载 FAISS 索引失败，将重建: %s", e)
                self._faiss_index = faiss.IndexFlatIP(
                    self._settings.embedding_dim
                )
                self._id_map = {}
        else:
            self._faiss_index = faiss.IndexFlatIP(self._settings.embedding_dim)
            self._id_map = {}

        # 加载 ID 映射
        if id_map_path.exists():
            try:
                data = json.loads(id_map_path.read_text(encoding="utf-8"))
                self._id_map = {int(k): v for k, v in data.items()}
            except Exception as e:
                logger.warning("加载 ID 映射失败: %s", e)
                self._id_map = {}

    def _add_to_index(self, embedding: list[float], obj_token: str) -> None:
        """将向量添加到内存中的 FAISS 索引"""
        import faiss

        # 维度校验
        if len(embedding) != self._settings.embedding_dim:
            raise ValueError(
                f"嵌入维度不匹配: 期望 {self._settings.embedding_dim}, "
                f"实际 {len(embedding)}"
            )

        vec = np.array([embedding], dtype=np.float32)
        # L2 归一化（用于余弦相似度）
        faiss.normalize_L2(vec)

        # 查找是否已存在，若存在则替换
        token_to_remove = None
        for idx, token in self._id_map.items():
            if token == obj_token:
                token_to_remove = idx
                break

        if token_to_remove is not None:
            # FAISS IndexFlatIP 不支持直接删除，重建索引
            self._faiss_index, self._id_map = self._rebuild_index_without(
                self._faiss_index, self._id_map, token_to_remove
            )

        new_id = self._faiss_index.ntotal
        self._faiss_index.add(vec)
        self._id_map[new_id] = obj_token

    def _rebuild_index_without(
        self,
        index,
        id_map: dict[int, str],
        remove_id: int,
    ):
        """重建索引，移除指定 ID 的向量"""
        import faiss

        dim = self._settings.embedding_dim
        new_index = faiss.IndexFlatIP(dim)
        new_id_map: dict[int, str] = {}
        new_idx = 0

        for old_id, token in id_map.items():
            if old_id == remove_id:
                continue
            vec = np.zeros((1, dim), dtype=np.float32)
            index.reconstruct(old_id, vec[0])
            new_index.add(vec)
            new_id_map[new_idx] = token
            new_idx += 1

        return new_index, new_id_map

    # ── 持久化 ────────────────────────────────────────────

    def _load_state(self) -> IndexState:
        """加载扫描状态"""
        path = self._settings.state_path
        if path.exists():
            try:
                return IndexState.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except Exception as e:
                logger.warning("加载状态文件失败，将重新开始: %s", e)
        return IndexState()

    def _load_metadata(self) -> dict[str, DocumentMetadata]:
        """加载文档元数据"""
        path = self._settings.metadata_path
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return {
                    k: DocumentMetadata.model_validate(v)
                    for k, v in data.items()
                }
            except Exception as e:
                logger.warning("加载元数据文件失败: %s", e)
        return {}

    def _save_all(self) -> None:
        """一次性保存所有数据（FAISS 索引、ID 映射、元数据、状态）"""
        import faiss

        self._state.last_scan_time = datetime.now().isoformat()
        self._state.total_indexed = len(self._metadata_map)

        # 保存 FAISS 索引
        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(self._settings.index_path))

        # 保存 ID 映射
        id_map_path = self._settings.data_dir / "id_map.json"
        id_map_path.write_text(
            json.dumps(
                {str(k): v for k, v in self._id_map.items()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # 保存状态
        self._settings.state_path.write_text(
            self._state.model_dump_json(indent=2), encoding="utf-8"
        )

        # 保存元数据
        metadata_data = {
            k: v.model_dump() for k, v in self._metadata_map.items()
        }
        self._settings.metadata_path.write_text(
            json.dumps(metadata_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def close(self) -> None:
        """关闭客户端连接"""
        self._feishu.close()
