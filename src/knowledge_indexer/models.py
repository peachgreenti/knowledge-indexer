"""数据模型定义"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocType(str, Enum):
    """飞书文档类型"""
    DOCX = "docx"
    DOC = "doc"
    SHEET = "sheet"
    MINDNOTE = "mindnote"
    BITABLE = "bitable"
    UNKNOWN = "unknown"


class WikiNode(BaseModel):
    """知识库节点信息"""
    space_id: str
    node_token: str
    obj_token: str
    obj_type: DocType = DocType.UNKNOWN
    title: str = ""
    parent_node_token: Optional[str] = None
    has_child: bool = False
    creator: str = ""
    owner: str = ""
    obj_create_time: str = ""
    obj_edit_time: str = ""
    node_create_time: str = ""


class DocumentMetadata(BaseModel):
    """文档索引元数据"""
    obj_token: str = Field(..., description="文档唯一标识")
    node_token: str = Field("", description="Wiki 节点 token")
    space_id: str = Field("", description="知识库空间 ID")
    title: str = Field("", description="文档标题")
    doc_type: DocType = Field(DocType.UNKNOWN, description="文档类型")
    summary: str = Field("", description="LLM 生成的摘要")
    tags: list[str] = Field(default_factory=list, description="LLM 生成的标签")
    content_preview: str = Field("", description="文档内容预览（截取前 N 字符）")
    source_url: str = Field("", description="文档原始链接")
    indexed_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="索引时间",
    )
    doc_edited_at: str = Field("", description="文档最后编辑时间")
    extra: dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class IndexState(BaseModel):
    """索引扫描状态，用于增量更新"""
    last_scan_time: str = Field("", description="上次扫描时间")
    processed_docs: dict[str, str] = Field(
        default_factory=dict,
        description="已处理文档 {obj_token: obj_edit_time}",
    )
    total_indexed: int = Field(0, description="已索引文档总数")


class SearchResult(BaseModel):
    """搜索结果"""
    obj_token: str
    title: str
    summary: str
    tags: list[str]
    score: float = Field(description="相似度分数")
    source_url: str = ""
    content_preview: str = ""
    doc_type: DocType = DocType.UNKNOWN
    indexed_at: str = ""


class ScanStats(BaseModel):
    """扫描统计信息"""
    total_nodes: int = 0
    new_docs: int = 0
    updated_docs: int = 0
    skipped_docs: int = 0
    failed_docs: int = 0
    errors: list[str] = Field(default_factory=list)
