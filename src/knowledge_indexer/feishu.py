"""飞书 API 客户端 — 基于 lark-cli 封装"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Generator, Optional

from .config import Settings
from .models import DocType, WikiNode

logger = logging.getLogger(__name__)


class FeishuClient:
    """飞书 API 客户端，通过 lark-cli 调用飞书开放平台 API

    所有 API 调用委托给 lark-cli（https://github.com/larksuite/cli），
    利用其内置的认证管理、安全保护、分页处理和结构化输出能力。

    lark-cli 返回结构统一为 {"ok": true/false, "data": {...}} 或
    {"code": 0, "data": {...}, "msg": "success"}，本客户端自动提取 data 层。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _cli(
        self,
        args: list[str],
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """执行 lark-cli 命令"""
        cmd = ["lark-cli"] + args
        logger.debug("执行: %s", " ".join(cmd))
        return subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=self._settings.feishu_request_timeout,
        )

    def _cli_json(self, args: list[str]) -> dict:
        """执行 lark-cli 命令并返回 data 层数据"""
        result = self._cli(args + ["--format", "json"])
        if result.returncode != 0:
            raise RuntimeError(
                f"lark-cli 执行失败: {result.stderr.strip() or result.stdout.strip()}"
            )
        raw = json.loads(result.stdout)

        # 检查 lark-cli 是否返回错误
        if raw.get("ok") is False:
            err = raw.get("error", {})
            raise RuntimeError(
                f"lark-cli 错误: {err.get('message', raw)}"
            )

        # 提取 data 层（统一结构 {"code":0,"data":{...}} 或 {"ok":true,"data":{...}}）
        data = raw.get("data", raw)
        return data

    def close(self) -> None:
        """lark-cli 无需关闭连接"""
        pass

    # ── 知识库空间 API ────────────────────────────────────

    def list_wiki_spaces(self) -> list[dict]:
        """获取知识库空间列表（通过 lark-cli wiki spaces list）"""
        data = self._cli_json(["wiki", "spaces", "list", "--page-all"])
        return data.get("items", [])

    def get_wiki_space_info(self, space_id: str) -> dict:
        """获取知识库空间信息"""
        data = self._cli_json([
            "wiki", "spaces", "get",
            "--params", json.dumps({"space_id": space_id}),
        ])
        return data.get("space", data)

    # ── 知识库节点 API ────────────────────────────────────

    def list_all_nodes(
        self, space_id: str
    ) -> Generator[WikiNode, None, None]:
        """递归遍历知识库空间中的所有节点（通过 lark-cli wiki nodes list）"""
        yield from self._list_nodes_recursive(space_id, parent_node_token=None)

    def _list_nodes_recursive(
        self,
        space_id: str,
        parent_node_token: Optional[str],
    ) -> Generator[WikiNode, None, None]:
        """递归遍历子节点"""
        params: dict = {"page_size": 50, "space_id": space_id}
        if parent_node_token:
            params["parent_node_token"] = parent_node_token

        data = self._cli_json([
            "wiki", "nodes", "list",
            "--params", json.dumps(params),
            "--page-all",
        ])
        items = data.get("items", [])

        for item in items:
            obj_type_str = item.get("obj_type", "unknown")
            try:
                obj_type = DocType(obj_type_str)
            except ValueError:
                obj_type = DocType.UNKNOWN

            node = WikiNode(
                space_id=item.get("space_id", space_id),
                node_token=item.get("node_token", ""),
                obj_token=item.get("obj_token", ""),
                obj_type=obj_type,
                title=item.get("title", ""),
                parent_node_token=item.get("parent_node_token"),
                has_child=item.get("has_child", False),
                creator=item.get("creator", ""),
                owner=item.get("owner", ""),
                obj_create_time=item.get("obj_create_time", ""),
                obj_edit_time=item.get("obj_edit_time", ""),
                node_create_time=item.get("node_create_time", ""),
            )
            yield node

            # 递归遍历子节点
            if node.has_child:
                yield from self._list_nodes_recursive(
                    space_id, node.node_token
                )

    # ── 文档内容 API ──────────────────────────────────────

    def get_document_content(
        self, obj_token: str, obj_type: DocType
    ) -> str:
        """获取文档内容（通过 lark-cli docs +fetch）"""
        if obj_type not in (DocType.DOCX, DocType.DOC):
            logger.warning(
                "暂不支持获取 %s 类型文档内容 (token: %s)", obj_type, obj_token
            )
            return ""

        try:
            data = self._cli_json([
                "docs", "+fetch",
                "--doc", obj_token,
                "--format", "json",
            ])
        except RuntimeError as e:
            logger.warning("获取文档内容失败 (token: %s): %s", obj_token, e)
            return ""

        # lark-cli +fetch 返回 {"data": {"markdown": "...", "title": "..."}}
        content = data.get("markdown", "")
        if not content:
            # 降级：尝试从 blocks 中提取文本
            content = self._extract_from_blocks(data)
        return content

    @staticmethod
    def _extract_from_blocks(data: dict) -> str:
        """从 lark-cli +fetch 返回的 blocks 结构中提取纯文本"""
        blocks = data.get("blocks", [])
        if not blocks:
            return ""

        texts: list[str] = []

        def _walk(block: dict) -> None:
            block_type = block.get("type", "")
            # 文本块
            if block_type in ("text", "heading1", "heading2", "heading3",
                              "heading4", "heading5", "heading6", "heading7",
                              "heading8", "heading9", "bullet", "ordered",
                              "quote", "todo", "callout"):
                text_key = block.get("text_key", "")
                if text_key:
                    texts.append(text_key)
            # 表格块
            elif block_type == "table":
                cells = block.get("property", {}).get("cells", [])
                for row in cells:
                    for cell in row:
                        cell_text = (
                            cell.get("text_key", "") if isinstance(cell, dict) else ""
                        )
                        if cell_text:
                            texts.append(cell_text)
            # 子块
            for child in block.get("children", []):
                _walk(child)

        for block in blocks:
            _walk(block)

        return "\n".join(texts)

    @staticmethod
    def build_doc_url(obj_token: str, obj_type: DocType) -> str:
        """构建文档的 Web 访问链接"""
        if obj_type == DocType.DOCX:
            return f"https://bytedance.feishu.cn/docx/{obj_token}"
        elif obj_type == DocType.DOC:
            return f"https://bytedance.feishu.cn/doc/{obj_token}"
        return f"https://bytedance.feishu.cn/wiki/{obj_token}"
