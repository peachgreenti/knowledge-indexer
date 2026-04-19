"""飞书开放平台 API 客户端"""

from __future__ import annotations

import logging
import time
from typing import Generator, Optional

import httpx

from .config import Settings
from .models import DocType, WikiNode

logger = logging.getLogger(__name__)

BASE_URL = "https://open.feishu.cn"


class FeishuClient:
    """飞书 API 客户端，支持自动 token 管理和请求重试"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._token: str = ""
        self._token_expire_at: float = 0.0
        self._client = httpx.Client(
            base_url=BASE_URL,
            timeout=httpx.Timeout(settings.feishu_request_timeout),
        )

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    @property
    def token(self) -> str:
        """获取 tenant_access_token，自动缓存和刷新"""
        if self._token and time.time() < self._token_expire_at - 300:
            return self._token
        self._refresh_token()
        return self._token

    def _refresh_token(self) -> None:
        """刷新 tenant_access_token"""
        resp = self._request(
            "POST",
            "/open-apis/auth/v3/tenant_access_token/internal",
            json_data={
                "app_id": self._settings.feishu_app_id,
                "app_secret": self._settings.feishu_app_secret,
            },
            skip_auth=True,
        )
        self._token = resp["tenant_access_token"]
        self._token_expire_at = time.time() + resp["expire"]
        logger.info("飞书 token 已刷新，有效期 %d 秒", resp["expire"])

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
        skip_auth: bool = False,
    ) -> dict:
        """发送 API 请求，支持自动重试"""
        headers = {} if skip_auth else self._headers
        last_error: Exception | None = None

        for attempt in range(self._settings.max_retries):
            try:
                resp = self._client.request(
                    method,
                    path,
                    headers=headers,
                    params=params,
                    json=json_data,
                )
                data = resp.json()

                if data.get("code") == 0:
                    return data.get("data", data)

                # Token 过期时自动刷新并重试
                if data.get("code") == 99991663 and not skip_auth:
                    logger.warning("token 过期，自动刷新重试")
                    self._refresh_token()
                    continue

                error_msg = f"飞书 API 错误 [{data.get('code')}]: {data.get('msg')}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            except (httpx.HTTPError, RuntimeError) as e:
                last_error = e
                if attempt < self._settings.max_retries - 1:
                    delay = self._settings.retry_delay * (2**attempt)
                    logger.warning(
                        "请求失败 (%s)，%0.1f 秒后重试: %s", method, delay, e
                    )
                    time.sleep(delay)

        raise RuntimeError(f"请求 {method} {path} 失败: {last_error}") from last_error

    def close(self) -> None:
        """关闭 HTTP 客户端"""
        self._client.close()

    # ── 知识库空间 API ────────────────────────────────────

    def list_wiki_spaces(self) -> list[dict]:
        """获取知识库空间列表"""
        all_spaces: list[dict] = []
        page_token = ""

        while True:
            params: dict = {"page_size": 50}
            if page_token:
                params["page_token"] = page_token

            data = self._request("GET", "/open-apis/wiki/v2/spaces", params=params)
            items = data.get("items", [])
            all_spaces.extend(items)

            if not data.get("has_more", False):
                break
            page_token = data.get("page_token", "")

        return all_spaces

    def get_wiki_space_info(self, space_id: str) -> dict:
        """获取知识库空间信息"""
        return self._request("GET", f"/open-apis/wiki/v2/spaces/{space_id}")

    # ── 知识库节点 API ────────────────────────────────────

    def list_all_nodes(
        self, space_id: str
    ) -> Generator[WikiNode, None, None]:
        """
        递归遍历知识库空间中的所有节点。

        Yields:
            WikiNode: 每个节点的信息
        """
        yield from self._list_nodes_recursive(space_id, parent_node_token=None)

    def _list_nodes_recursive(
        self,
        space_id: str,
        parent_node_token: Optional[str],
    ) -> Generator[WikiNode, None, None]:
        """递归遍历子节点"""
        page_token = ""

        while True:
            params: dict = {"page_size": 50}
            if parent_node_token:
                params["parent_node_token"] = parent_node_token
            if page_token:
                params["page_token"] = page_token

            data = self._request(
                "GET",
                f"/open-apis/wiki/v2/spaces/{space_id}/nodes",
                params=params,
            )
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

            if not data.get("has_more", False):
                break
            page_token = data.get("page_token", "")

    # ── 文档内容 API ──────────────────────────────────────

    def get_document_content(
        self, obj_token: str, obj_type: DocType
    ) -> str:
        """
        获取文档内容，返回 Markdown 格式文本。

        Args:
            obj_token: 文档对象 token
            obj_type: 文档类型

        Returns:
            文档的 Markdown 文本内容
        """
        if obj_type == DocType.DOCX:
            return self._get_docx_markdown(obj_token)
        elif obj_type == DocType.DOC:
            return self._get_doc_content(obj_token)
        else:
            logger.warning(
                "暂不支持获取 %s 类型文档内容 (token: %s)", obj_type, obj_token
            )
            return ""

    def _get_docx_markdown(self, obj_token: str) -> str:
        """获取新版文档 (docx) 的 Markdown 内容"""
        data = self._request(
            "GET",
            "/open-apis/docs/v1/content",
            params={
                "doc_token": obj_token,
                "doc_type": "docx",
                "content_type": "markdown",
            },
        )
        return data.get("content", "")

    def _get_doc_content(self, obj_token: str) -> str:
        """获取旧版文档 (doc) 的内容"""
        data = self._request(
            "GET",
            f"/open-apis/doc/v2/{obj_token}/content",
        )
        # 旧版文档返回的是 JSON 结构化内容，尝试提取纯文本
        content = data.get("content", "{}")
        return self._extract_text_from_doc_json(content)

    @staticmethod
    def _extract_text_from_doc_json(content: str) -> str:
        """从旧版文档 JSON 结构中提取纯文本"""
        import json

        try:
            body = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content

        texts: list[str] = []

        def _walk(node: dict) -> None:
            node_type = node.get("type", "")
            if node_type in ("text", "textRun"):
                text = node.get("text", "")
                if text:
                    texts.append(text)
            for child in node.get("children", []):
                _walk(child)

        _walk(body)
        return "\n".join(texts)

    def build_doc_url(self, obj_token: str, obj_type: DocType) -> str:
        """构建文档的 Web 访问链接"""
        if obj_type == DocType.DOCX:
            return f"https://bytedance.feishu.cn/docx/{obj_token}"
        elif obj_type == DocType.DOC:
            return f"https://bytedance.feishu.cn/doc/{obj_token}"
        return f"https://bytedance.feishu.cn/wiki/{obj_token}"
