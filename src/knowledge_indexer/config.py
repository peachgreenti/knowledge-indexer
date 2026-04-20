"""配置管理模块 - 支持环境变量和 .env 文件"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_DATA_DIR = Path.home() / ".knowledge-indexer"


class Settings(BaseSettings):
    """应用配置，支持从环境变量和 .env 文件加载"""

    model_config = SettingsConfigDict(
        env_prefix="KI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 知识库配置 ────────────────────────────────────────
    wiki_space_id: str = Field(..., description="要扫描的知识库空间 ID")

    # ── LLM 配置（OpenAI 兼容 API）───────────────────────
    llm_base_url: str = Field(
        "https://api.openai.com/v1",
        description="LLM API 基础 URL（兼容 OpenAI 格式）",
    )
    llm_api_key: str = Field(..., description="LLM API 密钥")
    llm_model: str = Field("gpt-4o-mini", description="用于生成摘要和标签的模型")
    llm_max_tokens: int = Field(2048, description="LLM 最大生成 token 数")
    llm_temperature: float = Field(0.3, description="LLM 生成温度")

    # ── 嵌入模型配置（支持独立服务商）─────────────────────
    llm_embedding_model: str = Field(
        "text-embedding-3-small", description="用于生成向量嵌入的模型"
    )
    embedding_base_url: str | None = Field(
        None,
        description="嵌入 API 基础 URL（为空则使用 llm_base_url）",
    )
    embedding_api_key: str | None = Field(
        None,
        description="嵌入 API 密钥（为空则使用 llm_api_key）",
    )

    # ── 索引配置 ──────────────────────────────────────────
    data_dir: Path = Field(
        _DEFAULT_DATA_DIR, description="索引数据存储目录"
    )
    embedding_dim: int = Field(1536, description="向量嵌入维度")
    max_content_length: int = Field(
        8000, description="送入 LLM 的文档内容最大字符数"
    )
    batch_size: int = Field(10, description="批量处理文档数量")

    # ── 调度配置 ──────────────────────────────────────────
    schedule_interval_minutes: int = Field(
        60, description="定时扫描间隔（分钟）"
    )

    # ── 请求配置 ──────────────────────────────────────────
    feishu_request_timeout: int = Field(30, description="飞书 API 请求超时（秒）")
    llm_request_timeout: int = Field(60, description="LLM API 请求超时（秒）")
    max_retries: int = Field(3, description="API 请求最大重试次数")
    retry_delay: float = Field(1.0, description="重试间隔（秒）")

    @field_validator("data_dir", mode="before")
    @classmethod
    def ensure_data_dir(cls, v: str | Path) -> Path:
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def index_path(self) -> Path:
        """FAISS 索引文件路径"""
        return self.data_dir / "faiss.index"

    @property
    def metadata_path(self) -> Path:
        """文档元数据 JSON 文件路径"""
        return self.data_dir / "metadata.json"

    @property
    def state_path(self) -> Path:
        """扫描状态文件路径（记录已处理文档）"""
        return self.data_dir / "state.json"


def load_settings(config_file: Optional[str] = None) -> Settings:
    """加载配置，优先级：环境变量 > .env 文件 > 默认值"""
    kwargs = {}
    if config_file:
        kwargs["_env_file"] = config_file
    return Settings(**kwargs)
