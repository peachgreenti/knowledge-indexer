<p align="center">
  <h1 align="center">knowledge-indexer</h1>
  <p align="center">
    <strong>飞书知识库自动索引工具</strong>
  </p>
  <p align="center">
    定时扫描飞书知识库 → LLM 自动生成摘要与标签 → FAISS 向量索引 → 自然语言语义搜索
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Feishu-API-orange.svg" alt="Feishu">
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-purple.svg" alt="FAISS">
</p>

---

## 目录

- [功能特性](#功能特性)
- [工作原理](#工作原理)
- [快速开始](#快速开始)
- [命令参考](#命令参考)
- [配置说明](#配置说明)
- [飞书应用配置](#飞书应用配置)
- [兼容的 LLM 服务](#兼容的-llm-服务)
- [高级用法](#高级用法)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [开发指南](#开发指南)
- [License](#license)

---

## 功能特性

| 特性 | 说明 |
|------|------|
| 📚 **知识库扫描** | 递归遍历飞书知识库空间中的所有文档节点（支持 docx / doc） |
| 🤖 **智能摘要** | 调用 LLM 为每个文档自动生成 100-200 字的中文摘要 |
| 🏷️ **自动标签** | LLM 提取 3-8 个语义标签，支持标签过滤搜索 |
| 🔍 **向量索引** | 基于 FAISS 构建本地向量索引，支持余弦相似度语义搜索 |
| 💬 **自然语言查询** | 支持中文自然语言搜索，返回按相似度排序的结果 |
| ⚡ **增量更新** | 基于文档编辑时间戳，仅处理新增和变更的文档 |
| ⏰ **定时调度** | 支持后台定时扫描，自动保持索引最新 |
| 🖥️ **交互式搜索** | 支持交互式搜索模式，持续查询无需重复启动 |
| 📊 **标签统计** | 查看所有标签及其关联文档数量 |
| 🛡️ **维度校验** | 自动检测嵌入维度不一致并提示重建 |

## 工作原理

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  飞书知识库   │────▶│  文档内容获取  │────▶│  LLM 处理     │────▶│  FAISS 索引   │
│  (Wiki API)  │     │  (Markdown)  │     │ 摘要+标签+嵌入 │     │  (本地存储)    │
└─────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │  自然语言搜索  │
                                                               │  (语义匹配)    │
                                                               └──────────────┘
```

**核心流程：**

1. **扫描** — 通过飞书 Wiki API 递归遍历知识库空间，获取所有文档节点
2. **获取** — 调用 Docs API 获取文档内容（Markdown 格式）
3. **处理** — 调用 LLM 生成摘要、标签，并生成向量嵌入
4. **索引** — 将向量写入 FAISS 索引，元数据写入 JSON 文件
5. **搜索** — 将查询文本转为向量，在 FAISS 中进行最近邻搜索

## 快速开始

### 前置条件

- Python >= 3.10
- 飞书开放平台自建应用（需配置 API 权限）
- OpenAI 兼容的 LLM 服务（用于生成摘要、标签和向量嵌入）

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/knowledge-indexer.git
cd knowledge-indexer

# 安装（推荐使用虚拟环境）
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -e .

# 如需 GPU 加速（可选）
pip install -e ".[gpu]"
```

### 2. 配置

```bash
# 生成 .env 配置模板
knowledge-indexer init

# 编辑 .env 文件，填入你的凭证
vim .env  # 或使用其他编辑器
```

`.env` 文件最少需要配置以下 4 项：

```env
KI_FEISHU_APP_ID=cli_xxxxxxxxxx
KI_FEISHU_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxxxx
KI_WIKI_SPACE_ID=xxxxxxxxxxxxxxxxxxxxxxxx
KI_LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. 验证知识库访问

```bash
# 查看可访问的知识库空间列表
knowledge-indexer spaces
```

输出示例：

```
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 空间名称           ┃ 空间 ID                  ┃ 描述                       ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 产品技术文档        │ 7123456789012345678      │ 产品技术团队知识库          │
│ 研发手册           │ 7987654321098765432      │ 研发团队内部文档            │
└──────────────────┴──────────────────────────┴──────────────────────────┘
```

### 4. 首次扫描构建索引

```bash
# 扫描知识库并构建索引
knowledge-indexer scan

# 强制重新索引所有文档（忽略增量更新）
knowledge-indexer scan --force

# 指定其他知识库空间
knowledge-indexer scan --space-id 7987654321098765432
```

### 5. 搜索

```bash
# 单次搜索
knowledge-indexer search "如何部署微服务"

# 交互式搜索模式
knowledge-indexer search -i

# 带标签过滤
knowledge-indexer search "API设计" -t "后端,架构"

# 限制返回结果数量
knowledge-indexer search "性能优化" -k 5
```

## 命令参考

```
knowledge-indexer [OPTIONS] COMMAND [ARGS...]

Options:
  --version          Show the version and exit.
  -v, --verbose      启用详细日志输出
  -c, --config TEXT  指定配置文件路径
  --help             Show this message and exit.

Commands:
  init      生成 .env 配置文件模板
  spaces    列出可访问的知识库空间
  scan      扫描知识库并构建/更新索引
  search    搜索已索引的文档（支持自然语言查询）
  tags      列出所有已索引的标签及其统计
  watch     启动定时扫描（后台持续运行）
  status    显示索引状态信息
```

### 命令详解

#### `init`

生成 `.env` 配置文件模板到当前目录。如果已存在 `.env` 文件，会提示是否覆盖。

```bash
knowledge-indexer init
```

#### `spaces`

列出飞书应用可访问的所有知识库空间，用于获取 `KI_WIKI_SPACE_ID`。

```bash
knowledge-indexer spaces
```

#### `scan`

扫描知识库空间中的所有文档，生成摘要、标签和向量索引。

```bash
knowledge-indexer scan [OPTIONS]

Options:
  --space-id TEXT  知识库空间 ID（覆盖配置文件中的值）
  --force          强制重新索引所有文档（忽略增量更新）
```

#### `search`

对已索引的文档进行自然语言语义搜索。

```bash
knowledge-indexer search [QUERY] [OPTIONS]

Arguments:
  QUERY            搜索查询文本

Options:
  -k, --top-k INTEGER   返回结果数量（默认 10）
  -t, --tag TEXT        标签过滤，多个标签用逗号分隔
  -i, --interactive     进入交互式搜索模式
```

#### `tags`

列出所有已索引文档的标签及其关联文档数量。

```bash
knowledge-indexer tags
```

#### `watch`

启动后台定时扫描模式，按配置的间隔自动更新索引。

```bash
knowledge-indexer watch [OPTIONS]

Options:
  --interval INTEGER   扫描间隔（分钟），覆盖配置值
  --no-immediate       不立即执行首次扫描，等待第一个间隔后才开始
```

#### `status`

显示当前索引的状态信息，包括文件存在性、上次扫描时间、已索引文档数等。

```bash
knowledge-indexer status
```

## 配置说明

所有配置通过环境变量设置（前缀 `KI_`），或写入项目根目录的 `.env` 文件。

### 必填配置

| 变量 | 说明 |
|------|------|
| `KI_FEISHU_APP_ID` | 飞书应用 App ID（在[飞书开放平台](https://open.feishu.cn)创建应用后获取） |
| `KI_FEISHU_APP_SECRET` | 飞书应用 App Secret |
| `KI_WIKI_SPACE_ID` | 要扫描的知识库空间 ID（运行 `spaces` 命令查看） |
| `KI_LLM_API_KEY` | LLM 服务 API 密钥 |

### 可选配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `KI_LLM_BASE_URL` | `https://api.openai.com/v1` | LLM API 基础 URL |
| `KI_LLM_MODEL` | `gpt-4o-mini` | 用于生成摘要和标签的模型 |
| `KI_LLM_EMBEDDING_MODEL` | `text-embedding-3-small` | 用于生成向量嵌入的模型 |
| `KI_LLM_MAX_TOKENS` | `2048` | LLM 最大生成 token 数 |
| `KI_LLM_TEMPERATURE` | `0.3` | LLM 生成温度 |
| `KI_DATA_DIR` | `~/.knowledge-indexer` | 索引数据存储目录 |
| `KI_EMBEDDING_DIM` | `1536` | 向量嵌入维度（需与嵌入模型匹配） |
| `KI_MAX_CONTENT_LENGTH` | `8000` | 送入 LLM 的文档内容最大字符数 |
| `KI_BATCH_SIZE` | `10` | 批量处理文档数量 |
| `KI_SCHEDULE_INTERVAL_MINUTES` | `60` | 定时扫描间隔（分钟） |
| `KI_FEISHU_REQUEST_TIMEOUT` | `30` | 飞书 API 请求超时（秒） |
| `KI_LLM_REQUEST_TIMEOUT` | `60` | LLM API 请求超时（秒） |
| `KI_MAX_RETRIES` | `3` | API 请求最大重试次数 |
| `KI_RETRY_DELAY` | `1.0` | 重试间隔（秒） |

> **注意**：`KI_EMBEDDING_DIM` 必须与所选嵌入模型的实际输出维度一致。常见维度：
> - `text-embedding-3-small` → `1536`
> - `text-embedding-3-large` → `3072`
> - `text-embedding-ada-002` → `1536`
> - DeepSeek Embedding → `1536`

## 飞书应用配置

### 1. 创建应用

1. 访问 [飞书开放平台](https://open.feishu.cn)
2. 点击「创建企业自建应用」
3. 记录 **App ID** 和 **App Secret**

### 2. 配置权限

在应用管理后台 → 「权限管理」中，申请以下权限：

| 权限标识 | 权限名称 | 用途 |
|----------|----------|------|
| `wiki:space:readonly` | 查看知识库空间 | 列出可访问的知识库空间 |
| `wiki:node:readonly` | 查看知识库节点 | 遍历知识库中的文档节点 |
| `docx:document:readonly` | 查看文档内容 | 获取文档的 Markdown 内容 |

### 3. 添加知识库成员

将应用添加为目标知识库空间的成员（至少需要「可阅读」权限），否则无法获取空间内的文档内容。

### 4. 发布应用

权限配置完成后，创建应用版本并发布。管理员审批通过后即可使用。

## 兼容的 LLM 服务

本项目使用 OpenAI 兼容 API，支持以下服务（通过修改 `KI_LLM_BASE_URL` 切换）：

| 服务 | Base URL | 模型示例 |
|------|----------|----------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini`, `gpt-4o` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-turbo` |
| 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| Moonshot | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| 本地 Ollama | `http://localhost:11434/v1` | `qwen2.5:7b` |

**示例 — 使用 DeepSeek：**

```env
KI_LLM_BASE_URL=https://api.deepseek.com/v1
KI_LLM_API_KEY=sk-your-deepseek-key
KI_LLM_MODEL=deepseek-chat
KI_LLM_EMBEDDING_MODEL=deepseek-chat
```

> **注意**：如果使用的嵌入模型维度不是 1536，需要同时修改 `KI_EMBEDDING_DIM`。

## 高级用法

### 指定配置文件

```bash
# 使用自定义配置文件
knowledge-indexer -c /path/to/custom.env scan
```

### 定时扫描

```bash
# 每 30 分钟扫描一次
knowledge-indexer watch --interval 30

# 不立即执行首次扫描
knowledge-indexer watch --no-immediate
```

### 查看索引状态

```bash
knowledge-indexer status
```

输出示例：

```
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 项目             ┃ 值                              ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 索引目录         │ /home/user/.knowledge-indexer   │
│ FAISS 索引       │ ✅                              │
│ 元数据文件       │ ✅                              │
│ 状态文件         │ ✅                              │
│ 上次扫描         │ 2026-04-20T10:30:00            │
│ 已索引文档       │ 128                             │
│ 元数据记录       │ 128                             │
└──────────────────┴────────────────────────────────┘
```

### 标签过滤搜索

```bash
# 搜索带有特定标签的文档
knowledge-indexer search "系统设计" -t "架构,后端"

# 交互模式下使用标签过滤
knowledge-indexer search -i -t "Python"
```

### 强制重建索引

当更换了嵌入模型或维度配置后，需要强制重建索引：

```bash
knowledge-indexer scan --force
```

### 数据存储

所有索引数据默认存储在 `~/.knowledge-indexer/` 目录下：

```
~/.knowledge-indexer/
├── faiss.index       # FAISS 向量索引文件
├── id_map.json       # FAISS 向量 ID → 文档 token 映射
├── metadata.json     # 文档元数据（标题、摘要、标签等）
└── state.json        # 扫描状态（已处理文档记录）
```

可通过 `KI_DATA_DIR` 环境变量修改存储位置。

## 项目结构

```
knowledge-indexer/
├── pyproject.toml              # 项目配置和依赖
├── .env.example                # 环境变量配置模板
├── README.md                   # 项目文档
└── src/
    └── knowledge_indexer/
        ├── __init__.py         # 包初始化
        ├── cli.py              # CLI 入口（Click + Rich）
        ├── config.py           # 配置管理（Pydantic Settings）
        ├── models.py           # 数据模型定义
        ├── feishu.py           # 飞书开放平台 API 客户端
        ├── llm.py              # LLM 集成（摘要/标签/嵌入）
        ├── indexer.py          # 索引构建器（FAISS）
        ├── search.py           # 搜索引擎（向量检索）
        └── scheduler.py        # 定时调度器
```

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| CLI 框架 | [Click](https://click.palletsprojects.com/) | 命令行解析和交互 |
| 终端美化 | [Rich](https://rich.readthedocs.io/) | 彩色输出、表格、面板 |
| HTTP 客户端 | [httpx](https://www.python-httpx.org/) | 飞书 API 请求 |
| LLM 客户端 | [OpenAI Python SDK](https://github.com/openai/openai-python) | 摘要/标签/嵌入生成 |
| 向量索引 | [FAISS](https://github.com/facebookresearch/faiss) | 高性能向量相似度搜索 |
| 数据验证 | [Pydantic v2](https://docs.pydantic.dev/) | 配置和数据模型验证 |
| 配置管理 | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | 环境变量/.env 加载 |
| 定时调度 | [schedule](https://schedule.readthedocs.io/) | 定时任务管理 |

## 开发指南

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

### 代码检查

```bash
ruff check src/
```

### 代码格式化

```bash
ruff format src/
```

## License

[MIT](LICENSE)
