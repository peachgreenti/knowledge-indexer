"""CLI 入口 - 命令行交互界面"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import Settings, load_settings
from .feishu import FeishuClient
from .indexer import IndexBuilder
from .models import IndexState
from .scheduler import Scheduler
from .search import SearchEngine

console = Console()


def _setup_logging(verbose: bool = False) -> None:
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _load_settings_or_exit(config_file: Optional[str]) -> Settings:
    """加载配置，失败时退出"""
    try:
        return load_settings(config_file)
    except Exception as e:
        console.print(f"[red]配置加载失败: {e}[/red]")
        console.print(
            "\n请确保已设置必要的环境变量或创建 .env 文件。"
            "\n运行 [cyan]knowledge-indexer init[/cyan] 查看配置模板。"
        )
        sys.exit(1)


@click.group()
@click.version_option(version=__version__, prog_name="knowledge-indexer")
@click.option("-v", "--verbose", is_flag=True, help="启用详细日志输出")
@click.option("-c", "--config", "config_file", default=None, help="配置文件路径")
@click.pass_context
def main(ctx: click.Context, verbose: bool, config_file: Optional[str]) -> None:
    """knowledge-indexer: 飞书知识库自动索引工具

    定时扫描飞书知识库，自动生成摘要和标签，构建本地搜索索引，支持自然语言查询。
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config_file"] = config_file
    _setup_logging(verbose)


@main.command()
def init() -> None:
    """生成 .env 配置文件模板"""
    from pathlib import Path

    env_template = """# ── 飞书应用凭证 ──────────────────────────────────
# 在飞书开放平台 (https://open.feishu.cn) 创建自建应用后获取
KI_FEISHU_APP_ID=your_app_id_here
KI_FEISHU_APP_SECRET=your_app_secret_here

# ── 知识库配置 ────────────────────────────────────────
# 要扫描的知识库空间 ID
KI_WIKI_SPACE_ID=your_space_id_here

# ── LLM 配置（OpenAI 兼容 API）───────────────────────
# 支持 OpenAI、DeepSeek、通义千问等兼容 API
KI_LLM_BASE_URL=https://api.openai.com/v1
KI_LLM_API_KEY=your_api_key_here
KI_LLM_MODEL=gpt-4o-mini
KI_LLM_EMBEDDING_MODEL=text-embedding-3-small

# ── 索引配置 ──────────────────────────────────────────
# 索引数据存储目录
# KI_DATA_DIR=~/.knowledge-indexer
# 向量嵌入维度（需与嵌入模型匹配）
# KI_EMBEDDING_DIM=1536
# 文档内容最大字符数（送入 LLM）
# KI_MAX_CONTENT_LENGTH=8000
# 批量处理文档数量
# KI_BATCH_SIZE=10

# ── 调度配置 ──────────────────────────────────────────
# 定时扫描间隔（分钟）
# KI_SCHEDULE_INTERVAL_MINUTES=60
"""

    env_path = Path.cwd() / ".env"
    if env_path.exists():
        if not click.confirm("已存在 .env 文件，是否覆盖？"):
            console.print("[yellow]已取消[/yellow]")
            return

    env_path.write_text(env_template, encoding="utf-8")
    console.print(
        Panel(
            f"配置模板已生成: [cyan]{env_path}[/cyan]\n\n"
            "请编辑 .env 文件，填入你的飞书应用凭证和 LLM API 密钥。\n\n"
            "获取知识库空间 ID:\n"
            "  1. 运行 [cyan]knowledge-indexer spaces[/cyan] 查看可用空间\n"
            "  2. 或在飞书知识库 URL 中找到空间 ID\n\n"
            "飞书应用所需权限:\n"
            "  • wiki:space:readonly\n"
            "  • wiki:node:readonly\n"
            "  • docx:document:readonly",
            title="[green]配置初始化完成[/green]",
            border_style="green",
        )
    )


@main.command()
@click.pass_context
def spaces(ctx: click.Context) -> None:
    """列出可访问的知识库空间"""
    settings = _load_settings_or_exit(ctx.obj.get("config_file"))
    client = FeishuClient(settings)

    try:
        spaces = client.list_wiki_spaces()
        if not spaces:
            console.print("[yellow]未找到可访问的知识库空间[/yellow]")
            console.print(
                "\n请确保:\n"
                "  1. 应用已启用 wiki:space:readonly 权限\n"
                "  2. 应用已被添加为知识库空间的成员"
            )
            return

        table = Table(title="可访问的知识库空间")
        table.add_column("空间名称", style="cyan")
        table.add_column("空间 ID", style="green")
        table.add_column("描述", style="dim")

        for space in spaces:
            table.add_row(
                space.get("name", ""),
                space.get("space_id", ""),
                space.get("description", "")[:50],
            )

        console.print(table)
        console.print(
            "\n使用 [cyan]KI_WIKI_SPACE_ID[/cyan] 配置要扫描的空间 ID"
        )

    finally:
        client.close()


@main.command()
@click.option("--space-id", default=None, help="知识库空间 ID（覆盖配置）")
@click.option("--force", is_flag=True, help="强制重新索引所有文档")
@click.pass_context
def scan(ctx: click.Context, space_id: Optional[str], force: bool) -> None:
    """扫描知识库并构建/更新索引"""
    settings = _load_settings_or_exit(ctx.obj.get("config_file"))

    with console.status("[bold green]正在扫描知识库..."):
        builder = IndexBuilder(settings)
        try:
            stats = builder.scan_and_index(space_id=space_id, force=force)
        finally:
            builder.close()

    # 显示结果
    table = Table(title="扫描结果")
    table.add_column("指标", style="cyan")
    table.add_column("数量", style="green", justify="right")
    table.add_row("总节点数", str(stats.total_nodes))
    table.add_row("新增文档", str(stats.new_docs))
    table.add_row("更新文档", str(stats.updated_docs))
    table.add_row("跳过文档", str(stats.skipped_docs))
    table.add_row("失败文档", str(stats.failed_docs))
    console.print(table)

    if stats.errors:
        console.print("\n[red]错误详情:[/red]")
        for err in stats.errors[:10]:
            console.print(f"  [red]•[/red] {err}")


@main.command()
@click.argument("query", required=False)
@click.option("-k", "--top-k", default=10, help="返回结果数量")
@click.option("-t", "--tag", default=None, help="标签过滤（逗号分隔）")
@click.option("-i", "--interactive", is_flag=True, help="进入交互式搜索模式")
@click.pass_context
def search(
    ctx: click.Context,
    query: Optional[str],
    top_k: int,
    tag: Optional[str],
    interactive: bool,
) -> None:
    """搜索已索引的文档（支持自然语言查询）"""
    settings = _load_settings_or_exit(ctx.obj.get("config_file"))
    engine = SearchEngine(settings)

    if not engine.is_ready:
        console.print(
            "[yellow]索引未就绪[/yellow]\n"
            "请先运行 [cyan]knowledge-indexer scan[/cyan] 构建索引。"
        )
        return

    console.print(
        f"[green]已加载 {engine.total_documents} 个文档的索引[/green]\n"
    )

    if interactive:
        _interactive_search(engine, top_k, tag)
        return

    if not query:
        console.print("请提供搜索查询，或使用 -i 进入交互模式")
        return

    _display_results(engine.search(query, top_k=top_k, tag_filter=tag))


def _interactive_search(engine: SearchEngine, top_k: int, tag: Optional[str]) -> None:
    """交互式搜索循环"""
    console.print("[bold]交互式搜索模式[/bold]（输入 [cyan]quit[/cyan] 退出）\n")

    while True:
        try:
            query = console.input("[bold cyan]🔍 搜索 > [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]再见！[/yellow]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[yellow]再见！[/yellow]")
            break

        results = engine.search(query, top_k=top_k, tag_filter=tag)
        _display_results(results)
        console.print()


def _display_results(results: list) -> None:
    """格式化显示搜索结果"""
    if not results:
        console.print("[yellow]未找到匹配的文档[/yellow]")
        return

    for i, result in enumerate(results, 1):
        tags_str = (
            ", ".join(f"[magenta]{t}[/magenta]" for t in result.tags)
            if result.tags
            else ""
        )
        preview = result.content_preview
        content_preview = preview[:150] + "..." if len(preview) > 150 else preview

        subtitle = (
            f"{tags_str}  [link={result.source_url}]{result.source_url}[/link]"
            if result.source_url
            else tags_str
        )

        console.print(Panel(
            f"{result.summary}\n\n"
            f"[dim]{content_preview}[/dim]",
            title=f"[bold cyan]{i}. {result.title}[/bold cyan]  "
                  f"[dim]相似度: {result.score:.4f}[/dim]",
            subtitle=subtitle,
            border_style="blue",
        ))


@main.command()
@click.pass_context
def tags(ctx: click.Context) -> None:
    """列出所有已索引的标签及其统计"""
    settings = _load_settings_or_exit(ctx.obj.get("config_file"))
    engine = SearchEngine(settings)

    if not engine.is_ready:
        console.print("[yellow]索引未就绪，请先运行 scan 命令[/yellow]")
        return

    tag_counts = engine.list_all_tags()
    if not tag_counts:
        console.print("[yellow]暂无标签数据[/yellow]")
        return

    table = Table(title=f"标签统计（共 {len(tag_counts)} 个标签）")
    table.add_column("标签", style="magenta")
    table.add_column("文档数", style="green", justify="right")

    for tag, count in tag_counts.items():
        table.add_row(tag, str(count))

    console.print(table)


@main.command()
@click.option("--interval", default=None, type=int, help="扫描间隔（分钟）")
@click.option("--no-immediate", is_flag=True, help="不立即执行首次扫描")
@click.pass_context
def watch(ctx: click.Context, interval: Optional[int], no_immediate: bool) -> None:
    """启动定时扫描（后台持续运行）"""
    settings = _load_settings_or_exit(ctx.obj.get("config_file"))

    console.print(
        Panel(
            f"定时扫描模式\n"
            f"  知识库空间: [cyan]{settings.wiki_space_id}[/cyan]\n"
            f"  扫描间隔: [cyan]{interval or settings.schedule_interval_minutes} 分钟[/cyan]\n"
            f"  索引目录: [cyan]{settings.data_dir}[/cyan]\n"
            f"  首次扫描: [cyan]{'否' if no_immediate else '是'}[/cyan]",
            title="[bold green]knowledge-indexer watch[/bold green]",
        )
    )

    scheduler = Scheduler(settings)
    scheduler.start(
        interval_minutes=interval,
        run_immediately=not no_immediate,
    )


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """显示索引状态信息"""
    settings = _load_settings_or_exit(ctx.obj.get("config_file"))

    table = Table(title="索引状态")
    table.add_column("项目", style="cyan")
    table.add_column("值", style="green")

    # 检查各文件是否存在
    index_exists = settings.index_path.exists()
    metadata_exists = settings.metadata_path.exists()
    state_exists = settings.state_path.exists()

    table.add_row("索引目录", str(settings.data_dir))
    table.add_row("FAISS 索引", "✅" if index_exists else "❌")
    table.add_row("元数据文件", "✅" if metadata_exists else "❌")
    table.add_row("状态文件", "✅" if state_exists else "❌")

    if state_exists:
        state = IndexState.model_validate_json(settings.state_path.read_text(encoding="utf-8"))
        table.add_row("上次扫描", state.last_scan_time or "从未")
        table.add_row("已索引文档", str(state.total_indexed))

    if metadata_exists:
        import json
        data = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
        table.add_row("元数据记录", str(len(data)))

    console.print(table)


if __name__ == "__main__":
    main()
