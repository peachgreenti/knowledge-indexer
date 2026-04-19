"""定时调度模块 - 周期性扫描知识库并更新索引"""

from __future__ import annotations

import logging
import signal
import threading
from typing import Optional

import schedule

from .config import Settings
from .indexer import IndexBuilder

logger = logging.getLogger(__name__)


class Scheduler:
    """定时扫描调度器"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._stop_event = threading.Event()
        self._running = False

    def start(
        self,
        interval_minutes: Optional[int] = None,
        run_immediately: bool = True,
    ) -> None:
        """
        启动定时调度。

        Args:
            interval_minutes: 扫描间隔（分钟），为空则使用配置值
            run_immediately: 是否立即执行首次扫描
        """
        interval = interval_minutes or self._settings.schedule_interval_minutes
        self._running = True

        # 清除可能残留的旧定时任务
        schedule.clear()

        # 设置定时任务
        schedule.every(interval).minutes.do(self._run_scan)

        # 注册信号处理
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_stop(signum, frame):
            logger.info("收到停止信号，正在关闭调度器...")
            self._stop_event.set()

        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

        try:
            if run_immediately:
                logger.info("执行首次扫描...")
                self._run_scan()

            logger.info("定时调度已启动，每 %d 分钟扫描一次", interval)
            logger.info("按 Ctrl+C 停止")

            while not self._stop_event.is_set():
                schedule.run_pending()
                self._stop_event.wait(1)

        except KeyboardInterrupt:
            logger.info("用户中断，正在停止...")
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            self._running = False
            logger.info("调度器已停止")

    def _run_scan(self) -> None:
        """执行一次扫描"""
        if self._stop_event.is_set():
            return

        try:
            logger.info("=" * 50)
            logger.info("开始定时扫描...")
            builder = IndexBuilder(self._settings)
            stats = builder.scan_and_index()
            builder.close()

            logger.info(
                "扫描完成: 新增=%d, 更新=%d, 跳过=%d, 失败=%d",
                stats.new_docs,
                stats.updated_docs,
                stats.skipped_docs,
                stats.failed_docs,
            )

            if stats.errors:
                for err in stats.errors[:5]:
                    logger.error("  错误: %s", err)

        except Exception as e:
            logger.error("扫描执行失败: %s", e, exc_info=True)

    def stop(self) -> None:
        """停止调度器"""
        self._stop_event.set()
