"""
统一的日志配置模块

提供统一的日志配置，支持：
- 控制台输出
- 文件输出
- 日志轮转
- 不同级别的日志格式
"""
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

from core.settings import LogLevel, settings


def setup_logging(
    log_level: Optional[LogLevel] = None,
    log_file: Optional[str] = None,
    log_dir: str = "./logs",
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    配置项目日志系统

    Args:
        log_level: 日志级别，默认使用 settings.LOG_LEVEL
        log_file: 日志文件名，默认使用 'agent-service.log'
        log_dir: 日志目录
        enable_file_logging: 是否启用文件日志
        enable_console_logging: 是否启用控制台日志
        max_bytes: 单个日志文件最大大小（字节）
        backup_count: 保留的备份文件数量
    """
    # 确定日志级别
    if log_level is None:
        log_level = settings.LOG_LEVEL
    
    level = log_level.to_logging_level()
    
    # 创建日志目录
    if enable_file_logging:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            log_file = "agent-service.log"
        
        log_file_path = log_path / log_file
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有的处理器
    root_logger.handlers.clear()
    
    # 定义日志格式
    detailed_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    simple_format = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台处理器
    if enable_console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_format)
        root_logger.addHandler(console_handler)
    
    # 文件处理器（带轮转）
    if enable_file_logging:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_format)
        root_logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    root_logger.info(f"日志系统已初始化，级别: {log_level.value}")


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称，通常使用 __name__
    
    Returns:
        logging.Logger 实例
    """
    return logging.getLogger(name)
