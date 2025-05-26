import os
import subprocess
import time
import sys
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def setup_logging(result_dir):
    """Setup logging configuration"""
    log_file = Path(result_dir) / "log.txt"
    
    # 创建格式化器
    formatter = logging.Formatter('%(message)s')
    
    # 创建并配置文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 创建并配置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 获取根日志记录器并配置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
