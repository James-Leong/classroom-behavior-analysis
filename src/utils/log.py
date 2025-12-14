"""日志配置"""

import logging
import os

from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)


def get_logger(name):
    """获取日志记录器"""
    logger = logging.getLogger(name)
    return logger


@contextmanager
def suppress_fds():
    """Context manager that redirects FD 1 and 2 to /dev/null.

    This suppresses output from C-level prints and other threads that bypass
    Python's sys.stdout/sys.stderr objects.
    """
    devnull = os.open(os.devnull, os.O_RDWR)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stdout)
        os.close(old_stderr)
