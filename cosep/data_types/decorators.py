"""Decorators for timing and conditional execution."""
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from loguru import logger

import cosep.config as config

T = TypeVar("T")


def verbose_decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
    """Run a function only in verbose mode.

    Args:
        func: Function to wrap.

    Returns:
        The wrapped function: it calls `func` when `cosep.config.verbose` is set, and otherwise returns None without
        calling it.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
        if config.verbose:
            return func(*args, **kwargs)
        return None
    return wrapper


def timing_decorator(func: Callable[..., T]) -> Callable[..., T]:
    """Log how long each call of a function takes, in verbose mode.

    Args:
        func: Function to wrap.

    Returns:
        The wrapped function, which returns what `func` returns.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        if config.verbose:
            logger.debug(f"{func.__name__} took {time.perf_counter() - start_time:.4f}s")
        return result
    return wrapper
