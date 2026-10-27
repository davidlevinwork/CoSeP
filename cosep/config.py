"""CoSeP configuration module."""
import os
import sys
from datetime import datetime

from loguru import logger

# Global parameters
verbose: bool = False
output_dir: str = ''  # Set by init_logger()


def _new_output_dir() -> str:
    """Create a new, unique output directory for this run: outputs/<timestamp>, or outputs/<timestamp>_<n>.

    Creating the folder fails if it already exists, so two runs that start in the same second get different folders.
    """
    base = os.path.join('outputs', datetime.now().strftime("%Y%m%d_%H%M%S"))
    path, n = base, 1
    while True:
        try:
            os.makedirs(path)
            return path
        except FileExistsError:
            n += 1
            path = f'{base}_{n}'


def get_output_dir() -> str:
    """Get the output directory of this run, creating it on first use.

    Returns:
        Path to the output directory for this run: the one init_logger() created, or a new one.
    """
    global output_dir
    if not output_dir:
        # If not initialized, create a default
        output_dir = _new_output_dir()
    return output_dir


def init_logger(verbose: bool = True) -> None:
    """Initialize logger with console and file outputs.

    Args:
        verbose: Show debug messages on the console. The log file always records them.
    """
    global output_dir
    logger.remove()

    output_dir = _new_output_dir()

    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{function: <20}</cyan> | "
               "<level>{message}</level>",
        level="TRACE" if verbose else "INFO"
    )

    log_filename = os.path.join(output_dir, 'log.log')
    logger.add(
        log_filename,
        format="{time:YYYY-MM-DD HH:mm:ss} | "
               "{level: <8} | "
               "{function: <20} | "
               "{message}",
        level="TRACE"
    )

    logger.info(f"Output directory: {output_dir}")
