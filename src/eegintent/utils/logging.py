import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger.

    Args:
        level: Logging level name, e.g., "INFO", "DEBUG".
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
