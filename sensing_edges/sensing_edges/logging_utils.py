"""Shared logging setup for sensing_edges CLIs and runtime entrypoints."""

from __future__ import annotations

import logging


def configure_logging(level: str | int = logging.INFO) -> None:
    """Configure a consistent logging format for local and Modal runs."""
    resolved_level = level
    if isinstance(level, str):
        resolved_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=resolved_level,
        format="%(levelname)s %(name)s %(message)s",
    )
