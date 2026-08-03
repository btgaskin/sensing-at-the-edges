"""JAX persistent compilation-cache configuration.

Ported from gradient_ascent/jax_cache.py (commit 93baec2, 2026-03-27).
Simplified: removed legacy fallback paths.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

log = logging.getLogger(__name__)

_VALID_MATMUL_PRECISIONS = {"default", "high", "highest"}
_CONFIGURED = False


def setup_jax_cache(cfg: DictConfig) -> dict[str, object]:
    """Configure JAX runtime cache + precision. Call once at train start.

    Reads from cfg.jax if present, otherwise uses safe defaults.
    Returns summary dict for logging/manifest.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return {"cache_enabled": False, "note": "already configured"}

    jax_cfg = cfg.get("jax", {})
    if not jax_cfg:
        _CONFIGURED = True
        return {"cache_enabled": False, "matmul_precision": "default"}

    precision = str(jax_cfg.get("matmul_precision", "default")).strip().lower()
    if precision not in _VALID_MATMUL_PRECISIONS:
        raise ValueError(
            f"jax.matmul_precision must be one of {sorted(_VALID_MATMUL_PRECISIONS)}; "
            f"got {precision!r}."
        )

    cache_cfg = jax_cfg.get("cache", {}) if jax_cfg else {}
    cache_enabled = bool(cache_cfg.get("enabled", False)) if cache_cfg else False

    import jax

    try:
        jax.config.update("jax_default_matmul_precision", precision)
    except Exception as exc:
        log.warning("Failed to set JAX matmul precision: %s", exc)

    if not cache_enabled:
        _CONFIGURED = True
        log.info("JAX: cache=disabled matmul_precision=%s", precision)
        return {"cache_enabled": False, "matmul_precision": precision}

    cache_dir = str(cache_cfg.get("dir", "/cache/jax"))
    min_secs = float(cache_cfg.get("min_compile_time_secs", 5.0))

    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", min_secs)
    except Exception as exc:
        log.warning("Failed to configure JAX cache at %s: %s", cache_dir, exc)
        _CONFIGURED = True
        return {"cache_enabled": False, "matmul_precision": precision}

    _CONFIGURED = True
    log.info("JAX: cache=%s min_secs=%.1f matmul_precision=%s", cache_dir, min_secs, precision)
    return {
        "cache_enabled": True,
        "cache_dir": cache_dir,
        "cache_min_compile_time_secs": min_secs,
        "matmul_precision": precision,
    }
