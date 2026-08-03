"""Shared Modal infrastructure for sensing_edges.

Single source of truth for app, image, volumes, and mounts.
Imported by modal_train.py, modal_smoke.py, modal_eval.py.
"""

from __future__ import annotations

from pathlib import Path

import modal

# --- App ---
app = modal.App("sensing-edges")

# --- Volumes ---
jax_cache_vol = modal.Volume.from_name("se-jax-cache", create_if_missing=True)
outputs_vol = modal.Volume.from_name("se-outputs", create_if_missing=True)

VOLUMES = {
    "/cache/jax": jax_cache_vol,
    "/outputs": outputs_vol,
}

# --- Source paths ---
def _resolve_roots(
    module_path: Path,
    *,
    container_experiment_root: Path = Path("/root/experiment"),
    container_repo_root: Path = Path("/root"),
) -> tuple[Path, Path]:
    """Resolve local or container paths for the sensing_edges experiment."""
    local_se_root = module_path.parent.parent
    if (local_se_root / "config").exists() and (local_se_root / "sensing_edges").exists():
        return local_se_root.parent, local_se_root

    if (container_experiment_root / "config").exists() and (container_experiment_root / "sensing_edges").exists():
        return container_repo_root, container_experiment_root

    raise FileNotFoundError(f"Could not resolve sensing_edges roots from {module_path}")


REPO_ROOT, SE_ROOT = _resolve_roots(Path(__file__).resolve())


def _maybe_add_local_dir(image: modal.Image, local_dir: Path, remote_dir: str) -> modal.Image:
    """Mount an optional local directory into the image when it exists."""
    if local_dir.exists():
        return image.add_local_dir(str(local_dir), remote_dir)
    return image


def _maybe_add_local_file(image: modal.Image, local_file: Path, remote_path: str) -> modal.Image:
    """Mount an optional local file into the image when it exists."""
    if local_file.exists():
        return image.add_local_file(str(local_file), remote_path)
    return image


# --- Image ---
se_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "jax[cuda12]==0.9.0.1",
        "jax-evogym @ git+https://github.com/btgaskin/jax-evogym@v0.1.0",
        "numpy>=1.24",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "evosax>=0.2.0",
        "matplotlib>=3.8",
        "pandas>=2.2",
        "pyarrow>=16",
        "wandb>=0.19",
        "jsonschema>=4.23",
        "scipy>=1.11",
    )
    .env({"PYTHONPATH": "/root/src:/root/experiment"})
)
se_image = _maybe_add_local_dir(se_image, REPO_ROOT / "src" / "jax_evogym", "/root/src/jax_evogym")
se_image = _maybe_add_local_dir(se_image, SE_ROOT / "sensing_edges", "/root/experiment/sensing_edges")
se_image = _maybe_add_local_dir(se_image, SE_ROOT / "config", "/root/experiment/config")
se_image = _maybe_add_local_dir(se_image, SE_ROOT / "schemas", "/root/experiment/schemas")
se_image = _maybe_add_local_dir(se_image, SE_ROOT / "terrain_suites", "/root/experiment/terrain_suites")
se_image = _maybe_add_local_dir(se_image, SE_ROOT / "local_outputs", "/root/experiment/local_outputs")
se_image = _maybe_add_local_file(
    se_image,
    SE_ROOT / "scripts" / "_modal_common.py",
    "/root/_modal_common.py",
)


def compose_config(
    config_name: str = "config",
    overrides: list[str] | None = None,
    *,
    preset: str | None = None,
):
    """Compose the canonical sensing_edges config inside the Modal container."""
    from sensing_edges.runtime_config import compose_experiment_config

    return compose_experiment_config(
        preset=preset if preset is not None else config_name,
        overrides=overrides,
        config_dir=str(SE_ROOT / "config"),
    )


def load_config(
    config_name: str = "config",
    overrides: list[str] | None = None,
    *,
    config_path: str | None = None,
):
    """Load a resolved config file or compose a Hydra preset."""
    if config_path:
        from omegaconf import OmegaConf

        return OmegaConf.load(config_path)
    return compose_config(config_name=config_name, overrides=overrides)


def parse_override_csv(override: str | None) -> list[str]:
    """Parse a comma-separated override string without rewriting Hydra tokens."""
    if not override:
        return []
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None

    for char in override:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in {"[", "{", "("}:
            depth += 1
            current.append(char)
            continue
        if char in {"]", "}", ")"}:
            depth = max(0, depth - 1)
            current.append(char)
            continue
        if char == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                parts.append(token)
            current = []
            continue
        current.append(char)

    token = "".join(current).strip()
    if token:
        parts.append(token)
    return parts


def commit_volumes() -> None:
    """Persist both shared Modal volumes after a run."""
    for volume in (jax_cache_vol, outputs_vol):
        volume.commit()
