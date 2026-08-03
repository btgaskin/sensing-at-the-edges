"""Modal smoke test for sensing_edges.

Runs 5 generations with a moderate population to verify the full pipeline
works on GPU: terrain -> env build -> phased episode -> NCA -> fitness -> QD archive.

Usage:
    modal run scripts/modal_smoke.py
    modal run scripts/modal_smoke.py::smoke
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import modal

from _modal_common import VOLUMES, app, commit_volumes, compose_config, se_image


# --- Smoke test ---


@app.function(
    image=se_image,
    gpu="L4",
    timeout=1800,
    volumes=VOLUMES,
)
def smoke(
    n_gens: int = 2,
    max_steps: int = 200,
    n_emitters: int = 2,
    total_population: int = 50,
    n_episodes: int = 2,
    physics_substeps: int = 15,
    sense_act_phases: int = 3,
):
    """Run a minimal training loop and verify outputs."""
    import os

    os.chdir("/root/experiment")
    from sensing_edges.logging_utils import configure_logging

    configure_logging("INFO")

    import jax
    import logging
    log = logging.getLogger("smoke")
    log.info("Backend: %s  Devices: %s", jax.default_backend(), jax.devices())

    if total_population <= 0:
        raise ValueError("total_population must be positive")
    if n_emitters <= 0:
        raise ValueError("n_emitters must be positive")
    if total_population % n_emitters != 0:
        raise ValueError("total_population must be divisible by n_emitters")
    emitter_pop_size = total_population // n_emitters
    run_id = time.strftime("smoke_%Y%m%d_%H%M%S")
    output_dir = Path("/outputs") / run_id
    if os.environ.get("SE_PROFILE_DEVICE_MEMORY"):
        os.environ["SE_DEVICE_MEMORY_PROFILE_PATH"] = str(
            output_dir / "device_memory_profile.pb"
        )

    cfg = compose_config(
        config_name="config",
        overrides=[
            f"train.n_generations={n_gens}",
            f"train.max_steps={max_steps}",
            f"train.n_episodes={n_episodes}",
            f"train.checkpoint_interval={n_gens}",
            f"qd.n_emitters={n_emitters}",
            f"qd.emitter_pop_size={emitter_pop_size}",
            f"sim.physics_substeps={physics_substeps}",
            f"sim.sense_act_phases_per_env_step={sense_act_phases}",
            "wandb.enabled=false",
            f"train.output_dir={output_dir}",
            "jax.cache.enabled=false",
        ],
    )

    from sensing_edges.train import train
    try:
        archive = train(cfg)

        # --- Assertions ---
        metrics_path = output_dir / "metrics.jsonl"
        manifest_path = output_dir / "manifest.json"

        assert metrics_path.exists(), f"metrics.jsonl not found at {metrics_path}"
        assert manifest_path.exists(), f"manifest.json not found at {manifest_path}"

        lines = metrics_path.read_text().strip().split("\n")
        assert len(lines) == n_gens, f"Expected {n_gens} metric rows, got {len(lines)}"

        last = json.loads(lines[-1])
        log.info("Final gen metrics: %s", last)

        for key in ["best_fitness", "qd_score", "coverage"]:
            val = last.get(key)
            assert val is not None, f"Missing metric: {key}"
            assert str(val) != "nan", f"NaN in metric: {key}"

        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "completed", f"Run status: {manifest['status']}"

        log.info(
            "SMOKE PASSED: best=%.2f  cov=%.4f  qd=%.2f  gens=%d",
            archive.best_fitness, archive.coverage, archive.qd_score, n_gens,
        )

        return {
            "status": "passed",
            "best_fitness": float(archive.best_fitness),
            "coverage": float(archive.coverage),
            "qd_score": float(archive.qd_score),
            "backend": str(jax.default_backend()),
            "output_dir": str(output_dir),
        }
    finally:
        commit_volumes()


@app.local_entrypoint()
def main():
    result = smoke.remote()
    print(f"\nSmoke test result: {result}")
    if result["status"] == "passed":
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED")
        raise SystemExit(1)
