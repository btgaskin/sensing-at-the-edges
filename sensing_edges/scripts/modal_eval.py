"""Modal entrypoint for held-out evaluation of trained archives.

Usage:
    modal run scripts/modal_eval.py \
        --checkpoint /outputs/paper_run_rect_7x2_seed137/checkpoints/gen_0149 \
        --rerank-schedule terrain_suites/seed_137_bilateral_goal_cliff_asymmetric_pairs_7x2_grid30_g150_ep6_val6_test6 \
        --eval-schedule terrain_suites/seed_137_bilateral_goal_cliff_asymmetric_pairs_7x2_grid30_g150_ep6_val6_test6 \
        --config-path /outputs/paper_run_rect_7x2_seed137/resolved_config.yaml
"""

from __future__ import annotations

import json

import modal

from _modal_common import VOLUMES, app, commit_volumes, load_config, parse_override_csv, se_image


# --- Evaluation ---


def _eval_result_payload(*, results, output: str) -> dict[str, object]:
    """Return a Modal-safe eval payload using only Python scalars."""
    return {
        "test_mean": float(results.test_mean_fitness),
        "test_median": float(results.test_median_fitness),
        "ci95": [float(results.test_ci95_low), float(results.test_ci95_high)],
        "selected_elite_rank": int(results.selected_elite_rank),
        "selected_elite_flat_index": int(results.selected_elite_flat_index),
        "output": str(output),
    }


@app.function(
    image=se_image,
    gpu="L4",
    timeout=3600,
    volumes=VOLUMES,
)
def eval_remote(
    checkpoint: str,
    rerank_schedule: str | None = None,
    eval_schedule: str | None = None,
    schedule: str | None = None,
    preset: str = "config",
    overrides: list[str] | None = None,
    config_path: str | None = None,
    top_k: int = 10,
    ou_noise_seed: int = 12345,
    output: str | None = None,
    publication_summary_output: str | None = None,
):
    """Run held-out evaluation on GPU."""
    import os
    from pathlib import Path

    os.chdir("/root/experiment")
    from sensing_edges.logging_utils import configure_logging

    configure_logging("INFO")

    cfg = load_config(config_name=preset, overrides=overrides, config_path=config_path)

    if output is None:
        output = f"{checkpoint}_eval_results.json"
    if rerank_schedule is None and eval_schedule is None:
        if schedule is None:
            raise ValueError("Provide either --schedule or both --rerank-schedule and --eval-schedule.")
        rerank_schedule = schedule
        eval_schedule = schedule
    if rerank_schedule is None or eval_schedule is None:
        raise ValueError("Both rerank_schedule and eval_schedule must be set together.")

    from sensing_edges.checkpoint import is_payload_complete
    from sensing_edges.evaluate import evaluate_checkpoint_with_schedules
    from sensing_edges.publication import canonicalize_post_train_eval_payload

    try:
        results = evaluate_checkpoint_with_schedules(
            checkpoint_path=checkpoint,
            rerank_schedule_dir=rerank_schedule,
            eval_schedule_dir=eval_schedule,
            cfg=cfg,
            top_k=top_k,
            ou_noise_seed=ou_noise_seed,
            output_path=output if publication_summary_output is None else None,
        )
        if publication_summary_output is not None:
            run_dir = Path(checkpoint).parent.parent
            manifest_path = run_dir / "manifest.json"
            config_resolved_path = Path(config_path) if config_path is not None else run_dir / "resolved_config.yaml"
            final_archive_prefix = run_dir / "final_archive"
            with open(manifest_path) as handle:
                manifest = json.load(handle)

            summary_payload = canonicalize_post_train_eval_payload(
                results.as_json_dict(),
                release_id=Path(publication_summary_output).parent.parent.name,
                run_id=str(manifest.get("run_id", run_dir.name)),
                run_dir_name=run_dir.name,
                manifest_path=manifest_path,
                config_path=config_resolved_path if config_resolved_path.exists() else None,
                checkpoint_prefix=Path(checkpoint),
                final_archive_prefix=final_archive_prefix if is_payload_complete(final_archive_prefix, require_done=True) else None,
                source_path=None,
            )
            target = Path(publication_summary_output)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w") as handle:
                json.dump(summary_payload, handle, indent=2)
        return _eval_result_payload(results=results, output=output)
    finally:
        commit_volumes()


@app.local_entrypoint()
def main(
    checkpoint: str,
    rerank_schedule: str | None = None,
    eval_schedule: str | None = None,
    schedule: str | None = None,
    preset: str = "config",
    config_path: str | None = None,
    top_k: int = 10,
    override: str | None = None,
    output: str | None = None,
    publication_summary_output: str | None = None,
):
    overrides = parse_override_csv(override) or None
    result = eval_remote.remote(
        checkpoint=checkpoint,
        rerank_schedule=rerank_schedule,
        eval_schedule=eval_schedule,
        schedule=schedule,
        preset=preset,
        config_path=config_path,
        overrides=overrides,
        top_k=top_k,
        output=output,
        publication_summary_output=publication_summary_output,
    )
    print(f"\nEvaluation result: {result}")
