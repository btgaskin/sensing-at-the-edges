from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from free_space_branch_common import (
    DEFAULT_BUNDLE_ROOT,
    build_branch_env,
    camera_spec_from_sequences,
    default_analysis_dir,
    encode_frames,
    ensure_import_path,
    load_branch_trace,
    load_cfg,
    write_json,
)


def load_summary(trace_dir: Path) -> dict[str, object]:
    return json.loads((trace_dir / "summary.json").read_text())


def load_font(*, size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def empty_overrides(render_info) -> np.ndarray:
    return np.full((int(render_info.cell_mask.shape[0]), 3), -1, dtype=np.int32)


def lock_rigid_body_positions(positions_seq: np.ndarray, *, render_info) -> np.ndarray:
    robot_point_indices = np.asarray(render_info.robot_point_indices, dtype=np.int32)
    if robot_point_indices.size == 0:
        return positions_seq
    reference_points = positions_seq[0, robot_point_indices, :].astype(np.float64, copy=False)
    reference_com = reference_points.mean(axis=0)
    reference_centered = reference_points - reference_com
    aligned = positions_seq.astype(np.float64, copy=True)

    for frame_idx in range(int(positions_seq.shape[0])):
        current_points = positions_seq[frame_idx, robot_point_indices, :].astype(np.float64, copy=False)
        current_com = current_points.mean(axis=0)
        current_centered = current_points - current_com
        covariance = current_centered.T @ reference_centered
        u, _, vt = np.linalg.svd(covariance, full_matrices=False)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        aligned[frame_idx] = (aligned[frame_idx] - current_com) @ rotation + reference_com

    return aligned.astype(np.float32, copy=False)


def render_robot_images(
    *,
    positions_seq: np.ndarray,
    spring_seq: np.ndarray,
    render_info,
    frame_phase_indices: np.ndarray,
    local_cell_index: int,
    perturb_phase_start: int,
    branch_is_perturbed: bool,
    fps: int,
    width: int,
    height: int,
    supersample: int,
    viewport_width: float | None,
    camera_center: np.ndarray | None,
    camera_mode: str,
    camera_padding: float,
) -> list[Image.Image]:
    ensure_import_path()
    import jax.numpy as jnp
    from jax_evogym.render import RenderConfig, render_episode
    from jax_evogym.types import RenderStepOutput

    config = RenderConfig(
        enabled=True,
        width=int(width),
        height=int(height),
        fps=int(fps),
        real_time=False,
        frame_skip=1,
        show_grid=False,
        show_edges=True,
        dynamic_actuator_colors=True,
        camera_padding=float(camera_padding),
        viewport_width=1.0 if viewport_width is None else float(viewport_width),
        camera_mode=str(camera_mode),
        camera_smoothing=0.0,
        supersample=int(supersample),
        edge_width_px=3,
    )
    del local_cell_index
    del perturb_phase_start
    del branch_is_perturbed
    del camera_center

    sampled_positions = positions_seq[frame_phase_indices]
    sampled_springs = spring_seq[frame_phase_indices]
    step_count = int(sampled_positions.shape[0])
    render_outputs = RenderStepOutput(
        positions=jnp.asarray(sampled_positions, dtype=jnp.float32),
        spring_rest_length=jnp.asarray(sampled_springs, dtype=jnp.float32),
        reward=jnp.zeros((step_count,), dtype=jnp.float32),
        done=jnp.zeros((step_count,), dtype=jnp.bool_),
    )
    return [frame.copy() for frame in render_episode(render_outputs, render_info, config)]


def compose_frame(
    *,
    title: str,
    top_image: Image.Image,
    bottom_image: Image.Image,
    frame_idx: int,
    phase_idx: int,
    total_phases: int,
    sense_act_phases: int,
    flash: bool,
    top_label: str,
    bottom_label: str,
    event_label: str,
) -> Image.Image:
    margin_x = 32
    margin_y = 24
    row_gap = 22
    header_h = 76
    footer_h = 44
    panel_w = int(top_image.width)
    panel_h = int(top_image.height)
    canvas = Image.new(
        "RGBA",
        (panel_w + (2 * margin_x), header_h + footer_h + (2 * panel_h) + row_gap + (2 * margin_y)),
        (251, 250, 247, 255),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(size=34, bold=True)
    label_font = load_font(size=24, bold=True)
    meta_font = load_font(size=20, bold=False)

    top_y = margin_y + header_h
    bottom_y = top_y + panel_h + row_gap
    panel_x = margin_x

    draw.text((margin_x, 22), title, fill=(28, 25, 23), font=title_font)
    draw.text((margin_x, top_y - 36), top_label, fill=(37, 99, 235), font=label_font)
    draw.text((margin_x, bottom_y - 36), bottom_label, fill=(217, 119, 6), font=label_font)

    env_step = int(phase_idx) // max(1, int(sense_act_phases))
    status = f"Phase {int(phase_idx) + 1} / {int(total_phases)}   Env step {env_step + 1}"
    draw.text((panel_x + panel_w - 12, 32), status, fill=(87, 83, 78), font=meta_font, anchor="ra")

    canvas.alpha_composite(top_image.convert("RGBA"), dest=(panel_x, top_y))
    canvas.alpha_composite(bottom_image.convert("RGBA"), dest=(panel_x, bottom_y))

    outline = (214, 211, 209, 255)
    draw.rectangle(
        [panel_x, top_y, panel_x + panel_w - 1, top_y + panel_h - 1],
        outline=outline,
        width=2,
    )
    draw.rectangle(
        [panel_x, bottom_y, panel_x + panel_w - 1, bottom_y + panel_h - 1],
        outline=outline,
        width=2,
    )

    if flash:
        overlay = Image.new("RGBA", canvas.size, (255, 243, 182, 76))
        canvas = Image.alpha_composite(canvas, overlay)
        draw = ImageDraw.Draw(canvas)
        flash_font = load_font(size=42, bold=True)
        draw.text(
            (canvas.width // 2, 30),
            event_label,
            fill=(159, 18, 57),
            font=flash_font,
            anchor="ma",
        )

    return canvas.convert("RGB")


def render_mode(
    *,
    trace_dir: Path,
    output_dir: Path,
    mode: str,
    trace: dict[str, np.ndarray],
    render_info,
    sense_act_phases: int,
    local_cell_index: int,
    fps: int,
    max_frames: int,
    flash_frames: int,
    supersample: int,
    top_label: str,
    bottom_label: str,
    event_label: str,
    lock_body_frame: bool,
) -> dict[str, object]:
    baseline_positions = np.asarray(trace["baseline_positions"], dtype=np.float32)
    perturbed_positions = np.asarray(trace["perturbed_positions"], dtype=np.float32)
    if lock_body_frame:
        baseline_positions = lock_rigid_body_positions(baseline_positions, render_info=render_info)
        perturbed_positions = lock_rigid_body_positions(perturbed_positions, render_info=render_info)

    phase_count = int(baseline_positions.shape[0])
    perturb_phase_start = int(trace["perturb_phase_index"][0])
    frame_phase_indices = np.linspace(0, phase_count - 1, num=min(max_frames, phase_count), dtype=np.int32)
    if perturb_phase_start not in frame_phase_indices:
        frame_phase_indices = np.unique(
            np.concatenate([frame_phase_indices, np.asarray([perturb_phase_start], dtype=np.int32)], axis=0)
        ).astype(np.int32, copy=False)
    frame_phase_indices.sort()

    if mode == "tight":
        title = "Robot Tight"
        width = 1280
        height = 520
        viewport_width = None
        camera_center = None
        camera_mode = "fit_robot"
        camera_padding = 0.25
    elif mode == "world":
        title = "Robot World"
        width = 1280
        height = 400
        camera_mode = "fixed"
        camera_padding = 0.0
        _, viewport_width = camera_spec_from_sequences(
            baseline_positions,
            perturbed_positions,
            render_info=render_info,
            sample_phase_indices_arr=frame_phase_indices,
            panel_aspect=float(width) / float(height),
        )
        camera_center = None
    else:
        raise ValueError(f"Unsupported camera mode: {mode}")

    baseline_images = render_robot_images(
        positions_seq=baseline_positions,
        spring_seq=trace["baseline_spring_rest_length"],
        render_info=render_info,
        frame_phase_indices=frame_phase_indices,
        local_cell_index=int(local_cell_index),
        perturb_phase_start=int(perturb_phase_start),
        branch_is_perturbed=False,
        fps=int(fps),
        width=int(width),
        height=int(height),
        supersample=int(supersample),
        viewport_width=viewport_width,
        camera_center=camera_center,
        camera_mode=camera_mode,
        camera_padding=float(camera_padding),
    )
    perturbed_images = render_robot_images(
        positions_seq=perturbed_positions,
        spring_seq=trace["perturbed_spring_rest_length"],
        render_info=render_info,
        frame_phase_indices=frame_phase_indices,
        local_cell_index=int(local_cell_index),
        perturb_phase_start=int(perturb_phase_start),
        branch_is_perturbed=True,
        fps=int(fps),
        width=int(width),
        height=int(height),
        supersample=int(supersample),
        viewport_width=viewport_width,
        camera_center=camera_center,
        camera_mode=camera_mode,
        camera_padding=float(camera_padding),
    )

    frames_dir = output_dir / mode / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    perturb_frame_idx = int(np.searchsorted(frame_phase_indices, perturb_phase_start))
    for idx, phase_idx in enumerate(frame_phase_indices):
        frame = compose_frame(
            title=title,
            top_image=baseline_images[idx],
            bottom_image=perturbed_images[idx],
            frame_idx=int(idx),
            phase_idx=int(phase_idx),
            total_phases=int(phase_count),
            sense_act_phases=int(sense_act_phases),
            flash=bool(perturb_frame_idx <= int(idx) < (perturb_frame_idx + int(flash_frames))),
            top_label=str(top_label),
            bottom_label=str(bottom_label),
            event_label=str(event_label),
        )
        frame.save(frames_dir / f"frame_{idx:04d}.png")

    mp4_path = output_dir / mode / f"robot_{mode}.mp4"
    gif_path = output_dir / mode / f"robot_{mode}.gif"
    encode_frames(frames_dir, fps=int(fps), mp4_path=mp4_path, gif_path=gif_path)

    payload = {
        "trace_dir": str(trace_dir),
        "mode": mode,
        "mp4_path": str(mp4_path),
        "gif_path": str(gif_path),
        "frame_count": int(frame_phase_indices.shape[0]),
        "fps": int(fps),
        "flash_frames": int(flash_frames),
        "phase_count": int(phase_count),
        "perturb_phase_index": int(perturb_phase_start),
        "lock_body_frame": bool(lock_body_frame),
    }
    write_json(output_dir / mode / "summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render stacked natural robot videos from a saved free-space branched trace.",
    )
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <trace-dir>/robot_renders.",
    )
    parser.add_argument(
        "--camera-mode",
        type=str,
        choices=("both", "tight", "world"),
        default="both",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--flash-frames", type=int, default=8)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument(
        "--lock-body-frame",
        "--lock-com",
        dest="lock_body_frame",
        action="store_true",
        help="Render in a body-fixed frame so the robots stay visually pinned without translating or rotating.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = (
        args.trace_dir.resolve()
        if args.trace_dir is not None
        else default_analysis_dir(args.bundle_root.resolve(), max_steps=int(args.max_steps)).resolve()
    )
    if not (trace_dir / "branch_trace.npz").exists():
        raise FileNotFoundError(
            f"Missing branch trace under {trace_dir}. Run free_space_branch_capture.py first."
        )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (trace_dir / "robot_renders").resolve()
    )
    summary = load_summary(trace_dir)
    cfg = load_cfg(Path(str(summary["config_snapshot_path"])), max_steps=int(summary["max_steps"]))
    env, render_info, _alive_mask = build_branch_env(
        cfg,
        environment_mode=str(summary.get("environment_mode", "free_space")),
        ground_width=int(summary.get("ground_width", 20)),
        ground_clearance_cells=float(summary.get("ground_clearance_cells", 0.1)),
    )
    trace = load_branch_trace(trace_dir)
    local_cell_index = int(trace["perturb_local_cell_index"][0])
    sense_act_phases = int(trace["sense_act_phases"][0])
    top_label = str(summary.get("row_label_top", "Baseline"))
    bottom_label = str(summary.get("row_label_bottom", "Perturbed"))
    event_label = str(summary.get("event_label", "PERTURBATION"))

    modes = ["tight", "world"] if args.camera_mode == "both" else [str(args.camera_mode)]
    results = []
    for mode in modes:
        results.append(
            render_mode(
                trace_dir=trace_dir,
                output_dir=output_dir,
                mode=mode,
                trace=trace,
                render_info=render_info,
                sense_act_phases=int(sense_act_phases),
                local_cell_index=int(local_cell_index),
                fps=int(args.fps),
                max_frames=int(args.max_frames),
                flash_frames=int(args.flash_frames),
                supersample=int(args.supersample),
                top_label=top_label,
                bottom_label=bottom_label,
                event_label=event_label,
                lock_body_frame=bool(args.lock_body_frame),
            )
        )

    write_json(output_dir / "summary.json", {"trace_dir": str(trace_dir), "renders": results})
    for result in results:
        print(f"{result['mode']}: {result['mp4_path']}")
        print(f"{result['mode']}: {result['gif_path']}")


if __name__ == "__main__":
    main()
