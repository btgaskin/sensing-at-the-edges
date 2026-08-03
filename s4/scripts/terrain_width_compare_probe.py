from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from free_space_branch_common import (
    DEFAULT_BUNDLE_ROOT,
    build_flat_ground_env,
    compute_raw_abs_hv_ratios,
    encode_frames,
    ensure_import_path,
    fit_pca,
    flatten_phase_trace,
    load_cfg,
    load_controller_params,
    reduce_controller_trace,
    run_controller_trace,
    run_phase_probe,
    transform_pca,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = DEFAULT_BUNDLE_ROOT
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "analysis_runs" / "terrain_compare_steps_400_plateau25_ledge15"
DEFAULT_CONFIG_PATH = RESULT_ROOT / "config_snapshot.yaml"
DEFAULT_PARAMS_PATH = RESULT_ROOT / "controller_params.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local terrain comparison probe on centered plateau and ledge terrains.",
    )
    parser.add_argument("--config-snapshot-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--controller-params-path", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--plateau-width", type=int, default=25)
    parser.add_argument("--ledge-width", type=int, default=15)
    parser.add_argument("--grid-width", type=int, default=None)
    parser.add_argument("--ground-clearance-cells", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--flash-frames", type=int, default=8)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=("intact", "deformation_off"),
        help="Subset of deformation conditions to run.",
    )
    return parser.parse_args()


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


def _platform_bounds(cfg, *, width: int) -> tuple[int, int]:
    grid_w = int(cfg.terrain.grid_w)
    platform_start = int((grid_w - int(width)) // 2)
    platform_end = int(platform_start + int(width) - 1)
    return platform_start, platform_end


def _capture_one(
    *,
    cfg,
    params,
    ground_width: int,
    ground_clearance_cells: float,
    seed: int,
) -> dict[str, Any]:
    env, render_info, alive_mask = build_flat_ground_env(
        cfg,
        ground_width=int(ground_width),
        ground_clearance_cells=float(ground_clearance_cells),
    )
    phase_result, phase_trace = run_phase_probe(
        env=env,
        params=params,
        alive_mask=alive_mask,
        cfg=cfg,
        seed=int(seed),
        max_steps=int(cfg.train.max_steps),
    )
    controller_result, controller_trace = run_controller_trace(
        env=env,
        params=params,
        alive_mask=alive_mask,
        cfg=cfg,
        seed=int(seed),
        max_steps=int(cfg.train.max_steps),
    )
    phase_arrays = flatten_phase_trace(phase_trace)
    controller_phase = reduce_controller_trace(
        controller_trace,
        nca_steps_per_phase=int(cfg.nca.nca_steps_per_phase),
    )
    positions = np.asarray(phase_arrays["phase_post_positions"], dtype=np.float32)
    spring_rest = np.asarray(phase_arrays["phase_post_spring_rest_length"], dtype=np.float32)
    robot_point_mask = np.asarray(env.robot_point_mask, dtype=bool)
    if not np.any(robot_point_mask):
        raise ValueError("Robot point mask is empty.")
    robot_positions = positions[:, robot_point_mask, :]
    plant_features = compute_raw_abs_hv_ratios(
        positions,
        topology=env.topology,
        spring_init_rest_length=env.sim_state.spring_init_rest_length,
        deformation_info=env.deformation_info,
    )
    return {
        "env": env,
        "render_info": render_info,
        "alive_mask": alive_mask,
        "phase_result": phase_result,
        "controller_result": controller_result,
        "phase_arrays": phase_arrays,
        "controller_phase": controller_phase,
        "positions": positions,
        "robot_point_mask": robot_point_mask,
        "spring_rest": spring_rest,
        "plant_features": plant_features,
        "controller_features": np.asarray(controller_phase["controller_flat"], dtype=np.float32),
        "front_x_phase": np.max(robot_positions[..., 0], axis=1).astype(np.float32, copy=False),
        "back_x_phase": np.min(robot_positions[..., 0], axis=1).astype(np.float32, copy=False),
    }


def _first_edge_phase(
    *,
    back_x_phase: np.ndarray,
    front_x_phase: np.ndarray,
    left_frontier_x: float,
    right_frontier_x: float,
) -> int | None:
    over_left = np.asarray(back_x_phase, dtype=np.float32) < float(left_frontier_x)
    over_right = np.asarray(front_x_phase, dtype=np.float32) > float(right_frontier_x)
    hits = np.flatnonzero(over_left | over_right)
    if hits.size == 0:
        return None
    return int(hits[0])


def _valid_event_phase_index(event_phase_index: int | None, *, phase_count: int) -> int | None:
    if event_phase_index is None:
        return None
    event_phase_index = int(event_phase_index)
    if 0 <= event_phase_index < int(phase_count):
        return event_phase_index
    return None


def _trim_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    count = int(min(a.shape[0], b.shape[0]))
    return a[:count], b[:count], count


def _save_trace_bundle(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _robot_corner_com_features(
    *,
    positions_seq: np.ndarray,
    robot_point_mask: np.ndarray,
) -> np.ndarray:
    robot_positions = np.asarray(positions_seq[:, np.asarray(robot_point_mask, dtype=bool), :], dtype=np.float32)
    if robot_positions.ndim != 3 or robot_positions.shape[1] <= 0:
        raise ValueError("Robot point mask must select at least one robot point.")
    com_xy = np.mean(robot_positions, axis=1, dtype=np.float32)
    centered = robot_positions - com_xy[:, None, :]
    x = robot_positions[..., 0]
    y = robot_positions[..., 1]
    bl_idx = np.argmin(x + y, axis=1)
    br_idx = np.argmax(x - y, axis=1)
    tr_idx = np.argmax(x + y, axis=1)
    tl_idx = np.argmin(x - y, axis=1)
    phase_idx = np.arange(robot_positions.shape[0], dtype=np.int32)
    corners = np.stack(
        [
            centered[phase_idx, bl_idx],
            centered[phase_idx, br_idx],
            centered[phase_idx, tr_idx],
            centered[phase_idx, tl_idx],
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    return np.concatenate(
        [
            np.asarray(com_xy, dtype=np.float32),
            corners.reshape(corners.shape[0], -1),
        ],
        axis=1,
    )


def _render_robot_images(
    *,
    positions_seq: np.ndarray,
    spring_seq: np.ndarray,
    render_info,
    frame_phase_indices: np.ndarray,
    fps: int,
    width: int,
    height: int,
    supersample: int,
    viewport_width: float,
):
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
        camera_padding=0.0,
        viewport_width=float(viewport_width),
        camera_mode="fixed",
        camera_smoothing=0.0,
        supersample=int(supersample),
        edge_width_px=3,
    )
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


def _compose_robot_frame(
    *,
    top_image: Image.Image,
    bottom_image: Image.Image,
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

    draw.text((margin_x, 22), "Robot Terrain Compare", fill=(28, 25, 23), font=title_font)
    draw.text((margin_x, top_y - 36), top_label, fill=(37, 99, 235), font=label_font)
    draw.text((margin_x, bottom_y - 36), bottom_label, fill=(217, 119, 6), font=label_font)
    env_step = int(phase_idx) // max(1, int(sense_act_phases))
    status = f"Phase {int(phase_idx) + 1} / {int(total_phases)}   Env step {env_step + 1}"
    draw.text((panel_x + panel_w - 12, 32), status, fill=(87, 83, 78), font=meta_font, anchor="ra")

    canvas.alpha_composite(top_image.convert("RGBA"), dest=(panel_x, top_y))
    canvas.alpha_composite(bottom_image.convert("RGBA"), dest=(panel_x, bottom_y))
    outline = (214, 211, 209, 255)
    draw.rectangle([panel_x, top_y, panel_x + panel_w - 1, top_y + panel_h - 1], outline=outline, width=2)
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


def _render_robot_compare(
    *,
    output_dir: Path,
    plateau_run: dict[str, Any],
    ledge_run: dict[str, Any],
    phase_count: int,
    sense_act_phases: int,
    event_phase_index: int | None,
    fps: int,
    max_frames: int,
    flash_frames: int,
    supersample: int,
    top_width_cells: int,
    bottom_width_cells: int,
    top_label: str,
    bottom_label: str,
) -> dict[str, Any]:
    from jax_evogym.constants import CELL_SIZE

    event_phase_index = _valid_event_phase_index(event_phase_index, phase_count=int(phase_count))
    frame_phase_indices = np.linspace(0, phase_count - 1, num=min(int(max_frames), int(phase_count)), dtype=np.int32)
    if event_phase_index is not None and int(event_phase_index) not in frame_phase_indices:
        frame_phase_indices = np.unique(
            np.concatenate([frame_phase_indices, np.asarray([event_phase_index], dtype=np.int32)], axis=0)
        ).astype(np.int32, copy=False)
        frame_phase_indices.sort()

    viewport_width = float(max(int(top_width_cells), int(bottom_width_cells)) + 6) * float(CELL_SIZE)
    plateau_images = _render_robot_images(
        positions_seq=plateau_run["positions"][:phase_count],
        spring_seq=plateau_run["spring_rest"][:phase_count],
        render_info=plateau_run["render_info"],
        frame_phase_indices=frame_phase_indices,
        fps=int(fps),
        width=1280,
        height=420,
        supersample=int(supersample),
        viewport_width=viewport_width,
    )
    ledge_images = _render_robot_images(
        positions_seq=ledge_run["positions"][:phase_count],
        spring_seq=ledge_run["spring_rest"][:phase_count],
        render_info=ledge_run["render_info"],
        frame_phase_indices=frame_phase_indices,
        fps=int(fps),
        width=1280,
        height=420,
        supersample=int(supersample),
        viewport_width=viewport_width,
    )

    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    event_frame_index = None
    if event_phase_index is not None:
        event_frame_index = int(np.searchsorted(frame_phase_indices, int(event_phase_index)))
    for idx, phase_idx in enumerate(frame_phase_indices):
        frame = _compose_robot_frame(
            top_image=plateau_images[idx],
            bottom_image=ledge_images[idx],
            phase_idx=int(phase_idx),
            total_phases=int(phase_count),
            sense_act_phases=int(sense_act_phases),
            flash=bool(
                event_frame_index is not None
                and int(event_frame_index) <= int(idx) < (int(event_frame_index) + int(flash_frames))
            ),
            top_label=str(top_label),
            bottom_label=str(bottom_label),
            event_label="OVER EDGE",
        )
        frame.save(frames_dir / f"frame_{idx:04d}.png")

    mp4_path = output_dir / "robot_compare.mp4"
    gif_path = output_dir / "robot_compare.gif"
    encode_frames(frames_dir, fps=int(fps), mp4_path=mp4_path, gif_path=gif_path)
    return {
        "mp4_path": str(mp4_path),
        "gif_path": str(gif_path),
        "frame_count": int(frame_phase_indices.shape[0]),
        "event_phase_index": None if event_phase_index is None else int(event_phase_index),
    }


def _limits_nd(xy_a: np.ndarray, xy_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate([xy_a, xy_b], axis=0)
    mins = np.min(pooled, axis=0).astype(np.float32, copy=False)
    maxs = np.max(pooled, axis=0).astype(np.float32, copy=False)
    spans = np.maximum(maxs - mins, np.float32(1.0))
    pads = np.maximum(np.float32(0.08), np.float32(0.1) * spans)
    return mins - pads, maxs + pads


def _trail_payload(
    xyz: np.ndarray,
    *,
    phase_idx: int,
    rgb: tuple[float, float, float],
    min_alpha: float,
    max_alpha: float,
    min_width: float,
    max_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trace = np.asarray(xyz[: phase_idx + 1], dtype=np.float32)
    if trace.shape[0] < 2:
        return (
            np.zeros((0, 2, trace.shape[1]), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    segments = np.stack([trace[:-1], trace[1:]], axis=1)
    segment_count = int(segments.shape[0])
    fade = np.linspace(0.0, 1.0, num=segment_count, dtype=np.float32)
    alphas = min_alpha + (max_alpha - min_alpha) * np.square(fade)
    widths = min_width + (max_width - min_width) * np.square(fade)
    colors = np.tile(np.asarray([rgb[0], rgb[1], rgb[2], 1.0], dtype=np.float32), (segment_count, 1))
    colors[:, 3] = alphas
    return segments, colors, widths


def _recent_trace_correlation(
    trace_a: np.ndarray,
    trace_b: np.ndarray,
    *,
    phase_idx: int,
    window: int = 240,
) -> float:
    end = int(phase_idx) + 1
    start = max(0, end - int(window))
    a = np.asarray(trace_a[start:end], dtype=np.float32).reshape(-1)
    b = np.asarray(trace_b[start:end], dtype=np.float32).reshape(-1)
    if a.size < 4 or b.size != a.size:
        return float("nan")
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denom) or denom <= 1.0e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _render_state_pca(
    *,
    output_dir: Path,
    plateau_plant_xy: np.ndarray,
    ledge_plant_xy: np.ndarray,
    plateau_ctrl_xy: np.ndarray,
    ledge_ctrl_xy: np.ndarray,
    plateau_aug_xy: np.ndarray | None = None,
    ledge_aug_xy: np.ndarray | None = None,
    aug_title: str = "Plant PCA + edge pose",
    env_steps: np.ndarray,
    event_phase_index: int | None,
    fps: int,
    max_frames: int,
    flash_frames: int,
) -> dict[str, Any]:
    phase_count = min(
        int(plateau_plant_xy.shape[0]),
        int(ledge_plant_xy.shape[0]),
        int(plateau_ctrl_xy.shape[0]),
        int(ledge_ctrl_xy.shape[0]),
        int(np.asarray(env_steps).shape[0]),
        int(plateau_aug_xy.shape[0]) if plateau_aug_xy is not None else int(plateau_plant_xy.shape[0]),
        int(ledge_aug_xy.shape[0]) if ledge_aug_xy is not None else int(ledge_plant_xy.shape[0]),
    )
    plateau_plant_xy = np.asarray(plateau_plant_xy[:phase_count], dtype=np.float32)
    ledge_plant_xy = np.asarray(ledge_plant_xy[:phase_count], dtype=np.float32)
    plateau_ctrl_xy = np.asarray(plateau_ctrl_xy[:phase_count], dtype=np.float32)
    ledge_ctrl_xy = np.asarray(ledge_ctrl_xy[:phase_count], dtype=np.float32)
    env_steps = np.asarray(env_steps[:phase_count], dtype=np.int32)
    if plateau_aug_xy is not None:
        plateau_aug_xy = np.asarray(plateau_aug_xy[:phase_count], dtype=np.float32)
    if ledge_aug_xy is not None:
        ledge_aug_xy = np.asarray(ledge_aug_xy[:phase_count], dtype=np.float32)
    event_phase_index = _valid_event_phase_index(event_phase_index, phase_count=int(phase_count))
    frame_phase_indices = np.linspace(0, phase_count - 1, num=min(int(max_frames), int(phase_count)), dtype=np.int32)
    if event_phase_index is not None and int(event_phase_index) not in frame_phase_indices:
        frame_phase_indices = np.unique(
            np.concatenate([frame_phase_indices, np.asarray([event_phase_index], dtype=np.int32)], axis=0)
        ).astype(np.int32, copy=False)
        frame_phase_indices.sort()

    plant_bounds = _limits_nd(plateau_plant_xy, ledge_plant_xy)
    ctrl_bounds = _limits_nd(plateau_ctrl_xy, ledge_ctrl_xy)
    panels = [
        ("Plant PCA: raw absolute H/V state", plateau_plant_xy, ledge_plant_xy, plant_bounds),
        ("Controller PCA", plateau_ctrl_xy, ledge_ctrl_xy, ctrl_bounds),
    ]
    if plateau_aug_xy is not None and ledge_aug_xy is not None:
        aug_bounds = _limits_nd(plateau_aug_xy, ledge_aug_xy)
        panels.append((str(aug_title), plateau_aug_xy, ledge_aug_xy, aug_bounds))

    fig_height = 5.2 * float(len(panels)) + 0.6
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(11.8, fig_height),
        dpi=110,
        subplot_kw={"projection": "3d"},
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("#fbfaf7")
    flash_overlay = Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=fig.transFigure,
        facecolor="#fff4bf",
        alpha=0.0,
        zorder=10,
    )
    fig.add_artist(flash_overlay)
    event_text = fig.text(
        0.5,
        0.982,
        "OVER EDGE",
        ha="center",
        va="top",
        fontsize=24,
        fontweight="bold",
        color="#9f1239",
        alpha=0.0,
        zorder=11,
    )
    status_text = fig.text(
        0.5,
        0.948,
        "",
        ha="center",
        va="top",
        fontsize=11,
        color="#44403c",
        zorder=11,
    )
    corr_text = fig.text(
        0.97,
        0.982,
        "",
        ha="right",
        va="top",
        fontsize=11,
        color="#44403c",
        zorder=11,
    )

    trails = []
    dots = []
    for ax, (title, xy_a, xy_b, bounds) in zip(axes, panels, strict=True):
        ax.set_facecolor("#fffdf8")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(float(bounds[0][0]), float(bounds[1][0]))
        ax.set_ylim(float(bounds[0][1]), float(bounds[1][1]))
        ax.set_zlim(float(bounds[0][2]), float(bounds[1][2]))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.grid(alpha=0.16, linewidth=0.6)
        ax.view_init(elev=24, azim=-58)
        trail_a = Line3DCollection([], zorder=5)
        trail_b = Line3DCollection([], zorder=4)
        ax.add_collection(trail_a)
        ax.add_collection(trail_b)
        head_a, = ax.plot([], [], [], color="#2563eb", linewidth=2.6, solid_capstyle="round", zorder=7)
        head_b, = ax.plot([], [], [], color="#d97706", linewidth=2.6, solid_capstyle="round", zorder=6)
        dot_a = ax.scatter([], [], [], s=132, facecolors="white", edgecolors="#2563eb", linewidths=2.3, marker="o", zorder=9)
        dot_b = ax.scatter([], [], [], s=132, color="#d97706", linewidths=2.3, marker="X", zorder=8)
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color="#2563eb", linewidth=2.0, label="Plateau"),
                plt.Line2D([0], [0], color="#d97706", linewidth=2.0, label="Ledge"),
            ],
            loc="upper right",
            frameon=False,
        )
        trails.append((trail_a, trail_b, head_a, head_b))
        dots.append((dot_a, dot_b))

    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.92])

    event_frame_index = None
    if event_phase_index is not None:
        event_frame_index = int(np.searchsorted(frame_phase_indices, int(event_phase_index)))
    for frame_idx, phase_idx in enumerate(frame_phase_indices):
        for seq_idx, (_title, xy_a, xy_b, _bounds) in enumerate(panels):
            trail_a, trail_b, head_a, head_b = trails[seq_idx]
            dot_a, dot_b = dots[seq_idx]
            seg_a, color_a, width_a = _trail_payload(
                xy_a,
                phase_idx=int(phase_idx),
                rgb=(37 / 255.0, 99 / 255.0, 235 / 255.0),
                min_alpha=0.015,
                max_alpha=0.20,
                min_width=0.20,
                max_width=0.85,
            )
            seg_b, color_b, width_b = _trail_payload(
                xy_b,
                phase_idx=int(phase_idx),
                rgb=(217 / 255.0, 119 / 255.0, 6 / 255.0),
                min_alpha=0.015,
                max_alpha=0.20,
                min_width=0.20,
                max_width=0.85,
            )
            trail_a.set_segments(seg_a)
            trail_a.set_color(color_a)
            trail_a.set_linewidths(width_a)
            trail_b.set_segments(seg_b)
            trail_b.set_color(color_b)
            trail_b.set_linewidths(width_b)
            head_window = min(48, int(phase_idx) + 1)
            head_trace_a = np.asarray(xy_a[max(0, int(phase_idx) + 1 - head_window) : int(phase_idx) + 1], dtype=np.float32)
            head_trace_b = np.asarray(xy_b[max(0, int(phase_idx) + 1 - head_window) : int(phase_idx) + 1], dtype=np.float32)
            head_a.set_data(head_trace_a[:, 0], head_trace_a[:, 1])
            head_a.set_3d_properties(head_trace_a[:, 2])
            head_b.set_data(head_trace_b[:, 0], head_trace_b[:, 1])
            head_b.set_3d_properties(head_trace_b[:, 2])
            current_a = np.asarray(xy_a[phase_idx], dtype=np.float32)
            current_b = np.asarray(xy_b[phase_idx], dtype=np.float32)
            dot_a._offsets3d = ([float(current_a[0])], [float(current_a[1])], [float(current_a[2])])
            dot_b._offsets3d = ([float(current_b[0])], [float(current_b[1])], [float(current_b[2])])

        flash = bool(
            event_frame_index is not None
            and int(event_frame_index) <= int(frame_idx) < (int(event_frame_index) + int(flash_frames))
        )
        flash_overlay.set_alpha(0.24 if flash else 0.0)
        event_text.set_alpha(1.0 if flash else 0.0)
        plant_corr = _recent_trace_correlation(
            plateau_plant_xy,
            ledge_plant_xy,
            phase_idx=int(phase_idx),
        )
        ctrl_corr = _recent_trace_correlation(
            plateau_ctrl_xy,
            ledge_ctrl_xy,
            phase_idx=int(phase_idx),
        )
        plant_corr_text = "nan" if not np.isfinite(plant_corr) else f"{plant_corr:+.3f}"
        ctrl_corr_text = "nan" if not np.isfinite(ctrl_corr) else f"{ctrl_corr:+.3f}"
        corr_lines = [f"Plant corr {plant_corr_text}", f"Ctrl corr {ctrl_corr_text}"]
        if plateau_aug_xy is not None and ledge_aug_xy is not None:
            aug_corr = _recent_trace_correlation(
                plateau_aug_xy,
                ledge_aug_xy,
                phase_idx=int(phase_idx),
            )
            aug_corr_text = "nan" if not np.isfinite(aug_corr) else f"{aug_corr:+.3f}"
            corr_lines.append(f"Aug corr {aug_corr_text}")
        corr_text.set_text("\n".join(corr_lines))
        status_text.set_text(f"Phase {int(phase_idx) + 1} / {phase_count}   Env step {int(env_steps[phase_idx])}")
        fig.savefig(frames_dir / f"frame_{frame_idx:04d}.png", dpi=110)

    plt.close(fig)

    mp4_path = output_dir / "state_pca.mp4"
    gif_path = output_dir / "state_pca.gif"
    encode_frames(frames_dir, fps=int(fps), mp4_path=mp4_path, gif_path=gif_path)
    return {
        "mp4_path": str(mp4_path),
        "gif_path": str(gif_path),
        "frame_count": int(frame_phase_indices.shape[0]),
        "event_phase_index": None if event_phase_index is None else int(event_phase_index),
    }


def _phase_env_steps(phase_count: int, *, sense_act_phases: int) -> np.ndarray:
    phase_idx = np.arange(int(phase_count), dtype=np.int32)
    return (phase_idx // int(sense_act_phases)) + 1


def _run_condition(
    *,
    condition_name: str,
    deformation_condition: str,
    cfg_template,
    params,
    output_dir: Path,
    plateau_width: int,
    ledge_width: int,
    ground_clearance_cells: float,
    seed: int,
    fps: int,
    max_frames: int,
    flash_frames: int,
    supersample: int,
    grid_width: int | None,
) -> dict[str, Any]:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg_template, resolve=False))
    cfg.eval.post_run.active_deformation_condition = str(deformation_condition)
    cfg.eval.post_run.active_signaling_condition = "intact"
    if grid_width is not None:
        cfg.terrain.grid_w = int(grid_width)

    plateau_run = _capture_one(
        cfg=cfg,
        params=params,
        ground_width=int(plateau_width),
        ground_clearance_cells=float(ground_clearance_cells),
        seed=int(seed),
    )
    ledge_run = _capture_one(
        cfg=cfg,
        params=params,
        ground_width=int(ledge_width),
        ground_clearance_cells=float(ground_clearance_cells),
        seed=int(seed),
    )

    plateau_positions, ledge_positions, phase_count = _trim_pair(
        plateau_run["positions"],
        ledge_run["positions"],
    )
    plateau_spring, ledge_spring, _ = _trim_pair(
        plateau_run["spring_rest"],
        ledge_run["spring_rest"],
    )
    plateau_plant, ledge_plant, _ = _trim_pair(
        plateau_run["plant_features"],
        ledge_run["plant_features"],
    )
    plateau_ctrl, ledge_ctrl, _ = _trim_pair(
        plateau_run["controller_features"],
        ledge_run["controller_features"],
    )
    plateau_front, ledge_front, _ = _trim_pair(
        plateau_run["front_x_phase"][:, None],
        ledge_run["front_x_phase"][:, None],
    )
    plateau_back, ledge_back, _ = _trim_pair(
        plateau_run["back_x_phase"][:, None],
        ledge_run["back_x_phase"][:, None],
    )
    plateau_front = plateau_front[:, 0]
    plateau_back = plateau_back[:, 0]
    ledge_front = ledge_front[:, 0]
    ledge_back = ledge_back[:, 0]

    plateau_start, plateau_end = _platform_bounds(cfg, width=int(plateau_width))
    ledge_start, ledge_end = _platform_bounds(cfg, width=int(ledge_width))
    from jax_evogym.constants import CELL_SIZE

    ledge_left_frontier = float(int(ledge_start)) * float(CELL_SIZE)
    ledge_right_frontier = float(int(ledge_end) + 1) * float(CELL_SIZE)
    event_phase_index = _first_edge_phase(
        back_x_phase=ledge_back[:phase_count],
        front_x_phase=ledge_front[:phase_count],
        left_frontier_x=ledge_left_frontier,
        right_frontier_x=ledge_right_frontier,
    )
    event_phase_index = _valid_event_phase_index(event_phase_index, phase_count=int(phase_count))

    plateau_aug = _robot_corner_com_features(
        positions_seq=plateau_positions,
        robot_point_mask=np.asarray(plateau_run["robot_point_mask"], dtype=bool),
    )
    ledge_aug = _robot_corner_com_features(
        positions_seq=ledge_positions,
        robot_point_mask=np.asarray(ledge_run["robot_point_mask"], dtype=bool),
    )

    plant_pca = fit_pca(plateau_plant, ledge_plant, n_components=3)
    ctrl_pca = fit_pca(plateau_ctrl, ledge_ctrl, n_components=3)
    aug_pca = fit_pca(plateau_aug, ledge_aug, n_components=3)
    plateau_plant_xy = transform_pca(plateau_plant, plant_pca)
    ledge_plant_xy = transform_pca(ledge_plant, plant_pca)
    plateau_ctrl_xy = transform_pca(plateau_ctrl, ctrl_pca)
    ledge_ctrl_xy = transform_pca(ledge_ctrl, ctrl_pca)
    plateau_aug_xy = transform_pca(plateau_aug, aug_pca)
    ledge_aug_xy = transform_pca(ledge_aug, aug_pca)
    env_steps = _phase_env_steps(phase_count, sense_act_phases=int(cfg.sim.sense_act_phases_per_env_step))

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_trace_bundle(
        output_dir / "trace_bundle.npz",
        {
            "plateau_positions": plateau_positions,
            "ledge_positions": ledge_positions,
            "plateau_spring_rest_length": plateau_spring,
            "ledge_spring_rest_length": ledge_spring,
            "plateau_raw_abs_hv": plateau_plant,
            "ledge_raw_abs_hv": ledge_plant,
            "plateau_controller_flat": plateau_ctrl,
            "ledge_controller_flat": ledge_ctrl,
            "plateau_corner_com": plateau_aug,
            "ledge_corner_com": ledge_aug,
            "plateau_front_x_phase": plateau_front[:phase_count],
            "plateau_back_x_phase": plateau_back[:phase_count],
            "ledge_front_x_phase": ledge_front[:phase_count],
            "ledge_back_x_phase": ledge_back[:phase_count],
            "env_steps": env_steps,
            "event_phase_index": np.asarray([-1 if event_phase_index is None else int(event_phase_index)], dtype=np.int32),
        },
    )
    np.savez_compressed(
        output_dir / "state_pca_coords.npz",
        plateau_plant_xy=plateau_plant_xy,
        ledge_plant_xy=ledge_plant_xy,
        plateau_controller_xy=plateau_ctrl_xy,
        ledge_controller_xy=ledge_ctrl_xy,
        plateau_aug_xy=plateau_aug_xy,
        ledge_aug_xy=ledge_aug_xy,
        plant_mean=plant_pca["mean"],
        plant_std=plant_pca["std"],
        plant_components=plant_pca["components"],
        controller_mean=ctrl_pca["mean"],
        controller_std=ctrl_pca["std"],
        controller_components=ctrl_pca["components"],
        aug_mean=aug_pca["mean"],
        aug_std=aug_pca["std"],
        aug_components=aug_pca["components"],
    )

    robot_summary = _render_robot_compare(
        output_dir=output_dir / "robot_compare",
        plateau_run={
            "positions": plateau_positions,
            "spring_rest": plateau_spring,
            "render_info": plateau_run["render_info"],
        },
        ledge_run={
            "positions": ledge_positions,
            "spring_rest": ledge_spring,
            "render_info": ledge_run["render_info"],
        },
        phase_count=int(phase_count),
        sense_act_phases=int(cfg.sim.sense_act_phases_per_env_step),
        event_phase_index=event_phase_index,
        fps=int(fps),
        max_frames=int(max_frames),
        flash_frames=int(flash_frames),
        supersample=int(supersample),
        top_width_cells=int(plateau_width),
        bottom_width_cells=int(ledge_width),
        top_label=f"Plateau {int(plateau_width)}",
        bottom_label=f"Ledge {int(ledge_width)}",
    )
    pca_summary = _render_state_pca(
        output_dir=output_dir / "state_pca",
        plateau_plant_xy=plateau_plant_xy,
        ledge_plant_xy=ledge_plant_xy,
        plateau_ctrl_xy=plateau_ctrl_xy,
        ledge_ctrl_xy=ledge_ctrl_xy,
        plateau_aug_xy=plateau_aug_xy,
        ledge_aug_xy=ledge_aug_xy,
        aug_title="Plant PCA + COM/corners",
        env_steps=env_steps,
        event_phase_index=event_phase_index,
        fps=int(fps),
        max_frames=int(max_frames),
        flash_frames=int(flash_frames),
    )

    summary = {
        "condition_name": str(condition_name),
        "deformation_condition": str(deformation_condition),
        "max_steps": int(cfg.train.max_steps),
        "seed": int(seed),
        "plateau_width": int(plateau_width),
        "ledge_width": int(ledge_width),
        "grid_width": int(cfg.terrain.grid_w),
        "ground_clearance_cells": float(ground_clearance_cells),
        "phase_count": int(phase_count),
        "sense_act_phases": int(cfg.sim.sense_act_phases_per_env_step),
        "nca_steps_per_phase": int(cfg.nca.nca_steps_per_phase),
        "plateau_platform_bounds": [int(plateau_start), int(plateau_end)],
        "ledge_platform_bounds": [int(ledge_start), int(ledge_end)],
        "ledge_event_phase_index": None if event_phase_index is None else int(event_phase_index),
        "ledge_event_env_step": None
        if event_phase_index is None
        else int(env_steps[int(event_phase_index)]),
        "robot_render": robot_summary,
        "state_pca": pca_summary,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config_snapshot_path.resolve(), max_steps=int(args.max_steps))
    params = load_controller_params(args.controller_params_path.resolve())
    output_root = args.output_dir.resolve()
    if output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    summaries = {}
    condition_names = [str(name) for name in args.conditions]
    effective_grid_width = (
        int(args.grid_width)
        if args.grid_width is not None
        else max(
            int(cfg.terrain.grid_w),
            int(max(args.plateau_width, args.ledge_width)) + 5,
        )
    )
    for condition_name in condition_names:
        summaries[condition_name] = _run_condition(
            condition_name=str(condition_name),
            deformation_condition=str(condition_name),
            cfg_template=cfg,
            params=params,
            output_dir=output_root / str(condition_name),
            plateau_width=int(args.plateau_width),
            ledge_width=int(args.ledge_width),
            ground_clearance_cells=float(args.ground_clearance_cells),
            seed=int(args.seed),
            fps=int(args.fps),
            max_frames=int(args.max_frames),
            flash_frames=int(args.flash_frames),
            supersample=int(args.supersample),
            grid_width=int(effective_grid_width),
        )

    write_json(output_root / "summary.json", summaries)
    for condition_name, summary in summaries.items():
        print(f"{condition_name}: {summary['robot_render']['mp4_path']}")
        print(f"{condition_name}: {summary['state_pca']['mp4_path']}")


if __name__ == "__main__":
    main()
