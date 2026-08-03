from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from free_space_branch_common import (
    DEFAULT_BUNDLE_ROOT,
    default_analysis_dir,
    encode_frames,
    load_branch_trace,
    sample_phase_indices,
    write_json,
)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


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


def sorted_pngs(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("frame_*.png"))


def crop_pca_panels(frame: Image.Image) -> tuple[Image.Image, Image.Image]:
    width, height = frame.size
    split_y = height // 2
    top = frame.crop((0, 0, width, split_y))
    bottom = frame.crop((0, split_y, width, height))
    return top, bottom


def resize_to_height(image: Image.Image, *, target_height: int) -> Image.Image:
    scale = float(target_height) / float(image.height)
    target_width = max(1, int(round(image.width * scale)))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def compose_frame(
    *,
    plant_panel: Image.Image,
    robot_panel: Image.Image,
    controller_panel: Image.Image,
    status_text: str,
    flash: bool,
    event_label: str,
) -> Image.Image:
    margin_x = 28
    margin_y = 20
    gutter = 24
    header_h = 56
    footer_h = 16
    panel_h = robot_panel.height

    plant_panel = resize_to_height(plant_panel, target_height=panel_h)
    controller_panel = resize_to_height(controller_panel, target_height=panel_h)

    canvas_w = (
        (2 * margin_x)
        + plant_panel.width
        + robot_panel.width
        + controller_panel.width
        + (2 * gutter)
    )
    canvas_h = margin_y + header_h + panel_h + footer_h + margin_y
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (251, 250, 247, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(size=28, bold=True)
    meta_font = load_font(size=18, bold=False)
    flash_font = load_font(size=34, bold=True)

    x_plant = margin_x
    x_robot = x_plant + plant_panel.width + gutter
    x_ctrl = x_robot + robot_panel.width + gutter
    y_panel = margin_y + header_h

    draw.text((margin_x, 18), "Plant / Robot / Controller", fill=(28, 25, 23), font=title_font)
    draw.text((canvas_w - margin_x, 24), status_text, fill=(87, 83, 78), font=meta_font, anchor="ra")

    canvas.alpha_composite(plant_panel.convert("RGBA"), dest=(x_plant, y_panel))
    canvas.alpha_composite(robot_panel.convert("RGBA"), dest=(x_robot, y_panel))
    canvas.alpha_composite(controller_panel.convert("RGBA"), dest=(x_ctrl, y_panel))

    outline = (214, 211, 209, 255)
    draw.rectangle(
        [x_plant, y_panel, x_plant + plant_panel.width - 1, y_panel + plant_panel.height - 1],
        outline=outline,
        width=2,
    )
    draw.rectangle(
        [x_robot, y_panel, x_robot + robot_panel.width - 1, y_panel + robot_panel.height - 1],
        outline=outline,
        width=2,
    )
    draw.rectangle(
        [x_ctrl, y_panel, x_ctrl + controller_panel.width - 1, y_panel + controller_panel.height - 1],
        outline=outline,
        width=2,
    )

    if flash:
        overlay = Image.new("RGBA", canvas.size, (255, 243, 182, 58))
        canvas = Image.alpha_composite(canvas, overlay)
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (canvas.width // 2, 20),
            event_label,
            fill=(159, 18, 57),
            font=flash_font,
            anchor="ma",
        )

    return canvas.convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose synchronized plant PCA, robot render, and controller PCA into one video.",
    )
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--robot-dir", type=Path, default=None)
    parser.add_argument("--pca-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--flash-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = (
        args.trace_dir.resolve()
        if args.trace_dir is not None
        else default_analysis_dir(args.bundle_root.resolve(), max_steps=int(args.max_steps)).resolve()
    )
    trace = load_branch_trace(trace_dir)
    summary = load_json(trace_dir / "summary.json")
    robot_dir = (
        args.robot_dir.resolve()
        if args.robot_dir is not None
        else (trace_dir / "robot_renders_body_frame" / "tight").resolve()
    )
    pca_dir = (
        args.pca_dir.resolve()
        if args.pca_dir is not None
        else (trace_dir / "state_pca_hold_compare").resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (trace_dir / "composite").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    robot_frames = sorted_pngs(robot_dir / "frames")
    pca_frames = sorted_pngs(pca_dir / "frames")
    if not robot_frames or not pca_frames:
        raise FileNotFoundError("Missing robot or PCA frames for composition.")
    if len(robot_frames) != len(pca_frames):
        raise ValueError(
            f"Frame count mismatch: robot={len(robot_frames)} pca={len(pca_frames)}."
        )

    phase_count = int(trace["baseline_positions"].shape[0])
    perturb_phase_index = int(trace["perturb_phase_index"][0])
    sense_act_phases = int(trace["sense_act_phases"][0])
    frame_phase_indices = sample_phase_indices(
        phase_count,
        max_frames=len(robot_frames),
        required_phase_index=perturb_phase_index,
    )
    perturb_frame_idx = int(np.searchsorted(frame_phase_indices, perturb_phase_index))
    event_label = str(summary.get("event_label", "PERTURBATION"))

    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, (robot_path, pca_path, phase_idx) in enumerate(
        zip(robot_frames, pca_frames, frame_phase_indices)
    ):
        robot_panel = Image.open(robot_path).convert("RGB")
        pca_frame = Image.open(pca_path).convert("RGB")
        plant_panel, controller_panel = crop_pca_panels(pca_frame)
        env_step = int(phase_idx) // max(1, sense_act_phases)
        status_text = f"Phase {int(phase_idx) + 1} / {phase_count}   Env step {env_step + 1}"
        frame = compose_frame(
            plant_panel=plant_panel,
            robot_panel=robot_panel,
            controller_panel=controller_panel,
            status_text=status_text,
            flash=bool(perturb_frame_idx <= frame_idx < (perturb_frame_idx + int(args.flash_frames))),
            event_label=event_label,
        )
        frame.save(frames_dir / f"frame_{frame_idx:04d}.png")

    mp4_path = output_dir / "composite.mp4"
    gif_path = output_dir / "composite.gif"
    encode_frames(frames_dir, fps=int(args.fps), mp4_path=mp4_path, gif_path=gif_path)
    write_json(
        output_dir / "summary.json",
        {
            "trace_dir": str(trace_dir),
            "robot_dir": str(robot_dir),
            "pca_dir": str(pca_dir),
            "mp4_path": str(mp4_path),
            "gif_path": str(gif_path),
            "frame_count": len(robot_frames),
            "flash_frames": int(args.flash_frames),
        },
    )
    print(f"mp4_path: {mp4_path}")
    print(f"gif_path: {gif_path}")


if __name__ == "__main__":
    main()
