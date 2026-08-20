"""Safe local portrait-motion renderer.

The preview path creates a normal MP4 from a still image and motion tracks.
The optional LivePortrait adapter delegates to an existing checkout and never
downloads weights or injects a virtual camera.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageEnhance

from motion import generate, to_json


def _load_track(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: np.asarray(value) if isinstance(value, list) else value for key, value in data.items()}


def _frame(image: Image.Image, yaw: float, pitch: float, zoom: float) -> Image.Image:
    """Render a restrained 2D preview without fabricating a camera stream."""
    w, h = image.size
    scale = max(1.0, float(zoom))
    crop_w = max(2, int(w / scale))
    crop_h = max(2, int(h / scale))
    cx = int(w / 2 + yaw * w / 90)
    cy = int(h / 2 + pitch * h / 90)
    left = min(max(0, cx - crop_w // 2), w - crop_w)
    top = min(max(0, cy - crop_h // 2), h - crop_h)
    out = image.crop((left, top, left + crop_w, top + crop_h)).resize((w, h), Image.Resampling.LANCZOS)
    return out.rotate(float(yaw) * 0.08, resample=Image.Resampling.BICUBIC, expand=False)


def render_preview(source: Path, track: dict, output: Path) -> Path:
    image = Image.open(source).convert("RGB")
    fps = int(track.get("fps", 30))
    yaw = np.asarray(track.get("yaw", np.zeros(track["n_frames"])), dtype=float)
    pitch = np.asarray(track.get("pitch", np.zeros(track["n_frames"])), dtype=float)
    zoom = np.asarray(track.get("zoom", np.ones(track["n_frames"])), dtype=float)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.Popen(
        [ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{image.width}x{image.height}", "-r", str(fps), "-i", "-",
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for i in range(int(track["n_frames"])):
            process.stdin.write(np.asarray(_frame(image, yaw[i], pitch[i], zoom[i]), dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg preview render failed ({code}): {stderr[-1000:]}")
    return output


def render_liveportrait(source: Path, driving: Path, output: Path, liveportrait_root: Path) -> Path:
    """Delegate to an existing LivePortrait checkout using its public CLI."""
    cli = liveportrait_root / "inference.py"
    if not cli.exists():
        raise FileNotFoundError(f"LivePortrait CLI not found: {cli}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["python", str(cli), "--source", str(source), "--driving", str(driving), "--output_dir", str(output.parent)]
    completed = subprocess.run(command, cwd=liveportrait_root, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"LivePortrait failed: {completed.stderr[-2000:]}")
    candidates = sorted(output.parent.glob(f"{source.stem}--{driving.stem}*.mp4"))
    if not candidates:
        raise RuntimeError("LivePortrait completed but no MP4 was produced")
    shutil.copyfile(candidates[-1], output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a local portrait animation to MP4")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--prompt", default="поворот направо на 20 градусов")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--track", type=Path)
    parser.add_argument("--driving", type=Path, help="Optional driving video for LivePortrait")
    parser.add_argument("--liveportrait-root", type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    track = _load_track(args.track) if args.track else generate(args.prompt)
    if args.track is None:
        track_path = args.out.with_suffix(".motion.json")
        to_json(track, str(track_path))
    if args.driving:
        if not args.liveportrait_root:
            parser.error("--liveportrait-root is required with --driving")
        render_liveportrait(args.src, args.driving, args.out, args.liveportrait_root)
    else:
        render_preview(args.src, track, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
