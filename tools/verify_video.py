from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed semantic MP4 verification")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("system ffmpeg and ffprobe are required")
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(args.video)],
        text=True, capture_output=True,
    )
    if probe.returncode:
        raise SystemExit(f"ffprobe failed: {probe.stderr}")
    stream = json.loads(probe.stdout)["streams"][0]
    frames = int(stream["nb_read_frames"])
    if frames != args.expected_frames:
        raise SystemExit(f"expected {args.expected_frames} frames, got {frames}")
    decode = subprocess.run([ffmpeg, "-v", "error", "-i", str(args.video), "-f", "null", "-"],
                            text=True, capture_output=True)
    if decode.returncode:
        raise SystemExit(f"full decode failed: {decode.stderr}")
    selected = [0, frames // 2, frames - 1]
    with tempfile.TemporaryDirectory(prefix="portrait-verify-") as tmp:
        paths = []
        for index in selected:
            path = Path(tmp) / f"frame-{index}.png"
            result = subprocess.run([ffmpeg, "-v", "error", "-i", str(args.video), "-vf", f"select=eq(n\\,{index})",
                                     "-frames:v", "1", str(path)], text=True, capture_output=True)
            if result.returncode or not path.exists():
                raise SystemExit(f"frame extraction failed at {index}: {result.stderr}")
            paths.append(path)
        arrays = [np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) for path in paths]
        source = Image.open(args.source).convert("RGB").resize((arrays[0].shape[1], arrays[0].shape[0]))
        source_array = np.asarray(source, dtype=np.float32)
        first_source_mae = float(np.abs(arrays[0] - source_array).mean())
        motion_mae = float(np.abs(arrays[1] - arrays[0]).mean())
        border = max(8, arrays[0].shape[0] // 12)
        border_mask = np.zeros(arrays[0].shape[:2], dtype=bool)
        border_mask[:border] = border_mask[-border:] = True
        border_mask[:, :border] = border_mask[:, -border:] = True
        border_mae = float(np.abs(arrays[0] - source_array)[border_mask].mean())
        if motion_mae <= 0.25:
            raise SystemExit(f"motion is not measurable: MAE={motion_mae:.4f}")
        if first_source_mae >= 35:
            raise SystemExit(f"first frame identity drift is excessive: MAE={first_source_mae:.4f}")
        if border_mae >= 8:
            raise SystemExit(f"pasteback changed the outer border excessively: MAE={border_mae:.4f}")
        for index, path in zip(selected, paths):
            shutil.copyfile(path, args.evidence_dir / f"frame-{index}.png")
    report = {
        "video": str(args.video.resolve()),
        "sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream["r_frame_rate"],
        "frames": frames,
        "full_decode": "ok",
        "first_source_mae": first_source_mae,
        "motion_mae": motion_mae,
        "border_pasteback_mae": border_mae,
        "verdict": "pass",
    }
    (args.evidence_dir / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
