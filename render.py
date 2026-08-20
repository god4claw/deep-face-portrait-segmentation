"""Direct LivePortrait motion-track renderer with source caching."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import imageio_ffmpeg
import numpy as np
from PIL import Image

from motion import generate, to_json

Progress = Callable[[float, str], None]


@dataclass
class SourceState:
    f_s: object
    x_s: object
    x_s_info: dict
    r_s: object
    source_lmk: np.ndarray
    m_c2o: np.ndarray
    mask_ori: np.ndarray
    image_rgb: np.ndarray


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()


def encode_png_sequence(frame_dir: Path, fps: int, output: Path, *, prefer_nvenc: bool = True) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    base = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(frame_dir / "%06d.png"), "-pix_fmt", "yuv420p"]
    attempts = []
    if prefer_nvenc:
        attempts.append(("h264_nvenc", base + ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "20", str(output)]))
    attempts.append(("libx264", base + ["-c:v", "libx264", "-crf", "18", "-preset", "medium", str(output)]))
    errors = []
    for encoder, command in attempts:
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0 and output.exists() and output.stat().st_size > 0:
            return encoder
        errors.append(f"{encoder}: exit={result.returncode} {result.stderr[-800:]}")
    raise RuntimeError("video encoding failed\n" + "\n".join(errors))


class PortraitRenderer:
    def __init__(self, liveportrait_root: Path, *, cache_size: int = 4):
        self.root = liveportrait_root.resolve()
        if not (self.root / "src" / "live_portrait_wrapper.py").exists():
            raise FileNotFoundError(f"invalid LivePortrait checkout: {self.root}")
        sys.path.insert(0, str(self.root)) if str(self.root) not in sys.path else None
        from src.config.crop_config import CropConfig
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.cropper import Cropper

        self.inference_cfg = InferenceConfig(flag_use_half_precision=True, flag_stitching=True)
        self.crop_cfg = CropConfig()
        self.wrapper = LivePortraitWrapper(self.inference_cfg)
        self.cropper = Cropper(crop_cfg=self.crop_cfg)
        self.cache_size = cache_size
        self.cache: OrderedDict[str, SourceState] = OrderedDict()
        self.lock = threading.RLock()

    @staticmethod
    def _load_rgb(path: Path, max_h: int) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        if max_h and image.height > max_h:
            width = max(2, round(image.width * max_h / image.height))
            width -= width % 2
            image = image.resize((width, max_h), Image.Resampling.LANCZOS)
        return np.asarray(image)

    def prepare_source(self, path: Path, *, max_h: int) -> SourceState:
        from src.utils.camera import get_rotation_matrix
        from src.utils.crop import prepare_paste_back

        key = hashlib.sha256(path.read_bytes() + str(max_h).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            image_rgb = self._load_rgb(path, max_h)
            crop = self.cropper.crop_source_image(image_rgb, self.crop_cfg)
            if crop is None:
                raise ValueError("no face detected in source image")
            source = self.wrapper.prepare_source(crop["img_crop_256x256"])
            info = self.wrapper.get_kp_info(source)
            state = SourceState(
                f_s=self.wrapper.extract_feature_3d(source),
                x_s=self.wrapper.transform_keypoint(info),
                x_s_info=info,
                r_s=get_rotation_matrix(info["pitch"], info["yaw"], info["roll"]),
                source_lmk=crop["lmk_crop"],
                m_c2o=crop["M_c2o"],
                mask_ori=prepare_paste_back(self.inference_cfg.mask_crop, crop["M_c2o"],
                                            dsize=(image_rgb.shape[1], image_rgb.shape[0])),
                image_rgb=image_rgb,
            )
            self.cache[key] = state
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            return state

    @staticmethod
    def _apply_gaze(delta, x: float, y: float):
        delta = delta.clone()
        delta[0, 11, 0] += x * (0.0007 if x > 0 else 0.001)
        delta[0, 15, 0] += x * (0.001 if x > 0 else 0.0007)
        delta[0, 11, 1] -= y * 0.001
        delta[0, 15, 1] -= y * 0.001
        return delta

    @staticmethod
    def _apply_brow(delta, value: float):
        delta = delta.clone()
        if value >= 0:
            delta[0, 1, 1] += value * 0.001
            delta[0, 2, 1] -= value * 0.001
        else:
            delta[0, 1, 0] -= value * 0.001
            delta[0, 2, 0] += value * 0.001
        return delta

    def render(self, track: dict, source_path: Path, output: Path, *, preview: bool,
               prefer_nvenc: bool = True, progress: Progress | None = None) -> dict:
        import torch
        from src.utils.camera import get_rotation_matrix
        from src.utils.crop import paste_back

        callback = progress or (lambda value, stage: None)
        max_h = 512 if preview else 720
        callback(0.02, "preparing source")
        with self.lock, torch.no_grad():
            state = self.prepare_source(source_path, max_h=max_h)
            n_frames = int(track["n_frames"])
            fps = int(track["fps"])
            with tempfile.TemporaryDirectory(prefix="deep-randy-face-") as tmp:
                frame_dir = Path(tmp)
                for i in range(n_frames):
                    info = state.x_s_info
                    r_d = get_rotation_matrix(
                        info["pitch"] + float(track["pitch"][i]),
                        info["yaw"] + float(track["yaw"][i]),
                        info["roll"] + float(track["roll"][i]),
                    )
                    r_new = (r_d @ state.r_s.permute(0, 2, 1)) @ state.r_s
                    delta = self._apply_gaze(info["exp"], float(track["gaze_x"][i]), float(track["gaze_y"][i]))
                    delta = self._apply_brow(delta, float(track["brow"][i]))
                    x_d = float(track["zoom"][i]) * info["scale"] * (info["kp"] @ r_new + delta) + info["t"]
                    eye_target = float(track["eye_open"][i])
                    if eye_target < 0.999:
                        from src.utils.retargeting_utils import calc_eye_close_ratio
                        source_ratio = float(calc_eye_close_ratio(state.source_lmk[None]).mean())
                        combined = self.wrapper.calc_combined_eye_ratio([[source_ratio * eye_target]], state.source_lmk)
                        x_d = x_d + self.wrapper.retarget_eye(state.x_s, combined)
                    x_d = self.wrapper.stitching(state.x_s, x_d)
                    crop_frame = self.wrapper.parse_output(self.wrapper.warp_decode(state.f_s, state.x_s, x_d)["out"])[0]
                    frame = crop_frame if preview else paste_back(crop_frame, state.m_c2o, state.image_rgb, state.mask_ori)
                    Image.fromarray(frame).save(frame_dir / f"{i:06d}.png", compress_level=1)
                    callback(0.08 + 0.80 * (i + 1) / n_frames, f"rendering {i + 1}/{n_frames}")
                callback(0.90, "encoding video")
                encoder = encode_png_sequence(frame_dir, fps, output, prefer_nvenc=prefer_nvenc)
        callback(1.0, "complete")
        return {"encoder": encoder, "frames": n_frames, "fps": fps, "cache_entries": len(self.cache)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render LivePortrait motion tracks to MP4")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--prompt", default="поворот направо на 20 градусов")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--liveportrait-root", type=Path,
                        default=Path(os.environ.get("LIVEPORTRAIT_ROOT", r"C:\Users\windows11\crazyrandypipe\LivePortrait")))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--no-nvenc", action="store_true")
    parser.add_argument("--zoom", type=float, default=0.0)
    parser.add_argument("--gaze", type=float, default=0.0)
    parser.add_argument("--brow", type=float, default=0.0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    track = generate(args.prompt, zoom=args.zoom, gaze=args.gaze, brow=args.brow)
    to_json(track, str(args.out.with_suffix(".motion.json")))
    renderer = PortraitRenderer(args.liveportrait_root)
    meta = renderer.render(track, args.src, args.out, preview=args.preview, prefer_nvenc=not args.no_nvenc,
                           progress=lambda p, s: print(f"{p:6.1%} {s}", flush=True))
    print(json.dumps({"output": str(args.out), **meta}))


if __name__ == "__main__":
    main()
