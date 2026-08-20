from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion import MotionSpec, generate
from render import PortraitRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the real eye-retargeting path")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--liveportrait-root", type=Path, required=True)
    args = parser.parse_args()
    track = generate("поворот направо на 8 градусов", MotionSpec(fps=15, duration=1.0))
    track["eye_open"][:] = 1.0
    track["eye_open"][5:10] = np.array([0.65, 0.2, 0.0, 0.3, 0.75], dtype=np.float32)
    renderer = PortraitRenderer(args.liveportrait_root)
    renderer.render(track, args.source, args.output, preview=True, prefer_nvenc=True,
                    progress=lambda value, stage: print(f"{value:.4f} {stage}", flush=True))


if __name__ == "__main__":
    main()
