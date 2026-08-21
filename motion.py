"""Deterministic manual motion tracks for the LivePortrait renderer.

There is deliberately no natural-language parser here. The UI/API provide
validated segments with explicit target, axis, angles, start and duration.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

BLINK_PRESETS = {"rest": 17.0, "talking": 23.0, "reading": 6.0, "nervous": 32.0}
AXES = ("yaw", "pitch", "roll")
TARGETS = ("face", "neck", "torso", "clothes")
DEFAULT_TAIL_SEC = 0.70


@dataclass
class Segment:
    target: str = "face"
    axis: str = "yaw"
    frm: float = 0.0
    to: float = 0.0
    start: float = 0.0
    duration: float = 0.55

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"unsupported target: {self.target}")
        if self.axis not in AXES:
            raise ValueError(f"unsupported face axis: {self.axis}")
        if not np.isfinite([self.frm, self.to, self.start, self.duration]).all():
            raise ValueError("segment values must be finite")
        if self.start < 0 or self.duration <= 0:
            raise ValueError("segment start must be >= 0 and duration > 0")
        if abs(self.frm) > 180 or abs(self.to) > 180:
            raise ValueError("angles must be between -180 and 180 degrees")

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class BlinkSpec:
    rate_per_min: float = BLINK_PRESETS["rest"]
    sigma: float = 0.55
    refractory: float = 0.6
    close_ms: float = 90.0
    hold_ms: float = 30.0
    open_ms: float = 210.0
    double_prob: float = 0.15


@dataclass
class MotionSpec:
    fps: int = 30
    duration: float | None = 3.0
    segments: list[Segment] = field(default_factory=list)
    blink: BlinkSpec = field(default_factory=BlinkSpec)
    micro_amp_deg: float = 1.4
    micro_freq_hz: tuple = (0.13, 0.29, 0.47)
    seed: int | None = 42

    def total_duration(self) -> float:
        if self.duration is not None:
            return max(0.1, float(self.duration))
        return max((s.end for s in self.segments), default=3.0) + DEFAULT_TAIL_SEC

    def n_frames(self) -> int:
        return max(1, int(round(self.total_duration() * self.fps)))


def segments_from_data(value: object) -> list[Segment]:
    if not isinstance(value, list) or len(value) > 64:
        raise ValueError("segments must be a list with at most 64 segments")
    segments = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each segment must be an object")
        segments.append(Segment(target=item.get("target", "face"), axis=item["axis"],
                                frm=float(item["frm"]), to=float(item["to"]),
                                start=float(item["start"]), duration=float(item["duration"])))
    return segments


def _min_jerk(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def render_axis(spec: MotionSpec, axis: str) -> np.ndarray:
    """Render only face-target segments for one explicit axis."""
    n = spec.n_frames()
    idx = np.arange(n)
    out = np.zeros(n, dtype=np.float64)
    for segment in sorted((s for s in spec.segments if s.target == "face" and s.axis == axis),
                          key=lambda item: item.start):
        start = segment.start * spec.fps
        span = max(1e-6, segment.duration * spec.fps)
        out += (segment.to - segment.frm) * _min_jerk((idx - start) / span)
    return out.astype(np.float32)


def render_target_axis(spec: MotionSpec, target: str, axis: str) -> np.ndarray:
    """Render a non-face target track for the full-frame body stage."""
    n = spec.n_frames()
    idx = np.arange(n)
    out = np.zeros(n, dtype=np.float64)
    for segment in sorted((s for s in spec.segments if s.target == target and s.axis == axis),
                          key=lambda item: item.start):
        start = segment.start * spec.fps
        span = max(1e-6, segment.duration * spec.fps)
        out += (segment.to - segment.frm) * _min_jerk((idx - start) / span)
    return out.astype(np.float32)


def micro_motion(spec: MotionSpec, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    n = spec.n_frames()
    t = np.arange(n) / spec.fps
    signal = sum(np.sin(2 * np.pi * frequency * t + rng.uniform(0, 2 * np.pi))
                 for frequency in spec.micro_freq_hz) / len(spec.micro_freq_hz)
    return (signal * spec.micro_amp_deg * scale).astype(np.float32)


def blink_kernel(fps: int, blink: BlinkSpec) -> np.ndarray:
    close = max(1, round(blink.close_ms / 1000 * fps))
    hold = max(0, round(blink.hold_ms / 1000 * fps))
    opening = max(1, round(blink.open_ms / 1000 * fps))
    closing_t = np.linspace(0, 1, close + 1)[1:]
    opening_t = np.linspace(0, 1, opening + 1)[1:]
    closing = 1.0 - closing_t**1.6
    opening_curve = 1.0 - (1.0 - opening_t)**2.1
    return np.clip(np.concatenate([closing, np.zeros(hold), opening_curve]), 0.0, 1.0)


def blink_track(spec: MotionSpec, rng: np.random.Generator) -> tuple[np.ndarray, list[float]]:
    n = spec.n_frames()
    track = np.ones(n, dtype=np.float32)
    blink = spec.blink
    if blink.rate_per_min <= 0.01:
        return track, []
    kernel = blink_kernel(spec.fps, blink)
    mean_interval = 60.0 / blink.rate_per_min * (1.0 + blink.double_prob)
    median = mean_interval / np.exp(blink.sigma**2 / 2)
    times: list[float] = []
    time = float(rng.uniform(0.15, max(0.3, min(mean_interval, spec.total_duration()))))
    while time < spec.total_duration():
        times.append(time)
        if rng.random() < blink.double_prob:
            second = time + float(rng.uniform(0.18, 0.42))
            if second < spec.total_duration():
                times.append(second)
                time = second
        time += max(float(rng.lognormal(np.log(median), blink.sigma)), blink.refractory)
    for blink_time in times:
        start = int(round(blink_time * spec.fps))
        segment = kernel[: max(0, n - start)]
        if segment.size:
            track[start:start + segment.size] = np.minimum(track[start:start + segment.size], segment)
    return track, times


def artistic_tracks(spec: MotionSpec, yaw: np.ndarray, pitch: np.ndarray,
                    rng: np.random.Generator, *, zoom: float = 0.0,
                    gaze: float = 0.0, brow: float = 0.0) -> tuple[np.ndarray, ...]:
    n = spec.n_frames()
    t = np.arange(n, dtype=np.float32) / spec.fps
    zoom_track = 1.0 + np.clip(zoom, 0.0, 1.0) * 0.08 * _min_jerk(t / max(spec.total_duration(), 1e-6))
    phase_x, phase_y = rng.uniform(0, 2 * np.pi, size=2)
    gaze_gain = np.clip(gaze, 0.0, 1.0)
    gaze_x = gaze_gain * (-0.18 * yaw + 3.0 * np.sin(2 * np.pi * 0.23 * t + phase_x))
    gaze_y = gaze_gain * (-0.12 * pitch + 2.0 * np.sin(2 * np.pi * 0.17 * t + phase_y))
    brow_track = np.clip(brow, 0.0, 1.0) * (2.5 - 0.25 * gaze_y + 0.8 * np.sin(2 * np.pi * 0.13 * t))
    return tuple(np.asarray(value, dtype=np.float32) for value in (zoom_track, gaze_x, gaze_y, brow_track))


def generate(spec: MotionSpec | None = None, *, zoom: float = 0.0,
             gaze: float = 0.0, brow: float = 0.0) -> dict:
    """Generate numeric tracks from an explicit MotionSpec only."""
    spec = spec or MotionSpec()
    rng = np.random.default_rng(spec.seed)
    eye_open, blink_times = blink_track(spec, rng)
    yaw = render_axis(spec, "yaw") + micro_motion(spec, rng, 1.0)
    pitch = render_axis(spec, "pitch") + micro_motion(spec, rng, 0.6)
    roll = render_axis(spec, "roll") + micro_motion(spec, rng, 0.4)
    body_yaw = sum((render_target_axis(spec, target, "yaw") for target in ("neck", "torso", "clothes")),
                   start=np.zeros(spec.n_frames(), dtype=np.float32))
    body_pitch = sum((render_target_axis(spec, target, "pitch") for target in ("neck", "torso", "clothes")),
                     start=np.zeros(spec.n_frames(), dtype=np.float32))
    body_roll = sum((render_target_axis(spec, target, "roll") for target in ("neck", "torso", "clothes")),
                    start=np.zeros(spec.n_frames(), dtype=np.float32))
    zoom_track, gaze_x, gaze_y, brow_track = artistic_tracks(
        spec, yaw, pitch, rng, zoom=zoom, gaze=gaze, brow=brow)
    return {
        "fps": spec.fps, "duration": spec.total_duration(), "n_frames": spec.n_frames(),
        "yaw": yaw, "pitch": pitch, "roll": roll, "zoom": zoom_track,
        "body_yaw": body_yaw, "body_pitch": body_pitch, "body_roll": body_roll,
        "gaze_x": gaze_x, "gaze_y": gaze_y, "brow": brow_track,
        "eye_open": eye_open, "blink_times": blink_times,
        "segments": [asdict(segment) for segment in spec.segments],
        "blink": asdict(spec.blink),
    }


def to_json(track: dict, path: str) -> None:
    output = {key: value.tolist() if isinstance(value, np.ndarray) else value
              for key, value in track.items()}
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    spec = MotionSpec(segments=[Segment(axis="yaw", frm=0, to=20, start=0.6, duration=0.55)])
    track = generate(spec)
    print(f"manual segments: {len(spec.segments)}, frames: {track['n_frames']}, blinks: {len(track['blink_times'])}")
