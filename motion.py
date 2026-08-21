"""Deterministic manual motion tracks for the LivePortrait renderer.

There is deliberately no natural-language parser here. The UI/API provide
validated segments with explicit target, axis, angles, start and duration.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

BLINK_PRESETS = {"rest": 17.0, "talking": 23.0, "reading": 6.0, "nervous": 32.0}
# zoom задаётся в процентах: +12 это наезд на 12%, -12 отъезд. Остальные оси - градусы.
AXES = ("yaw", "pitch", "roll", "zoom")
ROTATION_AXES = ("yaw", "pitch", "roll")
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
class GazeSpec:
    """Взгляд как последовательность фиксаций, а не плавная синусоида.

    Саккада длится 20-30 мс, то есть меньше кадра при 30 fps - на видео она
    мгновенная. Живым взгляд делает не сама саккада, а ритм: фиксация 200-400 мс,
    скачок, снова фиксация. Плавный дрейф вместо этого читается как кукольный.
    """
    fixation_ms: float = 280.0      # медиана длительности фиксации
    fixation_sigma: float = 0.45    # логнормальный разброс
    amplitude: float = 4.0          # типичный разброс точек фиксации
    centering: float = 0.55         # тяга обратно к центру (0 - блуждание, 1 - всегда центр)
    drift: float = 0.30             # микродрейф внутри фиксации
    vor_gain: float = 0.95          # контрвращение против движения головы
    vor_decay_ms: float = 260.0     # за сколько глаза догоняют цель после поворота


@dataclass
class BrowSpec:
    """Брови лежат в нейтрали и двигаются событиями, а не висят поднятыми."""
    events_per_min: float = 14.0
    rise_ms: float = 120.0
    hold_ms: float = 160.0
    fall_ms: float = 260.0
    amplitude: float = 3.0
    gaze_coupling: float = 0.25     # брови немного следуют за взглядом по вертикали
    on_gaze_shift: float = 0.45     # доля событий, привязанных к крупным скачкам взгляда


@dataclass
class MotionSpec:
    fps: int = 30
    duration: float | None = 3.0
    segments: list[Segment] = field(default_factory=list)
    blink: BlinkSpec = field(default_factory=BlinkSpec)
    gaze: GazeSpec = field(default_factory=GazeSpec)
    brow: BrowSpec = field(default_factory=BrowSpec)
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


def _fixation_times(spec: MotionSpec, rng: np.random.Generator) -> list[float]:
    """Моменты саккад. Длительность фиксации логнормальная, 120-900 мс."""
    median = spec.gaze.fixation_ms / 1000.0
    times, time = [], 0.0
    while time < spec.total_duration():
        times.append(time)
        step = float(rng.lognormal(np.log(median), spec.gaze.fixation_sigma))
        time += float(np.clip(step, 0.12, 0.9))
    return times


def _vor(spec: MotionSpec, head: np.ndarray) -> np.ndarray:
    """Вестибуло-окулярный рефлекс: глаза контрвращаются во время поворота головы,
    затем догоняют цель. Накопитель с экспоненциальным затуханием - поэтому это
    переходный процесс, а не постоянное смещение при повёрнутой голове."""
    decay = float(np.exp(-1.0 / spec.fps / max(spec.gaze.vor_decay_ms / 1000.0, 1e-3)))
    delta = np.diff(head, prepend=head[:1])
    out = np.zeros(head.shape[0], dtype=np.float64)
    accumulator = 0.0
    for i, step in enumerate(delta):
        accumulator = accumulator * decay - spec.gaze.vor_gain * float(step)
        out[i] = accumulator
    return out


def gaze_tracks(spec: MotionSpec, yaw: np.ndarray, pitch: np.ndarray,
                rng: np.random.Generator, gain: float) -> tuple[np.ndarray, np.ndarray]:
    n = spec.n_frames()
    gain = float(np.clip(gain, 0.0, 1.0))
    if gain <= 0.0:
        zero = np.zeros(n, dtype=np.float32)
        return zero, zero.copy()

    cfg = spec.gaze
    idx = np.arange(n)
    # саккада 25 мс: при 30 fps это один кадр, то есть практически мгновенно
    ramp = max(1.0, 0.025 * spec.fps)

    fix_x = np.zeros(n, dtype=np.float64)
    fix_y = np.zeros(n, dtype=np.float64)
    point_x = point_y = 0.0
    for start in _fixation_times(spec, rng):
        # затухающее блуждание: точка уходит от центра, но её тянет обратно
        next_x = (1.0 - cfg.centering) * point_x + rng.normal(0.0, cfg.amplitude)
        next_y = (1.0 - cfg.centering) * point_y + rng.normal(0.0, cfg.amplitude * 0.6)
        step = _min_jerk((idx - start * spec.fps) / ramp)
        fix_x += (next_x - point_x) * step
        fix_y += (next_y - point_y) * step
        point_x, point_y = next_x, next_y

    # микродрейф внутри фиксации - глаз не стоит идеально
    drift_x = micro_motion(spec, rng, cfg.drift / max(spec.micro_amp_deg, 1e-6))
    drift_y = micro_motion(spec, rng, cfg.drift / max(spec.micro_amp_deg, 1e-6) * 0.7)

    gaze_x = gain * (fix_x + drift_x + _vor(spec, yaw))
    gaze_y = gain * (fix_y + drift_y + _vor(spec, pitch) * 0.7)
    return gaze_x.astype(np.float32), gaze_y.astype(np.float32)


def _brow_kernel(spec: MotionSpec) -> np.ndarray:
    cfg = spec.brow
    rise = max(1, round(cfg.rise_ms / 1000 * spec.fps))
    hold = max(0, round(cfg.hold_ms / 1000 * spec.fps))
    fall = max(1, round(cfg.fall_ms / 1000 * spec.fps))
    up = _min_jerk(np.linspace(0, 1, rise + 1)[1:])
    down = 1.0 - _min_jerk(np.linspace(0, 1, fall + 1)[1:])
    return np.concatenate([up, np.ones(hold), down])


def brow_track(spec: MotionSpec, gaze_y: np.ndarray, rng: np.random.Generator,
               gain: float) -> np.ndarray:
    """Нейтраль плюс редкие события. Постоянно поднятые брови читаются как удивление."""
    n = spec.n_frames()
    gain = float(np.clip(gain, 0.0, 1.0))
    if gain <= 0.0 or spec.brow.events_per_min <= 0.0:
        return np.zeros(n, dtype=np.float32)

    cfg = spec.brow
    kernel = _brow_kernel(spec)
    track = np.zeros(n, dtype=np.float64)
    duration = spec.total_duration()

    mean_gap = 60.0 / cfg.events_per_min
    # Первое событие ставим рано: при среднем интервале в несколько секунд
    # на трёхсекундном клипе иначе не выпадает ни одного, и ползунок молчит.
    events = []
    time = float(rng.uniform(0.25, max(0.4, min(mean_gap, 0.45 * duration))))
    while time < duration:
        events.append(time)
        time += float(rng.exponential(mean_gap))

    # Часть событий привязана к крупным скачкам взгляда - живое лицо делает так же.
    if cfg.on_gaze_shift > 0 and gaze_y.size > 2:
        speed = np.abs(np.diff(gaze_y, prepend=gaze_y[:1]))
        threshold = max(float(np.quantile(speed, 0.97)), 1e-6)
        for index in np.flatnonzero(speed >= threshold):
            if rng.random() < cfg.on_gaze_shift:
                events.append(float(index) / spec.fps)

    for event in events:
        start = int(round(event * spec.fps))
        piece = kernel[: max(0, n - start)] * float(rng.uniform(0.55, 1.0))
        if piece.size:
            track[start:start + piece.size] = np.maximum(track[start:start + piece.size], piece)

    # брови немного следуют за вертикалью взгляда - это настоящая связка
    return (gain * (cfg.amplitude * track + cfg.gaze_coupling * gaze_y)).astype(np.float32)


def zoom_track(spec: MotionSpec, slider: float) -> np.ndarray:
    """Сегменты с осью zoom (в процентах) плюс общий ползунок. Отъезд разрешён."""
    n = spec.n_frames()
    from_segments = render_axis(spec, "zoom") / 100.0
    slider = float(np.clip(slider, -1.0, 1.0))
    ramp = _min_jerk(np.arange(n) / max(spec.total_duration() * spec.fps, 1e-6))
    return (1.0 + from_segments + slider * 0.08 * ramp).astype(np.float32)


def artistic_tracks(spec: MotionSpec, yaw: np.ndarray, pitch: np.ndarray,
                    rng: np.random.Generator, *, zoom: float = 0.0,
                    gaze: float = 0.0, brow: float = 0.0) -> tuple[np.ndarray, ...]:
    gaze_x, gaze_y = gaze_tracks(spec, yaw, pitch, rng, gaze)
    return (zoom_track(spec, zoom), gaze_x, gaze_y, brow_track(spec, gaze_y, rng, brow))


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
