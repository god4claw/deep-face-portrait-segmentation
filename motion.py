"""
Генератор управляющей траектории для LivePortrait.

Промт -> список фаз -> покадровые yaw/pitch/roll (градусы) и eye_open (0..1).
Зависимости: только numpy. Работает независимо от torch.

Разбор промта устроен так: текст режется на клаузы по разделителям
("потом", "затем", ",", " и "), и каждая клауза разбирается автономно.
Число ищется ВНУТРИ своей клаузы, поэтому "наклон 10 вниз и поворот 40 вправо"
даёт pitch=-10 и yaw=+40, а не 10 на обе оси. Отрицание ("не направо")
гасит направление в пределах своей клаузы.

Это по-прежнему не синтаксический разбор по грамматике: клауза плоская,
вложенности нет. Но привязка числа к референту, композиция и отрицание —
настоящие, а не побочный эффект порядка регулярок.

Физиология моргания зашита по литературным значениям:
  частота покоя        15-20 / мин (разговор 20-26, чтение с экрана 4-8)
  межморгательный инт. логнормальное распределение, медиана ~3.5-4 с
  рефрактерный период  ~0.6 с
  полный цикл          250-400 мс
  асимметрия           закрытие ~90 мс, открытие ~210 мс
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

import numpy as np

BLINK_PRESETS = {
    "rest": 17.0,      # спокойное бодрствование
    "talking": 23.0,   # разговор
    "reading": 6.0,    # чтение / фокус на экране
    "nervous": 32.0,   # возбуждение, стресс
}

AXES = ("yaw", "pitch", "roll")

DEFAULT_TURN_SEC = 0.55     # комфортный поворот на 30 град
DEFAULT_GAP_SEC = 0.40      # пауза между командами
DEFAULT_LEAD_SEC = 0.60     # затакт перед первым движением
DEFAULT_TAIL_SEC = 0.70     # хвост после последнего


# ============================================================ модель данных

@dataclass
class Segment:
    """Один переход по одной оси. Абсолютные углы, не приращения."""
    axis: str
    frm: float
    to: float
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class BlinkSpec:
    rate_per_min: float = BLINK_PRESETS["rest"]
    sigma: float = 0.55            # разброс логнормали по интервалам
    refractory: float = 0.6        # с, минимум между морганиями
    close_ms: float = 90.0
    hold_ms: float = 30.0
    open_ms: float = 210.0
    double_prob: float = 0.15


@dataclass
class MotionSpec:
    """Полное описание клипа. Это то, что раньше было плоским MotionConfig."""
    fps: int = 30
    duration: float | None = None          # None -> вывести из фаз
    segments: list = field(default_factory=list)
    blink: BlinkSpec = field(default_factory=BlinkSpec)
    micro_amp_deg: float = 1.4
    micro_freq_hz: tuple = (0.13, 0.29, 0.47)
    seed: int | None = 42

    def total_duration(self) -> float:
        if self.duration is not None:
            return self.duration
        if not self.segments:
            return 3.0
        return max(s.end for s in self.segments) + DEFAULT_TAIL_SEC

    def n_frames(self) -> int:
        return int(round(self.total_duration() * self.fps))


# ================================================================ разбор

_SPLIT_RE = re.compile(r"\s*(?:,|;|\bпотом\b|\bзатем\b|\bпосле\s+этого\b|\bдалее\b|\bи\b|\bthen\b|\band\b)\s*")

_DIRS = [
    (re.compile(r"направ|вправ|\bright\b"), "yaw", +1),
    (re.compile(r"налев|влев|\bleft\b"), "yaw", -1),
    (re.compile(r"вверх|подним|запрокид|\bup\b"), "pitch", +1),
    (re.compile(r"вниз|опуст|наклон\w*\s+вниз|\bdown\b"), "pitch", -1),
    (re.compile(r"наклон\w*\s+вправо|тилт\w*\s+вправо"), "roll", +1),
    (re.compile(r"наклон\w*\s+влево|тилт\w*\s+влево"), "roll", -1),
]

_DEG_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:град\w*|deg\w*|°)")
_BARE_RE = re.compile(r"\bна\s+(\d+(?:[.,]\d+)?)(?!\s*(?:сек|с\b|мин|s\b|min))")
_SEC_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:сек\w*|секунд\w*|s\b|second\w*)")
_NEG_RE = re.compile(r"\bне\s+\w*$|\bне\b(?=[^,]{0,12}$)|\bno\b|\bwithout\b")
_HOLD_RE = re.compile(r"держ|пауз|замер|застын|\bhold\b|\bpause\b|\bfreeze\b")
# \b обязателен: без него "по-ВЕРН-и" ловится как "вернуться" и команда пропадает
_BACK_RE = re.compile(r"\bверн\w*|\bвозврат\w*|\bобратн\w*|\bисходн\w*|\bback\b|\breturn\b|\bcenter\b")


def _num(m) -> float:
    return float(m.group(1).replace(",", "."))


def split_clauses(text: str) -> list:
    """Режем на самостоятельные команды. Пустые куски выбрасываем."""
    parts = _SPLIT_RE.split(text.lower())
    return [p.strip() for p in parts if p and p.strip()]


def _clause_degrees(clause: str) -> float | None:
    """Число градусов ИЗ ЭТОЙ клаузы. Явные 'N градусов' важнее голого 'на N'."""
    m = _DEG_RE.search(clause)
    if m:
        return _num(m)
    m = _BARE_RE.search(clause)
    if m:
        return _num(m)
    return None


def _is_negated(clause: str, dir_pos: int) -> bool:
    """Есть ли 'не' непосредственно перед направлением (в пределах ~12 символов)."""
    window = clause[max(0, dir_pos - 14): dir_pos]
    return bool(re.search(r"\bне\s*$|\bno\s*$|\bnot\s*$", window))


def parse_clauses(text: str, spec: MotionSpec) -> MotionSpec:
    """Клаузы -> сегменты. Состояние позы тянется от команды к команде."""
    pose = {a: 0.0 for a in AXES}
    cursor = DEFAULT_LEAD_SEC
    last_axis = "yaw"
    low = text.lower()

    for clause in split_clauses(text):
        # --- скорость: модификатор действует только в своей клаузе
        dur = DEFAULT_TURN_SEC
        if re.search(r"медленн|плавн|slow|smooth", clause):
            dur *= 1.8
        if re.search(r"быстр|резк|рывк|fast|quick|sharp", clause):
            dur *= 0.55

        # --- явная пауза
        if _HOLD_RE.search(clause):
            m = _SEC_RE.search(clause)
            cursor += _num(m) if m else 1.0
            continue

        # --- возврат в исходное
        if _BACK_RE.search(clause):
            axis = last_axis
            if abs(pose[axis]) > 1e-6:
                spec.segments.append(Segment(axis, pose[axis], 0.0, cursor, dur))
                pose[axis] = 0.0
                cursor += dur + DEFAULT_GAP_SEC
            continue

        # --- направление + число, привязанные внутри клаузы
        deg = _clause_degrees(clause)
        hit = False
        for rx, axis, sign in _DIRS:
            m = rx.search(clause)
            if not m:
                continue
            if _is_negated(clause, m.start()):
                continue
            amount = deg if deg is not None else 30.0
            target = sign * amount
            if abs(target - pose[axis]) < 1e-6:
                continue
            spec.segments.append(Segment(axis, pose[axis], target, cursor, dur))
            pose[axis] = target
            last_axis = axis
            hit = True
        if hit:
            cursor += dur + DEFAULT_GAP_SEC

    # --- глобальные параметры берём из всего текста, они не позиционные
    for name, rate in BLINK_PRESETS.items():
        if name in low:
            spec.blink.rate_per_min = rate
    if re.search(r"разговор|говор|\btalk", low):
        spec.blink.rate_per_min = BLINK_PRESETS["talking"]
    if re.search(r"поко[йея]|спокойн|calm|\brest\b", low):
        spec.blink.rate_per_min = BLINK_PRESETS["rest"]
    if re.search(r"нервн|волну|stress|nervous", low):
        spec.blink.rate_per_min = BLINK_PRESETS["nervous"]
    if re.search(r"не морга|без морга|no blink", low):
        spec.blink.rate_per_min = 0.0

    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:морган\w*|blink\w*)\s*(?:в|/|per)\s*(?:мин\w*|min\w*)", low)
    if m:
        spec.blink.rate_per_min = _num(m)

    # длительность клипа — только если задана явно и не «съедена» паузой
    m = re.search(r"(?:всего|длительн\w*|клип\w*|ролик\w*|итого)\D{0,12}(\d+(?:[.,]\d+)?)\s*сек", low)
    if m:
        spec.duration = _num(m)

    return spec


def prompt_to_spec(text: str, spec: MotionSpec | None = None) -> MotionSpec:
    return parse_clauses(text, spec or MotionSpec())


# ============================================================== рендер

def _min_jerk(t: np.ndarray) -> np.ndarray:
    """Минимум рывка: 10t^3 - 15t^4 + 6t^5. Профиль скорости как у живого движения."""
    t = np.clip(t, 0.0, 1.0)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def render_axis(spec: MotionSpec, axis: str) -> np.ndarray:
    """Сегменты одной оси -> покадровый угол. Дельты складываются: frm[i+1] == to[i]."""
    n = spec.n_frames()
    idx = np.arange(n)
    out = np.zeros(n, dtype=np.float64)
    for s in sorted((x for x in spec.segments if x.axis == axis), key=lambda x: x.start):
        i0 = s.start * spec.fps
        span = max(1e-6, s.duration * spec.fps)
        out += (s.to - s.frm) * _min_jerk((idx - i0) / span)
    return out.astype(np.float32)


def micro_motion(spec: MotionSpec, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Низкочастотный дрейф — сумма синусов со случайными фазами."""
    n = spec.n_frames()
    t = np.arange(n) / spec.fps
    sig = np.zeros(n, dtype=np.float64)
    for f in spec.micro_freq_hz:
        sig += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) / len(spec.micro_freq_hz)
    return (sig * spec.micro_amp_deg * scale).astype(np.float32)


def blink_kernel(fps: int, b: BlinkSpec) -> np.ndarray:
    """Один цикл: eye_open от 1 до 0 и обратно. Закрытие быстрее открытия."""
    n_close = max(1, round(b.close_ms / 1000 * fps))
    n_hold = max(0, round(b.hold_ms / 1000 * fps))
    n_open = max(1, round(b.open_ms / 1000 * fps))

    t = np.linspace(0, 1, n_close + 1)[1:]
    closing = 1.0 - t ** 1.6                    # веко падает, ускоряясь

    t = np.linspace(0, 1, n_open + 1)[1:]
    opening = 1.0 - (1.0 - t) ** 2.1            # поднимается медленнее

    return np.clip(np.concatenate([closing, np.zeros(n_hold), opening]), 0.0, 1.0)


def blink_track(spec: MotionSpec, rng: np.random.Generator) -> tuple:
    n = spec.n_frames()
    track = np.ones(n, dtype=np.float32)
    b = spec.blink
    if b.rate_per_min <= 0.01:
        return track, []

    kernel = blink_kernel(spec.fps, b)
    dur = spec.total_duration()

    # каждое «событие» даёт в среднем (1 + p) морганий из-за двойных,
    # поэтому интервал между событиями растягиваем — иначе частота уедет вверх на p
    mean_interval = 60.0 / b.rate_per_min * (1.0 + b.double_prob)
    median = mean_interval / np.exp(b.sigma ** 2 / 2)

    times = []
    t = float(rng.uniform(0.15, max(0.3, min(mean_interval, dur))))
    while t < dur:
        times.append(t)
        if rng.random() < b.double_prob:
            t2 = t + float(rng.uniform(0.18, 0.42))
            if t2 < dur:
                times.append(t2)
                t = t2
        t += max(float(rng.lognormal(np.log(median), b.sigma)), b.refractory)

    for bt in times:
        i = int(round(bt * spec.fps))
        seg = kernel[: max(0, n - i)]
        if seg.size:
            track[i: i + seg.size] = np.minimum(track[i: i + seg.size], seg)
    return track, times


def artistic_tracks(
    spec: MotionSpec,
    yaw: np.ndarray,
    pitch: np.ndarray,
    rng: np.random.Generator,
    *,
    zoom: float = 0.0,
    gaze: float = 0.0,
    brow: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create restrained, deterministic artistic controls for LivePortrait."""
    n = spec.n_frames()
    t = np.arange(n, dtype=np.float32) / spec.fps
    zoom_track = 1.0 + np.clip(zoom, 0.0, 1.0) * 0.08 * _min_jerk(t / max(spec.total_duration(), 1e-6))
    phase_x, phase_y = rng.uniform(0, 2 * np.pi, size=2)
    gaze_gain = np.clip(gaze, 0.0, 1.0)
    gaze_x = gaze_gain * (-0.18 * yaw + 3.0 * np.sin(2 * np.pi * 0.23 * t + phase_x))
    gaze_y = gaze_gain * (-0.12 * pitch + 2.0 * np.sin(2 * np.pi * 0.17 * t + phase_y))
    brow_track = np.clip(brow, 0.0, 1.0) * (2.5 - 0.25 * gaze_y + 0.8 * np.sin(2 * np.pi * 0.13 * t))
    return tuple(np.asarray(x, dtype=np.float32) for x in (zoom_track, gaze_x, gaze_y, brow_track))


def generate(
    prompt: str,
    spec: MotionSpec | None = None,
    *,
    zoom: float = 0.0,
    gaze: float = 0.0,
    brow: float = 0.0,
) -> dict:
    spec = prompt_to_spec(prompt, spec)
    rng = np.random.default_rng(spec.seed)

    eye_open, blink_times = blink_track(spec, rng)
    yaw = render_axis(spec, "yaw") + micro_motion(spec, rng, 1.0)
    pitch = render_axis(spec, "pitch") + micro_motion(spec, rng, 0.6)
    roll = render_axis(spec, "roll") + micro_motion(spec, rng, 0.4)
    zoom_track, gaze_x, gaze_y, brow_track = artistic_tracks(
        spec, yaw, pitch, rng, zoom=zoom, gaze=gaze, brow=brow
    )
    return {
        "prompt": prompt,
        "fps": spec.fps,
        "duration": spec.total_duration(),
        "n_frames": spec.n_frames(),
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "zoom": zoom_track,
        "gaze_x": gaze_x,
        "gaze_y": gaze_y,
        "brow": brow_track,
        "eye_open": eye_open,
        "blink_times": blink_times,
        "segments": [asdict(s) for s in spec.segments],
        "blink": asdict(spec.blink),
    }


def to_json(track: dict, path: str) -> None:
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in track.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


# =============================================================== проверка

def _show(prompt: str) -> None:
    tr = generate(prompt)
    print("промт: %s" % prompt)
    if not tr["segments"]:
        print("   (движения нет)")
    for s in tr["segments"]:
        print("   %4.2f-%4.2f  %-5s  %+6.1f -> %+6.1f" % (
            s["start"], s["start"] + s["duration"], s["axis"], s["frm"], s["to"]))
    print("   итого %.2f с, %d кадров, морганий %d" % (
        tr["duration"], tr["n_frames"], len(tr["blink_times"])))
    print()


def _selftest() -> None:
    print("=" * 72)
    print("РАЗБОР: КЛАУЗЫ, ПРИВЯЗКА ЧИСЛА, ОТРИЦАНИЕ, КОМПОЗИЦИЯ")
    print("=" * 72)
    print()
    for p in [
        "поворот головы направо на 30 градусов",
        "поверни налево на 20 градусов, потом направо на 40",
        "наклон 10 градусов вниз и поворот 40 градусов вправо",
        "поверни не направо, а налево на 25 градусов",
        "смотри прямо, голову не поворачивай",
        "направо на 30, подержать 2 секунды, вернуться",
        "быстро направо на 45, потом медленно налево на 15",
    ]:
        _show(p)

    print("=" * 72)
    print("МОРГАНИЕ")
    print("=" * 72)
    print()
    b = BlinkSpec()
    k = blink_kernel(30, b)
    print("цикл: %d кадров = %.0f мс (закрытие %d, удержание %d, открытие %d)" % (
        len(k), len(k) / 30 * 1000,
        round(b.close_ms / 1000 * 30), round(b.hold_ms / 1000 * 30), round(b.open_ms / 1000 * 30)))
    print()
    print("частота, 200 прогонов по 120 с:")
    for name, rate in BLINK_PRESETS.items():
        counts = []
        for s in range(200):
            sp = MotionSpec(duration=120.0, blink=BlinkSpec(rate_per_min=rate), seed=s)
            counts.append(len(blink_track(sp, np.random.default_rng(s))[1]) / 2.0)
        m = float(np.mean(counts))
        print("  %-8s цель %5.1f  факт %5.2f/мин  откл %+5.1f%%" % (
            name, rate, m, (m - rate) / rate * 100))

    print()
    print("=" * 72)
    print("КИНЕМАТИКА")
    print("=" * 72)
    tr = generate("направо на 30 градусов")
    v = np.abs(np.diff(tr["yaw"])) * tr["fps"]
    print("пик угловой скорости: %.0f град/с (норма для человека 100-150)" % v.max())
    print()
    print("профиль eye_open, первые 3 с:")
    seg = tr["eye_open"][: 3 * tr["fps"]]
    for lvl, ch in ((0.85, "открыт "), (0.50, "полу   "), (0.15, "закрыт ")):
        print("  " + ch + "".join("#" if v <= lvl else "." for v in seg))
    print()
    print("OK")


if __name__ == "__main__":
    _selftest()
