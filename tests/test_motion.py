import numpy as np

from motion import MotionSpec, Segment, generate


def manual_spec():
    return MotionSpec(segments=[Segment(target="face", axis="yaw", frm=0, to=30, start=.6, duration=.55)])


def test_manual_generation_is_deterministic():
    a = generate(manual_spec())
    b = generate(manual_spec())
    np.testing.assert_array_equal(a["yaw"], b["yaw"])
    assert a["n_frames"] == len(a["eye_open"])
    assert a["segments"][0]["axis"] == "yaw"


def test_tracks_have_expected_ranges():
    track = generate(manual_spec())
    assert np.isfinite(track["yaw"]).all()
    assert ((track["eye_open"] >= 0) & (track["eye_open"] <= 1)).all()
    assert np.allclose(track["zoom"], 1.0)


def test_artistic_controls_are_bounded_and_effective():
    track = generate(manual_spec(), zoom=1, gaze=1, brow=1)
    assert 1.0 <= track["zoom"].min() <= track["zoom"].max() <= 1.08
    assert np.ptp(track["gaze_x"]) > 0
    assert np.ptp(track["brow"]) > 0


def test_body_segments_are_preserved_for_future_stage_but_not_face_axis():
    spec = MotionSpec(segments=[Segment(target="neck", axis="yaw", frm=0, to=10, start=0, duration=1)])
    track = generate(spec)
    assert np.allclose(track["yaw"], track["yaw"][0], atol=2.5)
    assert track["segments"][0]["target"] == "neck"


def turning_spec(duration=6.0):
    return MotionSpec(duration=duration,
                      segments=[Segment(axis="yaw", frm=0, to=25, start=1.0, duration=.55)])


def test_gaze_is_saccadic_not_smooth():
    """Взгляд должен стоять и редко прыгать. Плавная синусоида читается как кукольная."""
    gaze_x = generate(turning_spec(), gaze=1)["gaze_x"]
    speed = np.abs(np.diff(gaze_x))
    assert np.median(speed) > 0                       # не заморожен
    assert speed.max() / np.median(speed) > 20        # выражённые скачки поверх фиксаций


def test_vor_is_transient_not_permanent():
    """После поворота глаза догоняют цель, а не висят отведёнными."""
    from motion import _vor
    spec = turning_spec()
    vor = _vor(spec, generate(spec)["yaw"])
    end = int(1.55 * spec.fps)
    assert abs(vor[int(1.2 * spec.fps):end].min()) > 1.0      # во время поворота есть отклик
    assert abs(vor[end + int(2.0 * spec.fps)]) < 0.2          # через 2 с вернулся к нулю


def test_brow_rests_at_neutral():
    """База должна быть нулевой: постоянно поднятые брови выглядят как удивление."""
    brow = generate(turning_spec(), brow=1)["brow"]
    assert abs(np.median(brow)) < 0.5
    assert brow.max() > 1.0                            # но события всё же случаются


def test_zoom_segment_supports_both_directions():
    forward = generate(MotionSpec(duration=3.0, segments=[
        Segment(axis="zoom", frm=0, to=15, start=.5, duration=.8)]))["zoom"]
    backward = generate(MotionSpec(duration=3.0, segments=[
        Segment(axis="zoom", frm=0, to=-15, start=.5, duration=.8)]))["zoom"]
    assert forward[-1] > 1.1                           # наезд
    assert backward[-1] < 0.9                          # отъезд


def test_zoom_slider_accepts_negative():
    out = generate(turning_spec(duration=3.0), zoom=-1)["zoom"]
    assert out.min() < 1.0
