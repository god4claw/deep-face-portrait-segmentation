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
