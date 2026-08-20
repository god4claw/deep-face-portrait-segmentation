import numpy as np

from motion import generate


def test_generation_is_deterministic():
    a = generate("поворот направо на 30 градусов")
    b = generate("поворот направо на 30 градусов")
    np.testing.assert_array_equal(a["yaw"], b["yaw"])
    assert a["n_frames"] == len(a["eye_open"])


def test_tracks_have_expected_ranges():
    track = generate("поворот налево на 20 градусов")
    assert np.isfinite(track["yaw"]).all()
    assert ((track["eye_open"] >= 0) & (track["eye_open"] <= 1)).all()

