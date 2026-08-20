import numpy as np

from render import frame_to_canvas


def test_source_aspect_preserves_input_shape():
    frame = np.full((300, 200, 3), 255, dtype=np.uint8)
    result = frame_to_canvas(frame, "source", "contain", 120)
    assert result.shape == frame.shape


def test_contain_creates_exact_canvas_without_cropping():
    frame = np.full((300, 200, 3), 255, dtype=np.uint8)
    result = frame_to_canvas(frame, "4:3", "contain", 120)
    assert result.shape == (120, 160, 3)
    assert np.all(result[:, 0] == (16, 20, 22))
    assert np.all(result[:, 80] == 255)


def test_cover_creates_exact_canvas_without_letterbox():
    frame = np.full((300, 200, 3), 255, dtype=np.uint8)
    result = frame_to_canvas(frame, "4:3", "cover", 120)
    assert result.shape == (120, 160, 3)
    assert np.all(result == 255)
