import numpy as np

from render import apply_body_motion, frame_to_canvas


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


def test_body_motion_keeps_upper_face_source_region_stable():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    frame[:80, :, 0] = 255
    frame[80:, :, 1] = np.arange(120, dtype=np.uint8)[:, None]
    moved = apply_body_motion(frame, 12, 0, 8)
    assert np.abs(moved[:45].astype(int) - frame[:45]).mean() < 1
    assert np.abs(moved[120:].astype(int) - frame[120:]).mean() > 0
