import io

from fastapi.testclient import TestClient
from PIL import Image

from server import app


client = TestClient(app)


def image_bytes(size=(256, 256)):
    stream = io.BytesIO()
    Image.new("RGB", size, (80, 120, 160)).save(stream, "JPEG")
    return stream.getvalue()


def test_upload_and_parse_entry_paths():
    uploaded = client.post("/api/upload", files={"photo": ("face.jpg", image_bytes(), "image/jpeg")})
    assert uploaded.status_code == 200
    assert len(uploaded.json()["photo_id"]) == 64
    parsed = client.post("/api/parse", json={"prompt": "поворот направо на 20 градусов"})
    assert parsed.status_code == 200
    assert parsed.json()["segments"][0]["axis"] == "yaw"


def test_invalid_upload_and_render_mode_fail_closed():
    invalid = client.post("/api/upload", files={"photo": ("bad.jpg", b"not an image", "image/jpeg")})
    assert invalid.status_code == 400
    render = client.post("/api/render", data={"photo_id": "0" * 64, "prompt": "test", "mode": "typo"})
    assert render.status_code == 400


def test_missing_photo_does_not_queue_job():
    render = client.post("/api/render", data={"photo_id": "0" * 64, "prompt": "test", "mode": "preview"})
    assert render.status_code == 404


def test_photo_id_rejects_path_syntax():
    render = client.post("/api/render", data={"photo_id": "../portrait", "prompt": "test", "mode": "preview"})
    assert render.status_code == 400
