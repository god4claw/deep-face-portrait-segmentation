import io
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from server import app

client = TestClient(app)


def image_bytes(size=(256, 256)):
    stream = io.BytesIO()
    Image.new("RGB", size, (80, 120, 160)).save(stream, "JPEG")
    return stream.getvalue()


def segment_data(target="face"):
    return json.dumps([{"target": target, "axis": "yaw", "frm": 0, "to": 20, "start": .6, "duration": .55}])


def test_upload_and_manual_render_contract():
    uploaded = client.post("/api/upload", files={"photo": ("face.jpg", image_bytes(), "image/jpeg")})
    assert uploaded.status_code == 200
    assert len(uploaded.json()["photo_id"]) == 64
    with patch("server.executor.submit") as submit:
        render = client.post("/api/render", data={"photo_id": uploaded.json()["photo_id"],
                                                   "segments_json": segment_data(), "mode": "preview"})
    assert render.status_code == 200
    submit.assert_called_once()


def test_parser_endpoint_is_removed():
    assert client.post("/api/parse", json={"prompt": "anything"}).status_code == 404


def test_invalid_upload_and_render_mode_fail_closed():
    invalid = client.post("/api/upload", files={"photo": ("bad.jpg", b"not an image", "image/jpeg")})
    assert invalid.status_code == 400
    render = client.post("/api/render", data={"photo_id": "0" * 64, "segments_json": "[]", "mode": "typo"})
    assert render.status_code == 400


def test_missing_photo_does_not_queue_job():
    render = client.post("/api/render", data={"photo_id": "0" * 64, "segments_json": "[]", "mode": "preview"})
    assert render.status_code == 404


def test_photo_id_rejects_path_syntax():
    render = client.post("/api/render", data={"photo_id": "../portrait", "segments_json": "[]", "mode": "preview"})
    assert render.status_code == 400


def test_invalid_manual_segment_and_framing_fail_closed():
    invalid_segment = client.post("/api/render", data={"photo_id": "0" * 64,
                                                        "segments_json": '{"axis":"yaw"}', "mode": "preview"})
    assert invalid_segment.status_code in {400, 404}
    invalid_aspect = client.post("/api/render", data={"photo_id": "0" * 64, "segments_json": "[]",
                                                       "mode": "preview", "aspect": "5:2"})
    assert invalid_aspect.status_code == 400
