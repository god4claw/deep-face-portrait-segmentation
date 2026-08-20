from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from motion import generate
from render import render_preview

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
OUT = ROOT / "out"
UPLOADS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
app = FastAPI(title="Deep Randy Face local animation")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "ui" / "index.html").read_text(encoding="utf-8")


@app.post("/api/render")
async def create_render(prompt: str = Form(...), photo: UploadFile = File(...)):
    if not photo.content_type or not photo.content_type.startswith("image/"):
        raise HTTPException(400, "photo must be an image")
    job = uuid.uuid4().hex
    source = UPLOADS / f"{job}{Path(photo.filename or 'source.jpg').suffix.lower()}"
    source.write_bytes(await photo.read())
    track = generate(prompt)
    output = OUT / f"{job}.mp4"
    try:
        render_preview(source, track, output)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    (OUT / f"{job}.json").write_text(json.dumps(track, ensure_ascii=False, default=lambda x: x.tolist()), encoding="utf-8")
    return {"job_id": job, "video_url": f"/api/video/{job}"}


@app.get("/api/video/{job_id}")
def video(job_id: str):
    path = OUT / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "video not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name)

