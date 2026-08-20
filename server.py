from __future__ import annotations

import hashlib
import io
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from motion import MotionSpec, generate, prompt_to_spec
from render import PortraitRenderer

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
OUT = ROOT / "out"
LIVEPORTRAIT_ROOT = Path(os.environ.get("LIVEPORTRAIT_ROOT", r"C:\Users\windows11\crazyrandypipe\LivePortrait"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
FILE_TTL_SECONDS = 24 * 60 * 60
UPLOADS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="portrait-gpu")
jobs: dict[str, dict] = {}
jobs_lock = threading.RLock()
renderer_lock = threading.Lock()
renderer: PortraitRenderer | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    cleanup_old_files()
    yield
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Deep Randy Face local animation", version="0.2.0", lifespan=lifespan)


class ParseRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)


def validate_image_bytes(data: bytes) -> tuple[Image.Image, str]:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"image must be between 1 byte and {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("invalid or unsupported image") from exc
    if image.width < 128 or image.height < 128:
        raise ValueError("image must be at least 128x128")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("image dimensions are too large")
    digest = hashlib.sha256(data).hexdigest()
    return image, digest


def cleanup_old_files(now: float | None = None) -> None:
    cutoff = (now or time.time()) - FILE_TTL_SECONDS
    for folder in (UPLOADS, OUT):
        for path in folder.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
            except FileNotFoundError:
                pass
    with jobs_lock:
        stale = [job_id for job_id, job in jobs.items()
                 if job.get("finished_at", job["created_at"]) < cutoff]
        for job_id in stale:
            jobs.pop(job_id, None)


def valid_hex_id(value: str, length: int) -> bool:
    return len(value) == length and all(ch in "0123456789abcdef" for ch in value)


def get_renderer() -> PortraitRenderer:
    global renderer
    if renderer is None:
        with renderer_lock:
            if renderer is None:
                renderer = PortraitRenderer(LIVEPORTRAIT_ROOT)
    return renderer


def set_job(job_id: str, **changes) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(changes)


def run_job(job_id: str, photo_path: Path, prompt: str, mode: str,
            zoom: float, gaze: float, brow: float, prefer_nvenc: bool) -> None:
    try:
        set_job(job_id, status="running", stage="loading models", progress=0.01)
        fps = 15 if mode == "preview" else 30
        track = generate(prompt, MotionSpec(fps=fps), zoom=zoom, gaze=gaze, brow=brow)
        output = OUT / f"{job_id}.mp4"

        def on_progress(value: float, stage: str) -> None:
            set_job(job_id, progress=round(value, 4), stage=stage)

        meta = get_renderer().render(track, photo_path, output, preview=mode == "preview",
                                     prefer_nvenc=prefer_nvenc, progress=on_progress)
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("renderer returned without a non-empty MP4")
        set_job(job_id, status="complete", progress=1.0, stage="complete",
                video_url=f"/api/video/{job_id}", metadata=meta, finished_at=time.time())
    except Exception as exc:
        set_job(job_id, status="failed", stage="failed", error=str(exc), finished_at=time.time())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "ui" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    with jobs_lock:
        active = sum(job["status"] in {"queued", "running"} for job in jobs.values())
    return {"status": "ok", "model_loaded": renderer is not None, "active_jobs": active}


@app.post("/api/upload")
async def upload(photo: UploadFile = File(...)):
    cleanup_old_files()
    data = await photo.read(MAX_UPLOAD_BYTES + 1)
    try:
        image, photo_id = validate_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    path = UPLOADS / f"{photo_id}.jpg"
    if not path.exists():
        image.save(path, "JPEG", quality=95, optimize=True)
    return {"photo_id": photo_id, "width": image.width, "height": image.height,
            "preview_url": f"/api/photo/{photo_id}"}


@app.get("/api/photo/{photo_id}")
def photo(photo_id: str):
    if not valid_hex_id(photo_id, 64):
        raise HTTPException(404, "photo not found")
    path = UPLOADS / f"{photo_id}.jpg"
    if not path.exists():
        raise HTTPException(404, "photo not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/parse")
def parse_prompt(request: ParseRequest):
    spec = prompt_to_spec(request.prompt)
    return {"segments": [segment.__dict__ for segment in spec.segments],
            "duration": spec.total_duration(), "n_frames": spec.n_frames(), "fps": spec.fps}


@app.post("/api/render")
def create_render(photo_id: str = Form(...), prompt: str = Form(...), mode: str = Form("preview"),
                  zoom: float = Form(0.0), gaze: float = Form(0.0), brow: float = Form(0.0),
                  prefer_nvenc: bool = Form(True)):
    if mode not in {"preview", "final"}:
        raise HTTPException(400, "mode must be preview or final")
    if not prompt.strip() or len(prompt) > 500:
        raise HTTPException(400, "prompt must be 1-500 characters")
    if any(value < 0 or value > 1 for value in (zoom, gaze, brow)):
        raise HTTPException(400, "control values must be between 0 and 1")
    if not valid_hex_id(photo_id, 64):
        raise HTTPException(400, "invalid photo id")
    photo_path = UPLOADS / f"{photo_id}.jpg"
    if not photo_path.exists():
        raise HTTPException(404, "uploaded photo not found")
    with jobs_lock:
        if sum(job["status"] in {"queued", "running"} for job in jobs.values()) >= 4:
            raise HTTPException(429, "render queue is full")
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"job_id": job_id, "status": "queued", "progress": 0.0,
                        "stage": "queued", "mode": mode, "created_at": time.time()}
    executor.submit(run_job, job_id, photo_path, prompt.strip(), mode, zoom, gaze, brow, prefer_nvenc)
    return {"job_id": job_id, "status_url": f"/api/job/{job_id}"}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return dict(job)


@app.get("/api/video/{job_id}")
def video(job_id: str):
    path = OUT / f"{job_id}.mp4"
    with jobs_lock:
        job = jobs.get(job_id)
    if not path.exists() or not job or job.get("status") != "complete":
        raise HTTPException(404, "completed video not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name)
