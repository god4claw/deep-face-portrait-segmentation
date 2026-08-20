# deep-randy-face

Local portrait-animation workflow built around LivePortrait. It turns a still
image and a compact motion prompt into a normal MP4 file. It does not implement
virtual-camera injection, identity spoofing, or liveness/KYC bypass.

## Current workflow

```text
image upload -> prompt tracks -> cached LivePortrait source state
             -> GPU frame render -> 256 preview or 720p pasteback
             -> h264_nvenc (libx264 fallback) -> MP4
```

The FastAPI service has a single-worker GPU queue. LivePortrait models are
loaded lazily once per server process, and prepared source tensors are kept in a
four-entry LRU cache.

## Setup

Use the Python environment in which LivePortrait and its weights already work.
Install this repository's web/test dependencies into that same environment:

```powershell
python -m pip install -r requirements.txt
$env:LIVEPORTRAIT_ROOT = "C:\path\to\LivePortrait"
```

Weights stay in the external LivePortrait checkout and are excluded from git.

## Web UI

```powershell
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The UI provides image upload, prompt parsing,
preview/final modes, zoom/gaze/brow controls, progress, and MP4 playback.

API flow:

1. `POST /api/upload`
2. `POST /api/parse`
3. `POST /api/render`
4. Poll `GET /api/job/{job_id}`
5. Read `GET /api/video/{job_id}`

Uploads are decoded and normalized by Pillow, limited to 10 MB and 24 MP, and
removed after 24 hours. At most four queued/running jobs are accepted.

## CLI

```powershell
python render.py --src .\portrait.jpg `
  --prompt "поворот направо на 20 градусов" `
  --zoom 0.2 --gaze 0.2 --brow 0.2 `
  --out .\out\portrait.mp4
```

Add `--preview` for cropped 256 output or `--no-nvenc` to force `libx264`.

## Verification

```powershell
python -m pytest -q
python tools\verify_video.py --video .\out\portrait.mp4 `
  --source .\portrait.jpg --evidence-dir .\evidence\manual `
  --expected-frames 55
```

The semantic verifier requires an exact frame count, full decode, measurable
movement, bounded first-frame drift, and stable pasteback borders. Evidence and
generated media are local-only and ignored by git.
