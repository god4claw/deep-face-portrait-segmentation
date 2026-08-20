# deep-randy-face

Local portrait animation baseline: motion tracks -> MP4 preview, with an optional
adapter for an existing LivePortrait checkout. This repository is intentionally a
local media workflow; it does not implement camera injection, identity spoofing,
or liveness/KYC bypass.

## Quick start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python render.py --src .\portrait.jpg --prompt "поворот направо на 20 градусов" --out .\out\preview.mp4
```

The preview renderer uses the motion generator and FFmpeg to produce a normal
MP4. It is deterministic for a fixed prompt and seed.

## Web UI

```powershell
python -m uvicorn server:app --reload
```

Open `http://127.0.0.1:8000`. Uploaded files and outputs are kept in ignored
local directories (`uploads/`, `out/`).

## Optional LivePortrait backend

Keep model code and weights outside this repository, then provide a checkout and
a driving video explicitly:

```powershell
python render.py --src .\portrait.jpg --driving .\driving.mp4 `
  --liveportrait-root C:\path\to\LivePortrait --out .\out\liveportrait.mp4
```

The adapter delegates to LivePortrait's public `inference.py` CLI. It does not
download weights, alter browser/device APIs, or create a virtual camera feed.

## Tests

```powershell
python -m pytest -q
```
