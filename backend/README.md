# VoiceLab — Backend

FastAPI service for creator voice cloning with CosyVoice 2 + LoRA.

## Setup (local)

```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# clone CosyVoice + download CosyVoice2-0.5B (~2GB)
python scripts/download_models.py

cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the OpenAPI UI.

## Pipeline flow

1. `POST /api/creators` → create creator
2. `PATCH /api/creators/{id}/consent` → record signed consent
3. `POST /api/datasets/{id}/upload` → upload raw `.wav/.mp3/.m4a`
4. `POST /api/datasets/{id}/preprocess` → denoise + VAD + ASR (background)
5. `GET /api/datasets/{id}/clips` → review clips, mark `is_reference=true`, set `emotion`
6. `POST /api/training/{id}/start` → kick off LoRA fine-tune (optional)
7. `POST /api/tts` → generate a message

## GPU notes

- Preprocess runs CPU-only fine but slower; ASR (`large-v3`) wants ≥6 GB VRAM
- Inference: CosyVoice 2 0.5B fits in ~6 GB; an L4 (24 GB) handles ~10–20 concurrent creators
- LoRA training: rank 16 needs ~10 GB; rank 8 fits in 8 GB
