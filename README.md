# VoiceLab

End-to-end voice-cloning platform for creator agencies. Stack:

- **Backend** — FastAPI + SQLAlchemy + CosyVoice 2 (TTS) + PEFT/LoRA (fine-tuning)
- **Frontend** — Next.js 15 (App Router) + Tailwind v4
- **Storage** — local filesystem + SQLite (swap for S3/R2 + Postgres in prod)

## Quick start

Two terminals, one for each side.

### Backend
```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py   # clones CosyVoice + downloads weights (~2GB)
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open http://localhost:3000.

## Workflow

1. **Onboard creator** at `/creators` (capture signed consent — required before any audio touches the system).
2. **Upload + preprocess** raw audio on the creator detail page. Pipeline: Demucs denoise → 24 kHz mono → Silero VAD → loudness norm (-20 LUFS) → faster-whisper ASR.
3. **Mark reference clips** per emotion (3–5 per emotion is enough for zero-shot).
4. **(Optional) LoRA fine-tune** if you have ≥1 hour of clean transcribed audio. Adapter saves to `backend/data/loras/<creator>/`.
5. **Synthesize** at `/creators/<id>/synth` with emotion + speed control.

## Data layout

```
backend/data/
  raw/<creator_id>/           # original uploads
  processed/<creator_id>/     # 24k mono WAV clips
  loras/<creator_id>/         # PEFT adapters
  outputs/<creator_id>/       # synthesized WAVs
```

## Hardware

| Stage         | Min VRAM | Notes                                  |
|---------------|----------|----------------------------------------|
| Preprocess    | CPU ok   | ASR `large-v3` is much faster on GPU   |
| TTS inference | 6 GB     | CosyVoice 2 0.5B fp16                  |
| LoRA train    | 8–10 GB  | Rank 8 fits in 8 GB; rank 16 wants 10  |

Recommended cloud: RunPod / Modal / Lambda Labs L4 ($0.40/hr) or A10.

## Production notes

- Replace SQLite with Postgres (one URL change in `.env`)
- Swap local FS for S3/R2 (wrap `app.core.config.settings.DATA_DIR` access)
- Move background tasks from FastAPI `BackgroundTasks` to Celery/RQ
- Add per-creator quotas + auth on the API
- Tag every generated file with creator-side metadata so consent revocation can purge

## Consent + disclosure

The system enforces `consent_signed=true` on the creator before upload or synthesis at the API layer (`creators.py`, `datasets.py`, `tts.py`). Keep AI-generated voice messages disclosed to the end recipient — that's the line between voice tooling and impersonation.
