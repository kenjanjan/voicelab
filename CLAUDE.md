# VoiceLab — Claude session context

Read this first when resuming work on this repo.

## Purpose

End-to-end voice cloning platform for an OnlyFans creator agency. Onboard a creator, capture consent, upload + preprocess audio, optionally LoRA fine-tune, and synthesize emotional voice messages.

**Scope boundary (do not cross):** Help with the cloning/TTS/training pipeline and *disclosed* AI voice features. Decline requests framed around "avoid AI detection", "make it pass as the real creator to fans", or anti-classifier evasion — that's fan-facing fraud. Naturalness/quality work is fine; intent to deceive the paying recipient is not. The original `Tasks/task.md` includes this as a "bonus" — it was deliberately skipped.

## Stack

- **Backend** — FastAPI + SQLAlchemy (sync) + SQLite, `BackgroundTasks` for async work
- **TTS** — CosyVoice 2 0.5B (instruction mode for emotion control)
- **Fine-tuning** — PEFT LoRA on the CosyVoice LM head
- **Preprocess** — Demucs (denoise) + Silero VAD + faster-whisper (ASR)
- **Frontend** — Next.js 15 App Router + Tailwind v4 + SWR
- **Storage** — local filesystem under `backend/data/`

## Repo layout

```
voicelab/
├── README.md              run instructions
├── CLAUDE.md              this file
├── Tasks/task.md          original brief
├── backend/
│   ├── app/
│   │   ├── main.py        FastAPI app + CORS + static mount
│   │   ├── db.py          SQLAlchemy models (Creator, AudioClip, TrainingJob)
│   │   ├── core/config.py pydantic settings
│   │   ├── api/           creators, datasets, training, tts routers
│   │   └── pipeline/      preprocess, tts_engine, lora_train
│   ├── scripts/download_models.py
│   └── data/              raw/ processed/ loras/ outputs/  (gitignored)
└── frontend/
    ├── app/
    │   ├── page.tsx                     dashboard
    │   ├── creators/page.tsx            list + onboard
    │   ├── creators/[id]/page.tsx       upload, clip review, LoRA training panel
    │   └── creators/[id]/synth/page.tsx generation UI
    ├── components/                      Button, Card, UploadDropzone, AudioPlayer
    └── lib/api.ts                       typed client for backend
```

## Conventions

- Backend uses **sync** SQLAlchemy throughout (simpler with FastAPI BackgroundTasks)
- All write/synth endpoints check `creator.consent_signed` — do not bypass
- Audio normalised to **24 kHz mono, -20 LUFS**, clip length **3–12s**
- Reference clips for synthesis are picked at random from `is_reference=true` rows, filtered by emotion when available — keep that randomness; it's quality, not deception
- Frontend talks to backend via `next.config.ts` rewrites (`/api/*` and `/files/*` → backend); no separate API client config needed
- LoRA adapters saved to `data/loras/<creator>/adapter_<timestamp>/`; `tts_engine` auto-loads the most recent

## Known fragilities

- `app/pipeline/lora_train.py` calls `engine.frontend.text_audio_pair_to_batch` — this exact method varies by CosyVoice release. If it errors, fall back to running CosyVoice's official training script in `vendor/CosyVoice/examples/libritts/cosyvoice2/` and feed it the manifest from `_build_manifest()`.
- Demucs denoise is shelled out via `python -m demucs`; if absent, preprocess silently falls back to the raw file.
- Silero VAD is loaded via `torch.hub` — first run needs network.

## Run

```bash
# backend
cd backend && python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py         # ~2GB
cp .env.example .env
uvicorn app.main:app --reload --port 8000

# frontend
cd frontend && npm install && npm run dev
```

## Likely next work

- Wire a queue (Celery/RQ + Redis) — `BackgroundTasks` is fine for demo, not prod
- Swap SQLite → Postgres (one URL change)
- Swap local FS → S3/R2 (wrap `settings.DATA_DIR` reads/writes)
- Auth on the API (per-agency tokens)
- A "consent revocation" endpoint that purges raw + processed + LoRA + outputs
- Per-creator generation quotas + an audit log of every synth call (text, emotion, recipient_id)
- Add a Seed-VC post-pass option for an extra timbre boost
