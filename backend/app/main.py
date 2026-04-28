import json
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from app.api import creators, datasets, system, training, tts
from app.core.config import settings
from app.db import SessionLocal, TrainingJob, init_db


def _reset_stale_jobs() -> None:
    with SessionLocal() as db:
        stale = db.execute(
            select(TrainingJob).where(TrainingJob.status.in_(["running", "pending"]))
        ).scalars().all()
        for j in stale:
            j.status = "failed"
            j.log = (j.log or "") + "\n[backend restart] job marked failed — subprocess no longer running"
            j.finished_at = datetime.utcnow()
        if stale:
            db.commit()
            print(f"[startup] reset {len(stale)} stale training jobs")

    for status_path in (settings.DATA_DIR / "processed").glob("*/_status.json"):
        try:
            data = json.loads(status_path.read_text())
            if data.get("status") == "running":
                data["status"] = "failed"
                data["log"] = (data.get("log", "") or "") + "\n[backend restart] preprocess marked failed"
                status_path.write_text(json.dumps(data))
                print(f"[startup] reset stale preprocess: {status_path.parent.name}")
        except Exception as e:
            print(f"[startup] could not check {status_path}: {e}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    _reset_stale_jobs()
    yield


app = FastAPI(title="VoiceLab API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(creators.router, prefix="/api/creators", tags=["creators"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(tts.router, prefix="/api/tts", tags=["tts"])

app.mount("/files", StaticFiles(directory=str(settings.DATA_DIR)), name="files")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
