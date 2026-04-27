from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import AudioClip, Creator, get_session
from app.pipeline.setup import detect_models, gpu_info, read_status, run_setup

router = APIRouter()


class SystemStatus(BaseModel):
    backend_ok: bool
    gpu: dict
    models: dict
    storage: dict
    counts: dict
    setup: dict


@router.get("/status", response_model=SystemStatus)
def status(db: Session = Depends(get_session)) -> SystemStatus:
    n_creators = db.execute(select(func.count()).select_from(Creator)).scalar_one()
    n_clips = db.execute(select(func.count()).select_from(AudioClip)).scalar_one()
    n_refs = db.execute(
        select(func.count()).select_from(AudioClip).where(AudioClip.is_reference.is_(True))
    ).scalar_one()

    return SystemStatus(
        backend_ok=True,
        gpu=gpu_info(),
        models=detect_models(),
        storage={
            "data_dir": str(settings.DATA_DIR.resolve()),
            "raw_dir": str((settings.DATA_DIR / "raw").resolve()),
            "processed_dir": str((settings.DATA_DIR / "processed").resolve()),
            "loras_dir": str((settings.DATA_DIR / "loras").resolve()),
            "outputs_dir": str((settings.DATA_DIR / "outputs").resolve()),
        },
        counts={
            "creators": int(n_creators),
            "clips": int(n_clips),
            "reference_clips": int(n_refs),
        },
        setup=read_status(),
    )


@router.post("/setup", status_code=202)
def trigger_setup(bg: BackgroundTasks) -> dict:
    current = read_status()
    if current.get("status") == "running":
        return {"status": "already_running"}
    bg.add_task(run_setup)
    return {"status": "started"}


@router.get("/setup/status")
def setup_status() -> dict:
    return read_status()
