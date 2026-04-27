import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import AudioClip, Creator, get_session
from app.pipeline.preprocess import preprocess_creator

router = APIRouter()


class ClipOut(BaseModel):
    id: int
    path: str
    text: str
    emotion: str | None
    duration: float
    is_reference: bool


class ClipPatch(BaseModel):
    emotion: str | None = None
    is_reference: bool | None = None
    text: str | None = None


@router.post("/{creator_id}/upload")
def upload_audio(
    creator_id: str,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_session),
) -> dict:
    creator = db.get(Creator, creator_id)
    if not creator:
        raise HTTPException(404, "Creator not found")
    if not creator.consent_signed:
        raise HTTPException(403, "Creator consent must be signed before audio upload")

    raw_dir = settings.DATA_DIR / "raw" / creator_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for f in files:
        safe_name = Path(f.filename or "audio").name
        target = raw_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        with target.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(target.name)

    return {"saved": saved, "count": len(saved)}


@router.post("/{creator_id}/preprocess")
def trigger_preprocess(
    creator_id: str,
    bg: BackgroundTasks,
    db: Session = Depends(get_session),
) -> dict:
    creator = db.get(Creator, creator_id)
    if not creator:
        raise HTTPException(404, "Creator not found")
    bg.add_task(preprocess_creator, creator_id)
    return {"status": "started"}


@router.get("/{creator_id}/clips", response_model=list[ClipOut])
def list_clips(creator_id: str, db: Session = Depends(get_session)) -> list[ClipOut]:
    rows = db.execute(
        select(AudioClip).where(AudioClip.creator_id == creator_id).order_by(AudioClip.id)
    ).scalars().all()
    return [
        ClipOut(
            id=c.id, path=c.path, text=c.text, emotion=c.emotion,
            duration=c.duration, is_reference=c.is_reference,
        )
        for c in rows
    ]


@router.patch("/clips/{clip_id}", response_model=ClipOut)
def update_clip(
    clip_id: int,
    body: ClipPatch,
    db: Session = Depends(get_session),
) -> ClipOut:
    clip = db.get(AudioClip, clip_id)
    if not clip:
        raise HTTPException(404, "Clip not found")
    if body.emotion is not None:
        clip.emotion = body.emotion or None
    if body.is_reference is not None:
        clip.is_reference = body.is_reference
    if body.text is not None:
        clip.text = body.text
    db.commit()
    return ClipOut(
        id=clip.id, path=clip.path, text=clip.text, emotion=clip.emotion,
        duration=clip.duration, is_reference=clip.is_reference,
    )


@router.delete("/clips/{clip_id}", status_code=204)
def delete_clip(clip_id: int, db: Session = Depends(get_session)) -> None:
    clip = db.get(AudioClip, clip_id)
    if not clip:
        raise HTTPException(404, "Clip not found")
    audio_path = settings.DATA_DIR / clip.path
    audio_path.unlink(missing_ok=True)
    db.delete(clip)
    db.commit()
