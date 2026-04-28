import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import AudioClip, Creator, get_session
from app.pipeline.preprocess import preprocess_creator, read_status as read_preprocess_status

router = APIRouter()


class ClipOut(BaseModel):
    id: int
    path: str
    text: str
    emotion: str | None
    duration: float
    is_reference: bool


class RawFileOut(BaseModel):
    name: str
    size: int


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


@router.get("/{creator_id}/raw", response_model=list[RawFileOut])
def list_raw(creator_id: str, db: Session = Depends(get_session)) -> list[RawFileOut]:
    creator = db.get(Creator, creator_id)
    if not creator:
        raise HTTPException(404, "Creator not found")
    raw_dir = settings.DATA_DIR / "raw" / creator_id
    if not raw_dir.exists():
        return []
    out: list[RawFileOut] = []
    for p in sorted(raw_dir.iterdir()):
        if p.is_file():
            out.append(RawFileOut(name=p.name, size=p.stat().st_size))
    return out


@router.delete("/{creator_id}/raw/{filename}", status_code=204)
def delete_raw(creator_id: str, filename: str, db: Session = Depends(get_session)) -> None:
    creator = db.get(Creator, creator_id)
    if not creator:
        raise HTTPException(404, "Creator not found")
    raw_dir = settings.DATA_DIR / "raw" / creator_id
    target = raw_dir / Path(filename).name
    if not target.is_file() or target.parent.resolve() != raw_dir.resolve():
        raise HTTPException(404, "File not found")
    target.unlink()


@router.get("/{creator_id}/preprocess/status")
def preprocess_status(creator_id: str, db: Session = Depends(get_session)) -> dict:
    if not db.get(Creator, creator_id):
        raise HTTPException(404, "Creator not found")
    return read_preprocess_status(creator_id)


@router.post("/{creator_id}/preprocess/reset")
def reset_preprocess_status(creator_id: str, db: Session = Depends(get_session)) -> dict:
    if not db.get(Creator, creator_id):
        raise HTTPException(404, "Creator not found")
    import json as _json
    status_path = settings.DATA_DIR / "processed" / creator_id / "_status.json"
    if status_path.exists():
        try:
            data = _json.loads(status_path.read_text())
            if data.get("status") == "running":
                data["status"] = "failed"
                data["log"] = (data.get("log", "") or "") + "\n[manual reset]"
                status_path.write_text(_json.dumps(data))
        except Exception:
            status_path.unlink(missing_ok=True)
    return read_preprocess_status(creator_id)


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
